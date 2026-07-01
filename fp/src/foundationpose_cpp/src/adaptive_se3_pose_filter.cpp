#include "adaptive_se3_pose_filter.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace foundationpose_filter
{

namespace
{

constexpr float kDefaultDt = 1.0F / 30.0F;
constexpr float kMinPositive = 1e-6F;

} // namespace

void AdaptiveSE3PoseFilter::initialize(const Eigen::Matrix4f &init_pose,
                                       float                  dt,
                                       const AdaptiveSE3PoseFilterParams &params)
{
  params_ = params;
  params_.q_pos = positiveOrDefault(params_.q_pos, AdaptiveSE3PoseFilterParams{}.q_pos);
  params_.q_rot = positiveOrDefault(params_.q_rot, AdaptiveSE3PoseFilterParams{}.q_rot);
  params_.q_vel = positiveOrDefault(params_.q_vel, AdaptiveSE3PoseFilterParams{}.q_vel);
  params_.q_omega = positiveOrDefault(params_.q_omega, AdaptiveSE3PoseFilterParams{}.q_omega);
  params_.sigma_pos_obs =
      positiveOrDefault(params_.sigma_pos_obs, AdaptiveSE3PoseFilterParams{}.sigma_pos_obs);
  params_.sigma_rot_obs =
      positiveOrDefault(params_.sigma_rot_obs, AdaptiveSE3PoseFilterParams{}.sigma_rot_obs);
  params_.sigma_pos_motion =
      positiveOrDefault(params_.sigma_pos_motion, AdaptiveSE3PoseFilterParams{}.sigma_pos_motion);
  params_.sigma_rot_motion =
      positiveOrDefault(params_.sigma_rot_motion, AdaptiveSE3PoseFilterParams{}.sigma_rot_motion);
  params_.min_quality =
      std::clamp(positiveOrDefault(params_.min_quality, AdaptiveSE3PoseFilterParams{}.min_quality),
                 kMinPositive,
                 1.0F);
  params_.gate_threshold =
      positiveOrDefault(params_.gate_threshold, AdaptiveSE3PoseFilterParams{}.gate_threshold);
  params_.reject_cov_increase = std::max(0.0F, params_.reject_cov_increase);

  T_ = init_pose;
  orthonormalizeSE3(&T_);
  xi_.setZero();

  P_.setZero();
  P_.diagonal().segment<3>(0).setConstant(params_.sigma_pos_obs * params_.sigma_pos_obs);
  P_.diagonal().segment<3>(3).setConstant(params_.sigma_rot_obs * params_.sigma_rot_obs);
  P_.diagonal().segment<3>(6).setConstant(params_.q_vel);
  P_.diagonal().segment<3>(9).setConstant(params_.q_omega);

  T_pred_ = T_;
  xi_pred_ = xi_;
  P_pred_ = P_;
  last_dt_ = sanitizeDt(dt);
  last_quality_ = 1.0F;
  last_mahalanobis_distance_ = 0.0F;
  last_observation_accepted_ = true;
  last_translation_residual_norm_ = 0.0F;
  last_rotation_residual_norm_ = 0.0F;
  initialized_ = true;
  has_prediction_ = true;
}

Eigen::Matrix4f AdaptiveSE3PoseFilter::predict(float dt)
{
  if (!initialized_)
  {
    return T_;
  }

  last_dt_ = sanitizeDt(dt);
  T_pred_ = se3Exp(xi_ * last_dt_) * T_;
  orthonormalizeSE3(&T_pred_);
  xi_pred_ = xi_;

  Matrix12f F = Matrix12f::Identity();
  F.block<6, 6>(0, 6) = Matrix6f::Identity() * last_dt_;
  P_pred_ = F * P_ * F.transpose() + buildProcessNoise(last_dt_);
  symmetrize(&P_pred_);

  has_prediction_ = true;
  return T_pred_;
}

Eigen::Matrix4f AdaptiveSE3PoseFilter::update(const Eigen::Matrix4f &T_obs)
{
  if (!initialized_)
  {
    return T_;
  }

  if (!has_prediction_)
  {
    predict(last_dt_);
  }

  Eigen::Matrix4f T_obs_orth = T_obs;
  orthonormalizeSE3(&T_obs_orth);

  const Vector6f residual = se3Log(inverseSE3(T_pred_) * T_obs_orth);
  last_translation_residual_norm_ = residual.head<3>().norm();
  last_rotation_residual_norm_ = residual.tail<3>().norm();

  const float pos_sigma = positiveOrDefault(params_.sigma_pos_motion,
                                           AdaptiveSE3PoseFilterParams{}.sigma_pos_motion);
  const float rot_sigma = positiveOrDefault(params_.sigma_rot_motion,
                                           AdaptiveSE3PoseFilterParams{}.sigma_rot_motion);
  last_quality_ = std::exp(-last_translation_residual_norm_ / pos_sigma) *
                  std::exp(-last_rotation_residual_norm_ / rot_sigma);
  last_quality_ = std::clamp(last_quality_, 0.0F, 1.0F);

  const Matrix6f R = buildObservationNoise(last_quality_);

  Eigen::Matrix<float, 6, 12> H = Eigen::Matrix<float, 6, 12>::Zero();
  H.block<6, 6>(0, 0) = Matrix6f::Identity();

  const Matrix6f S = H * P_pred_ * H.transpose() + R;
  Eigen::LDLT<Matrix6f> ldlt(S);
  if (ldlt.info() != Eigen::Success)
  {
    T_ = T_pred_;
    xi_ = xi_pred_;
    P_ = P_pred_ + Matrix12f::Identity() * params_.reject_cov_increase;
    symmetrize(&P_);
    last_mahalanobis_distance_ = std::numeric_limits<float>::infinity();
    last_observation_accepted_ = false;
    has_prediction_ = false;
    return T_;
  }

  const Vector6f S_inv_residual = ldlt.solve(residual);
  last_mahalanobis_distance_ = residual.dot(S_inv_residual);

  if (params_.enable_gating && last_mahalanobis_distance_ > params_.gate_threshold)
  {
    T_ = T_pred_;
    xi_ = xi_pred_;
    P_ = P_pred_ + Matrix12f::Identity() * params_.reject_cov_increase;
    symmetrize(&P_);
    last_observation_accepted_ = false;
    has_prediction_ = false;
    return T_;
  }

  const Matrix6f S_inv = ldlt.solve(Matrix6f::Identity());
  const Eigen::Matrix<float, 12, 6> K = P_pred_ * H.transpose() * S_inv;
  const Vector12f delta_x = K * residual;

  T_ = se3Exp(delta_x.head<6>()) * T_pred_;
  orthonormalizeSE3(&T_);
  xi_ = xi_pred_ + delta_x.tail<6>();

  const Matrix12f I12 = Matrix12f::Identity();
  const Matrix12f KH = K * H;
  P_ = (I12 - KH) * P_pred_ * (I12 - KH).transpose() + K * R * K.transpose();
  symmetrize(&P_);

  last_observation_accepted_ = true;
  has_prediction_ = false;
  return T_;
}

Eigen::Matrix4f AdaptiveSE3PoseFilter::rejectObservation()
{
  if (!initialized_)
  {
    return T_;
  }

  if (!has_prediction_)
  {
    predict(last_dt_);
  }

  T_ = T_pred_;
  xi_ = xi_pred_;
  P_ = P_pred_ + Matrix12f::Identity() * params_.reject_cov_increase;
  symmetrize(&P_);

  last_quality_ = 0.0F;
  last_mahalanobis_distance_ = std::numeric_limits<float>::infinity();
  last_observation_accepted_ = false;
  last_translation_residual_norm_ = 0.0F;
  last_rotation_residual_norm_ = 0.0F;
  has_prediction_ = false;
  return T_;
}

const Eigen::Matrix4f &AdaptiveSE3PoseFilter::getPose() const
{
  return T_;
}

float AdaptiveSE3PoseFilter::getLastQuality() const
{
  return last_quality_;
}

float AdaptiveSE3PoseFilter::getLastMahalanobisDistance() const
{
  return last_mahalanobis_distance_;
}

bool AdaptiveSE3PoseFilter::wasLastObservationAccepted() const
{
  return last_observation_accepted_;
}

float AdaptiveSE3PoseFilter::getLastTranslationResidualNorm() const
{
  return last_translation_residual_norm_;
}

float AdaptiveSE3PoseFilter::getLastRotationResidualNorm() const
{
  return last_rotation_residual_norm_;
}

bool AdaptiveSE3PoseFilter::isInitialized() const
{
  return initialized_;
}

Matrix12f AdaptiveSE3PoseFilter::buildProcessNoise(float dt) const
{
  const float safe_dt = sanitizeDt(dt);
  Matrix12f Q = Matrix12f::Zero();
  Q.diagonal().segment<3>(0).setConstant(params_.q_pos * safe_dt);
  Q.diagonal().segment<3>(3).setConstant(params_.q_rot * safe_dt);
  Q.diagonal().segment<3>(6).setConstant(params_.q_vel * safe_dt);
  Q.diagonal().segment<3>(9).setConstant(params_.q_omega * safe_dt);
  return Q;
}

Matrix6f AdaptiveSE3PoseFilter::buildObservationNoise(float quality) const
{
  Matrix6f R = Matrix6f::Zero();
  R.diagonal().segment<3>(0).setConstant(params_.sigma_pos_obs * params_.sigma_pos_obs);
  R.diagonal().segment<3>(3).setConstant(params_.sigma_rot_obs * params_.sigma_rot_obs);

  if (params_.enable_adaptive_R)
  {
    const float scale = 1.0F / std::max(quality, params_.min_quality);
    R *= scale;
  }
  return R;
}

float AdaptiveSE3PoseFilter::sanitizeDt(float dt)
{
  if (!std::isfinite(dt) || dt <= 0.0F)
  {
    return kDefaultDt;
  }
  return std::clamp(dt, 1e-4F, 1.0F);
}

float AdaptiveSE3PoseFilter::positiveOrDefault(float value, float fallback)
{
  if (!std::isfinite(value) || value <= 0.0F)
  {
    return fallback;
  }
  return value;
}

void AdaptiveSE3PoseFilter::symmetrize(Matrix12f *matrix)
{
  if (matrix == nullptr)
  {
    return;
  }

  *matrix = 0.5F * (*matrix + matrix->transpose());
}

} // namespace foundationpose_filter
