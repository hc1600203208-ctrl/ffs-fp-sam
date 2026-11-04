#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// Port of FoundationPoseROS2/FoundationPose/tracking/adaptive_se3_pose_filter.py.
// Poses are T_cam_mesh; twists are right-invariant, body-frame [translation, rotation].
namespace adaptive_pose
{
struct Params
{
  double q_pos{1e-4}, q_rot{1e-3}, q_vel{1e-2}, q_omega{1e-1};
  double sigma_pos_obs{0.005}, sigma_rot_obs{0.02};
  double sigma_pos_motion{0.05}, sigma_rot_motion{0.7};
  double min_quality{0.01}, gate_threshold{100.0}, reject_cov_increase{1e-4};
  bool enable_gating{true}, enable_adaptive_R{true};
  double fps{20.0};
};

struct Debug
{
  double quality{1.0}, mahalanobis{0.0}, trans_residual{0.0}, rot_residual{0.0};
  bool accepted{true};
};

using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector12 = Eigen::Matrix<double, 12, 1>;
using Matrix12 = Eigen::Matrix<double, 12, 12>;

inline Eigen::Matrix3d Skew(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d result;
  result << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return result;
}

inline Eigen::Matrix4d NormalizePose(Eigen::Matrix4d pose)
{
  Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
  q.normalize();
  pose.block<3, 3>(0, 0) = q.toRotationMatrix();
  pose.row(3) << 0, 0, 0, 1;
  return pose;
}

inline Eigen::Matrix4d Exp(const Vector6 &xi)
{
  const Eigen::Vector3d rho = xi.head<3>(), phi = xi.tail<3>();
  const double theta = phi.norm();
  const Eigen::Matrix3d phi_skew = Skew(phi);
  const Eigen::Matrix3d square = phi_skew * phi_skew;
  Eigen::Matrix3d rotation, V;
  if (theta < 1e-8)
  {
    rotation = Eigen::Matrix3d::Identity() + phi_skew;
    V = Eigen::Matrix3d::Identity() + 0.5 * phi_skew;
  }
  else
  {
    rotation = Eigen::Matrix3d::Identity() + std::sin(theta) / theta * phi_skew +
               (1.0 - std::cos(theta)) / (theta * theta) * square;
    V = Eigen::Matrix3d::Identity() + (1.0 - std::cos(theta)) / (theta * theta) * phi_skew +
        (theta - std::sin(theta)) / (theta * theta * theta) * square;
  }
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3, 3>(0, 0) = rotation;
  pose.block<3, 1>(0, 3) = V * rho;
  return pose;
}

inline Vector6 Log(const Eigen::Matrix4d &pose)
{
  const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
  const double theta = std::acos(std::clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0));
  Eigen::Vector3d phi = Eigen::Vector3d::Zero();
  if (theta >= 1e-8)
  {
    // AngleAxis avoids division by sin(theta) at a half turn.
    const Eigen::AngleAxisd angle_axis(rotation);
    phi = angle_axis.angle() * angle_axis.axis();
  }
  const Eigen::Matrix3d phi_skew = Skew(phi);
  Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity() - 0.5 * phi_skew;
  if (theta >= 1e-8)
  {
    V_inv += (1.0 - 0.5 * theta / std::tan(0.5 * theta)) /
             (theta * theta) * phi_skew * phi_skew;
  }
  Vector6 xi;
  xi.head<3>() = V_inv * pose.block<3, 1>(0, 3);
  xi.tail<3>() = phi;
  return xi;
}

class Filter
{
public:
  explicit Filter(Params params = {}) : params_(params) {}

  void Initialize(const Eigen::Matrix4f &pose)
  {
    pose_ = NormalizePose(pose.cast<double>());
    predicted_pose_ = pose_;
    velocity_.setZero();
    predicted_velocity_.setZero();
    covariance_ = Matrix12::Identity() * 1e-3;
    predicted_covariance_ = covariance_;
    debug_ = {};
    initialized_ = true;
  }

  bool initialized() const { return initialized_; }
  Eigen::Matrix4f pose() const { return pose_.cast<float>(); }
  const Debug &debug() const { return debug_; }

  Eigen::Matrix4f Predict(double dt)
  {
    if (!initialized_) return Eigen::Matrix4f::Identity();
    dt_ = (std::isfinite(dt) && dt > 0.0 && dt <= 1.0) ? dt : 1.0 / std::max(params_.fps, 1e-6);
    predicted_pose_ = NormalizePose(pose_ * Exp(velocity_ * dt_));
    predicted_velocity_ = velocity_;
    Matrix12 F = Matrix12::Identity();
    F.block<6, 6>(0, 6) = Matrix6::Identity() * dt_;
    Matrix12 Q = Matrix12::Zero();
    Q.diagonal() << params_.q_pos, params_.q_pos, params_.q_pos,
        params_.q_rot, params_.q_rot, params_.q_rot,
        params_.q_vel, params_.q_vel, params_.q_vel,
        params_.q_omega, params_.q_omega, params_.q_omega;
    predicted_covariance_ = F * covariance_ * F.transpose() + Q;
    return predicted_pose_.cast<float>();
  }

  Eigen::Matrix4f Update(const Eigen::Matrix4f &observation)
  {
    if (!initialized_) { Initialize(observation); return pose(); }
    if (!observation.allFinite()) return CommitPrediction();
    const Eigen::Matrix4d observed = NormalizePose(observation.cast<double>());
    const Vector6 residual = Log(predicted_pose_.inverse() * observed);
    debug_.trans_residual = residual.head<3>().norm();
    debug_.rot_residual = residual.tail<3>().norm();
    debug_.quality = std::clamp(
        std::exp(-debug_.trans_residual / std::max(params_.sigma_pos_motion, 1e-6) -
                 debug_.rot_residual / std::max(params_.sigma_rot_motion, 1e-6)),
        params_.min_quality, 1.0);
    const double pos_var = params_.sigma_pos_obs * params_.sigma_pos_obs;
    const double rot_var = params_.sigma_rot_obs * params_.sigma_rot_obs;
    const double scale = params_.enable_adaptive_R ? 1.0 / std::max(debug_.quality, params_.min_quality) : 1.0;
    Matrix6 R = Matrix6::Zero();
    R.diagonal() << std::clamp(pos_var * scale, pos_var * 0.5, pos_var * 100.0),
        std::clamp(pos_var * scale, pos_var * 0.5, pos_var * 100.0),
        std::clamp(pos_var * scale, pos_var * 0.5, pos_var * 100.0),
        std::clamp(rot_var * scale, rot_var * 0.5, rot_var * 100.0),
        std::clamp(rot_var * scale, rot_var * 0.5, rot_var * 100.0),
        std::clamp(rot_var * scale, rot_var * 0.5, rot_var * 100.0);
    Matrix6 S = predicted_covariance_.block<6, 6>(0, 0) + R;
    Eigen::LDLT<Matrix6> solver(S);
    if (solver.info() != Eigen::Success)
    {
      S.diagonal().array() += 1e-6;
      solver.compute(S);
    }
    if (solver.info() != Eigen::Success) { CommitPrediction(); return pose(); }
    debug_.mahalanobis = residual.dot(solver.solve(residual));
    if (!std::isfinite(debug_.mahalanobis) ||
        (params_.enable_gating && debug_.mahalanobis > params_.gate_threshold))
    {
      pose_ = predicted_pose_;
      velocity_ = predicted_velocity_;
      covariance_ = predicted_covariance_ + Matrix12::Identity() * params_.reject_cov_increase;
      debug_.accepted = false;
      return pose();
    }
    const Eigen::Matrix<double, 12, 6> K = solver.solve(predicted_covariance_.block<6, 12>(0, 0)).transpose();
    const Vector12 delta = K * residual;
    pose_ = NormalizePose(predicted_pose_ * Exp(delta.head<6>()));
    velocity_ = predicted_velocity_ + delta.tail<6>();
    Matrix12 A = Matrix12::Identity();
    A.block<12, 6>(0, 0) -= K;
    covariance_ = A * predicted_covariance_ * A.transpose() + K * R * K.transpose();
    debug_.accepted = true;
    return pose();
  }

  // Failed FoundationPose tracking: retain exactly the Python prediction fallback.
  Eigen::Matrix4f CommitPrediction()
  {
    pose_ = predicted_pose_;
    velocity_ = predicted_velocity_;
    covariance_ = predicted_covariance_;
    debug_.quality = 0.0;
    debug_.mahalanobis = std::numeric_limits<double>::quiet_NaN();
    debug_.trans_residual = debug_.mahalanobis;
    debug_.rot_residual = debug_.mahalanobis;
    debug_.accepted = false;
    return pose();
  }

private:
  Params params_;
  bool initialized_{false};
  double dt_{1.0 / 20.0};
  Eigen::Matrix4d pose_{Eigen::Matrix4d::Identity()}, predicted_pose_{Eigen::Matrix4d::Identity()};
  Vector6 velocity_{Vector6::Zero()}, predicted_velocity_{Vector6::Zero()};
  Matrix12 covariance_{Matrix12::Identity() * 1e-3}, predicted_covariance_{Matrix12::Identity() * 1e-3};
  Debug debug_;
};
}  // namespace adaptive_pose
