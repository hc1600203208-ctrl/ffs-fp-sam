#pragma once

#include <Eigen/Dense>

#include "se3_utils.h"

namespace foundationpose_filter
{

using Matrix6f = Eigen::Matrix<float, 6, 6>;
using Vector12f = Eigen::Matrix<float, 12, 1>;
using Matrix12f = Eigen::Matrix<float, 12, 12>;

struct AdaptiveSE3PoseFilterParams
{
  float q_pos = 1e-4F;
  float q_rot = 1e-4F;
  float q_vel = 1e-3F;
  float q_omega = 1e-3F;

  float sigma_pos_obs = 0.01F;
  float sigma_rot_obs = 0.05F;

  float sigma_pos_motion = 0.05F;
  float sigma_rot_motion = 0.35F;

  float min_quality = 0.01F;
  float gate_threshold = 16.81F;
  float reject_cov_increase = 1e-4F;

  bool enable_gating = true;
  bool enable_adaptive_R = true;
};

class AdaptiveSE3PoseFilter
{
public:
  void initialize(const Eigen::Matrix4f &init_pose,
                  float                  dt,
                  const AdaptiveSE3PoseFilterParams &params);

  Eigen::Matrix4f predict(float dt);
  Eigen::Matrix4f update(const Eigen::Matrix4f &T_obs);
  Eigen::Matrix4f rejectObservation();

  const Eigen::Matrix4f &getPose() const;
  float getLastQuality() const;
  float getLastMahalanobisDistance() const;
  bool wasLastObservationAccepted() const;
  float getLastTranslationResidualNorm() const;
  float getLastRotationResidualNorm() const;
  bool isInitialized() const;

private:
  Matrix12f buildProcessNoise(float dt) const;
  Matrix6f buildObservationNoise(float quality) const;
  static float sanitizeDt(float dt);
  static float positiveOrDefault(float value, float fallback);
  static void symmetrize(Matrix12f *matrix);

  AdaptiveSE3PoseFilterParams params_;

  Eigen::Matrix4f T_{Eigen::Matrix4f::Identity()};
  Vector6f xi_{Vector6f::Zero()};
  Matrix12f P_{Matrix12f::Identity()};

  Eigen::Matrix4f T_pred_{Eigen::Matrix4f::Identity()};
  Vector6f xi_pred_{Vector6f::Zero()};
  Matrix12f P_pred_{Matrix12f::Identity()};

  bool initialized_{false};
  bool has_prediction_{false};
  float last_dt_{1.0F / 30.0F};
  float last_quality_{1.0F};
  float last_mahalanobis_distance_{0.0F};
  bool last_observation_accepted_{true};
  float last_translation_residual_norm_{0.0F};
  float last_rotation_residual_norm_{0.0F};
};

} // namespace foundationpose_filter
