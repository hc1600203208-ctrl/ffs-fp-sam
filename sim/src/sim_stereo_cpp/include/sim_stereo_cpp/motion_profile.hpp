#pragma once

#include <chrono>
#include <string>

#include <Eigen/Dense>

namespace sim_stereo_cpp
{

struct MotionProfile
{
  Eigen::Vector3f base_translation = Eigen::Vector3f(0.0F, 0.0F, 0.8F);
  Eigen::Vector3f base_rpy_rad     = Eigen::Vector3f::Zero();
  Eigen::Vector3f trans_amp        = Eigen::Vector3f::Zero();
  Eigen::Vector3f trans_freq       = Eigen::Vector3f::Zero();
  Eigen::Vector3f rot_amp_rad      = Eigen::Vector3f::Zero();
  Eigen::Vector3f rot_freq         = Eigen::Vector3f::Zero();

  static MotionProfile PromptFromStdin();

  Eigen::Matrix4f PoseAt(double t_sec) const;
};

} // namespace sim_stereo_cpp

