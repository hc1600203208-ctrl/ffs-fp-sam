#include "sim_stereo_cpp/motion_profile.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace sim_stereo_cpp
{

namespace
{

constexpr float kPi = 3.14159265358979323846F;

bool ParseVector3(const std::string &line, Eigen::Vector3f *out)
{
  if (out == nullptr)
  {
    return false;
  }

  std::stringstream ss(line);
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  if (!(ss >> x >> y >> z))
  {
    return false;
  }
  *out = Eigen::Vector3f(x, y, z);
  return true;
}

Eigen::Vector3f PromptVector3(const std::string &prompt,
                              const Eigen::Vector3f &default_value,
                              bool degrees = false)
{
  const Eigen::Vector3f display_default =
      degrees ? default_value * (180.0F / kPi) : default_value;
  std::cout << prompt;
  if (degrees)
  {
    std::cout << " (deg)";
  }
  std::cout << " ["
            << display_default.x() << ", " << display_default.y() << ", " << display_default.z()
            << "]" << std::endl;

  std::string line;
  std::istream *input_stream = &std::cin;
  std::ifstream tty_stream("/dev/tty");
  if (tty_stream.is_open())
  {
    input_stream = &tty_stream;
  }

  if (!std::getline(*input_stream, line))
  {
    std::cout << "\nNo interactive input available, using defaults.\n";
    return default_value;
  }

  if (line.empty())
  {
    return default_value;
  }

  Eigen::Vector3f parsed = default_value;
  if (!ParseVector3(line, &parsed))
  {
    std::cout << "Invalid input, using defaults.\n";
    return default_value;
  }
  if (degrees)
  {
    parsed *= (kPi / 180.0F);
  }
  return parsed;
}

Eigen::Matrix3f RotationFromRpy(const Eigen::Vector3f &rpy_rad)
{
  const float roll = rpy_rad.x();
  const float pitch = rpy_rad.y();
  const float yaw = rpy_rad.z();

  const float cr = std::cos(roll);
  const float sr = std::sin(roll);
  const float cp = std::cos(pitch);
  const float sp = std::sin(pitch);
  const float cy = std::cos(yaw);
  const float sy = std::sin(yaw);

  Eigen::Matrix3f rx;
  rx << 1.0F, 0.0F, 0.0F,
        0.0F, cr, -sr,
        0.0F, sr, cr;

  Eigen::Matrix3f ry;
  ry << cp, 0.0F, sp,
        0.0F, 1.0F, 0.0F,
        -sp, 0.0F, cp;

  Eigen::Matrix3f rz;
  rz << cy, -sy, 0.0F,
        sy, cy, 0.0F,
        0.0F, 0.0F, 1.0F;

  return rz * ry * rx;
}

} // namespace

MotionProfile MotionProfile::PromptFromStdin()
{
  MotionProfile profile;
  profile.base_translation =
      PromptVector3("Initial translation x y z", profile.base_translation, false);
  profile.base_rpy_rad = PromptVector3("Initial rotation roll pitch yaw",
                                       profile.base_rpy_rad,
                                       true);
  profile.trans_amp = PromptVector3("Translation amplitude x y z", profile.trans_amp, false);
  profile.trans_freq = PromptVector3("Translation frequency x y z", profile.trans_freq, false);
  profile.rot_amp_rad =
      PromptVector3("Rotation amplitude roll pitch yaw", profile.rot_amp_rad, true);
  profile.rot_freq = PromptVector3("Rotation frequency roll pitch yaw", profile.rot_freq, false);
  return profile;
}

Eigen::Matrix4f MotionProfile::PoseAt(double t_sec) const
{
  const float t = static_cast<float>(t_sec);
  const Eigen::Vector3f trans_offset =
      trans_amp.array() * (2.0F * kPi * trans_freq.array() * t).sin();
  const Eigen::Vector3f rot_offset =
      rot_amp_rad.array() * (2.0F * kPi * rot_freq.array() * t).sin();

  const Eigen::Vector3f translation = base_translation + trans_offset;
  const Eigen::Matrix3f rotation = RotationFromRpy(base_rpy_rad + rot_offset);

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3, 3>(0, 0) = rotation;
  pose.block<3, 1>(0, 3) = translation;
  return pose;
}

} // namespace sim_stereo_cpp
