#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <Eigen/Dense>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "sim_stereo_cpp/motion_profile.hpp"
#include "sim_stereo_cpp/stereo_mesh_renderer.hpp"

#include "stereo_calibration_utils.hpp"

namespace sim_stereo_cpp
{

namespace
{

Eigen::Matrix3f ToEigen(const cv::Matx33d &m)
{
  Eigen::Matrix3f out = Eigen::Matrix3f::Identity();
  out(0, 0) = static_cast<float>(m(0, 0));
  out(0, 1) = static_cast<float>(m(0, 1));
  out(0, 2) = static_cast<float>(m(0, 2));
  out(1, 0) = static_cast<float>(m(1, 0));
  out(1, 1) = static_cast<float>(m(1, 1));
  out(1, 2) = static_cast<float>(m(1, 2));
  out(2, 0) = static_cast<float>(m(2, 0));
  out(2, 1) = static_cast<float>(m(2, 1));
  out(2, 2) = static_cast<float>(m(2, 2));
  return out;
}

Eigen::Matrix4f StereoToLeftTransform(const StereoCalibration &calibration)
{
  Eigen::Matrix4f right_from_left = Eigen::Matrix4f::Identity();
  right_from_left.block<3, 3>(0, 0) = ToEigen(calibration.rotation);
  right_from_left(0, 3) = static_cast<float>(calibration.translation[0]);
  right_from_left(1, 3) = static_cast<float>(calibration.translation[1]);
  right_from_left(2, 3) = static_cast<float>(calibration.translation[2]);
  return right_from_left;
}

sensor_msgs::msg::CameraInfo MakeCameraInfo(const Eigen::Matrix3f &K,
                                            const cv::Vec<double, 5> &D,
                                            const std::string &frame_id,
                                            int width,
                                            int height)
{
  sensor_msgs::msg::CameraInfo info;
  info.width = static_cast<uint32_t>(width);
  info.height = static_cast<uint32_t>(height);
  info.k = std::array<double, 9>{static_cast<double>(K(0, 0)), static_cast<double>(K(0, 1)),
                                 static_cast<double>(K(0, 2)), static_cast<double>(K(1, 0)),
                                 static_cast<double>(K(1, 1)), static_cast<double>(K(1, 2)),
                                 static_cast<double>(K(2, 0)), static_cast<double>(K(2, 1)),
                                 static_cast<double>(K(2, 2))};
  info.d = {D[0], D[1], D[2], D[3], D[4]};
  info.r = std::array<double, 9>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  info.p = std::array<double, 12>{static_cast<double>(K(0, 0)), static_cast<double>(K(0, 1)),
                                  static_cast<double>(K(0, 2)), 0.0,
                                  static_cast<double>(K(1, 0)), static_cast<double>(K(1, 1)),
                                  static_cast<double>(K(1, 2)), 0.0,
                                  static_cast<double>(K(2, 0)), static_cast<double>(K(2, 1)),
                                  static_cast<double>(K(2, 2)), 0.0};
  info.distortion_model = "plumb_bob";
  info.header.frame_id = frame_id;
  return info;
}

sensor_msgs::msg::Image MakeImageMsg(const cv::Mat &bgr,
                                     const std::string &frame_id,
                                     const builtin_interfaces::msg::Time &stamp)
{
  cv::Mat contiguous_bgr = bgr.isContinuous() ? bgr : bgr.clone();
  sensor_msgs::msg::Image msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = static_cast<uint32_t>(contiguous_bgr.rows);
  msg.width = static_cast<uint32_t>(contiguous_bgr.cols);
  msg.encoding = sensor_msgs::image_encodings::BGR8;
  msg.is_bigendian = 0;
  msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(contiguous_bgr.cols *
                                                             contiguous_bgr.elemSize());
  msg.data.resize(static_cast<size_t>(msg.step) * contiguous_bgr.rows);
  std::memcpy(msg.data.data(), contiguous_bgr.data, msg.data.size());
  return msg;
}

Eigen::Quaternionf ToQuaternion(const Eigen::Matrix4f &pose)
{
  Eigen::Quaternionf q(pose.block<3, 3>(0, 0));
  q.normalize();
  return q;
}

std::vector<std::string> SplitWords(const std::string &line)
{
  std::istringstream ss(line);
  std::vector<std::string> words;
  std::string word;
  while (ss >> word)
  {
    words.push_back(word);
  }
  return words;
}

bool ParseSixFloats(const std::vector<std::string> &tokens,
                    size_t start_index,
                    std::array<float, 6> *values)
{
  if (values == nullptr || tokens.size() < start_index + values->size())
  {
    return false;
  }
  for (size_t i = 0; i < values->size(); ++i)
  {
    try
    {
      (*values)[i] = std::stof(tokens[start_index + i]);
    }
    catch (const std::exception &)
    {
      return false;
    }
  }
  return true;
}

Eigen::Vector3f DegreesToRadians(const Eigen::Vector3f &deg)
{
  constexpr float kPi = 3.14159265358979323846F;
  return deg * (kPi / 180.0F);
}

void PrintMotionProfile(const MotionProfile &motion, bool running, double motion_time_sec)
{
  std::cout << (running ? "running" : "paused")
            << ", motion_time=" << motion_time_sec << "s\n";
  std::cout << "base_translation: " << motion.base_translation.transpose() << "\n";
  std::cout << "base_rpy_rad: " << motion.base_rpy_rad.transpose() << "\n";
  std::cout << "trans_amp: " << motion.trans_amp.transpose() << "\n";
  std::cout << "trans_freq: " << motion.trans_freq.transpose() << "\n";
  std::cout << "rot_amp_rad: " << motion.rot_amp_rad.transpose() << "\n";
  std::cout << "rot_freq: " << motion.rot_freq.transpose() << std::endl;
}

} // namespace

class StereoRenderNode : public rclcpp::Node
{
public:
  StereoRenderNode()
      : Node("stereo_render_cpp_node")
  {
    mesh_path_ = declare_parameter<std::string>(
        "mesh_path", "/home/hc/weizi/dataset/jrnew-blue/mesh1/textured_mesh.obj");
    calibration_path_ = declare_parameter<std::string>(
        "calibration_path", "/home/hc/weizi/ffs+fp+sam/ffs/640.txt");
    width_ = declare_parameter<int>("width", 640);
    height_ = declare_parameter<int>("height", 480);
    fps_ = declare_parameter<double>("fps", 30.0);
    left_image_topic_ = declare_parameter<std::string>("left_image_topic", "/sim/left/image_raw");
    right_image_topic_ = declare_parameter<std::string>("right_image_topic", "/sim/right/image_raw");
    left_info_topic_ = declare_parameter<std::string>("left_camera_info_topic", "/sim/left/camera_info");
    right_info_topic_ = declare_parameter<std::string>("right_camera_info_topic", "/sim/right/camera_info");
    frame_left_ = declare_parameter<std::string>("frame_id_left", "stereo_left_optical_frame");
    frame_right_ = declare_parameter<std::string>("frame_id_right", "stereo_right_optical_frame");
    interactive_ = declare_parameter<bool>("interactive", true);

    if (!LoadStereoCalibrationFromTxt(calibration_path_, &calibration_, &calibration_error_))
    {
      throw std::runtime_error("Failed to load stereo calibration: " + calibration_error_);
    }

    left_k_ = ToEigen(calibration_.k_left);
    right_k_ = ToEigen(calibration_.k_right);
    left_info_ = MakeCameraInfo(left_k_, calibration_.dist_left, frame_left_, width_, height_);
    right_info_ = MakeCameraInfo(right_k_, calibration_.dist_right, frame_right_, width_, height_);
    right_from_left_ = StereoToLeftTransform(calibration_);

    auto mesh_loader = detection_6d::CreateAssimpMeshLoader("sim_mesh", mesh_path_);
    renderer_ = std::make_shared<StereoMeshRenderer>(mesh_loader, width_, height_);

    left_pub_ = create_publisher<sensor_msgs::msg::Image>(left_image_topic_, 10);
    right_pub_ = create_publisher<sensor_msgs::msg::Image>(right_image_topic_, 10);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(left_info_topic_, 10);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(right_info_topic_, 10);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/sim/object_pose", 10);

    if (interactive_)
    {
      motion_ = MotionProfile::PromptFromStdin();
      motion_running_ = false;
      motion_elapsed_sec_ = 0.0;
      motion_run_started_at_ = std::chrono::steady_clock::now();
    }
    else
    {
      motion_running_ = true;
      motion_elapsed_sec_ = 0.0;
      motion_run_started_at_ = std::chrono::steady_clock::now();
    }

    const auto period = std::chrono::duration<double>(1.0 / std::max(1e-3, fps_));
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                               std::bind(&StereoRenderNode::OnTimer, this));

    if (interactive_)
    {
      command_loop_running_.store(true);
      command_thread_ = std::thread(&StereoRenderNode::CommandLoop, this);
      RCLCPP_INFO(get_logger(),
                  "Interactive mode enabled. Commands: start | pause | toggle | show | reset | quit");
      RCLCPP_INFO(get_logger(), "Motion is paused until you type 'start' or 'toggle'.");
    }

    RCLCPP_INFO(get_logger(), "Stereo C++ renderer running at %.1f Hz", fps_);
  }

  ~StereoRenderNode() override
  {
    StopCommandLoop();
  }

private:
  void StopCommandLoop()
  {
    command_loop_running_.store(false);
    if (command_thread_.joinable())
    {
      command_thread_.join();
    }
  }

  double MotionElapsedSec()
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(motion_mutex_);
    double elapsed = motion_elapsed_sec_;
    if (motion_running_)
    {
      elapsed += std::chrono::duration<double>(now - motion_run_started_at_).count();
    }
    return elapsed;
  }

  Eigen::Matrix4f CurrentPose()
  {
    MotionProfile motion;
    {
      std::lock_guard<std::mutex> lock(motion_mutex_);
      motion = motion_;
    }
    return motion.PoseAt(MotionElapsedSec());
  }

  void UpdateMotionProfile(const MotionProfile &motion)
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(motion_mutex_);
    motion_ = motion;
    motion_elapsed_sec_ = 0.0;
    if (motion_running_)
    {
      motion_run_started_at_ = now;
    }
  }

  void StartMotion()
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(motion_mutex_);
    if (motion_running_)
    {
      return;
    }
    motion_running_ = true;
    motion_run_started_at_ = now;
  }

  void PauseMotion()
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(motion_mutex_);
    if (!motion_running_)
    {
      return;
    }
    motion_elapsed_sec_ += std::chrono::duration<double>(now - motion_run_started_at_).count();
    motion_running_ = false;
  }

  void ToggleMotion()
  {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(motion_mutex_);
    if (motion_running_)
    {
      motion_elapsed_sec_ += std::chrono::duration<double>(now - motion_run_started_at_).count();
      motion_running_ = false;
    }
    else
    {
      motion_running_ = true;
      motion_run_started_at_ = now;
    }
  }

  void ResetMotion()
  {
    PauseMotion();
    UpdateMotionProfile(MotionProfile{});
  }

  void ApplyPoseCommand(const std::array<float, 6> &values)
  {
    MotionProfile current;
    {
      std::lock_guard<std::mutex> lock(motion_mutex_);
      current = motion_;
    }
    MotionProfile updated = current;
    updated.base_translation = Eigen::Vector3f(values[0], values[1], values[2]);
    updated.base_rpy_rad = DegreesToRadians(Eigen::Vector3f(values[3], values[4], values[5]));
    UpdateMotionProfile(updated);
  }

  void ApplyAmpCommand(const std::array<float, 6> &values)
  {
    MotionProfile current;
    {
      std::lock_guard<std::mutex> lock(motion_mutex_);
      current = motion_;
    }
    MotionProfile updated = current;
    updated.trans_amp = Eigen::Vector3f(values[0], values[1], values[2]);
    updated.rot_amp_rad = DegreesToRadians(Eigen::Vector3f(values[3], values[4], values[5]));
    UpdateMotionProfile(updated);
  }

  void ApplyFreqCommand(const std::array<float, 6> &values)
  {
    MotionProfile current;
    {
      std::lock_guard<std::mutex> lock(motion_mutex_);
      current = motion_;
    }
    MotionProfile updated = current;
    updated.trans_freq = Eigen::Vector3f(values[0], values[1], values[2]);
    updated.rot_freq = Eigen::Vector3f(values[3], values[4], values[5]);
    UpdateMotionProfile(updated);
  }

  void ProcessCommand(const std::string &line)
  {
    const auto tokens = SplitWords(line);
    if (tokens.empty())
    {
      return;
    }

    const std::string &cmd = tokens[0];
    if (cmd == "help")
    {
      std::cout << "Commands: help | show | start | pause | toggle | "
                   "pose x y z roll pitch yaw | amp tx ty tz roll pitch yaw | "
                   "freq fx fy fz fr fp fyaw | reset | quit"
                << std::endl;
      return;
    }

    if (cmd == "show")
    {
      MotionProfile current;
      bool running = false;
      double motion_time = 0.0;
      {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        current = motion_;
        running = motion_running_;
        motion_time = motion_elapsed_sec_;
        if (motion_running_)
        {
          motion_time += std::chrono::duration<double>(
              std::chrono::steady_clock::now() - motion_run_started_at_).count();
        }
      }
      PrintMotionProfile(current, running, motion_time);
      return;
    }

    if (cmd == "start")
    {
      StartMotion();
      std::cout << "Motion started." << std::endl;
      return;
    }

    if (cmd == "pause")
    {
      PauseMotion();
      std::cout << "Motion paused." << std::endl;
      return;
    }

    if (cmd == "toggle")
    {
      ToggleMotion();
      std::cout << "Motion toggled." << std::endl;
      return;
    }

    if (cmd == "reset")
    {
      ResetMotion();
      std::cout << "Motion reset to defaults and paused." << std::endl;
      return;
    }

    if (cmd == "quit")
    {
      command_loop_running_.store(false);
      rclcpp::shutdown();
      return;
    }

    std::array<float, 6> values{};
    if (cmd == "pose")
    {
      if (ParseSixFloats(tokens, 1, &values))
      {
        ApplyPoseCommand(values);
        std::cout << "Updated motion profile." << std::endl;
      }
      else
      {
        std::cout << "Usage: pose x y z roll pitch yaw" << std::endl;
      }
      return;
    }

    if (cmd == "amp")
    {
      if (ParseSixFloats(tokens, 1, &values))
      {
        ApplyAmpCommand(values);
        std::cout << "Updated motion profile." << std::endl;
      }
      else
      {
        std::cout << "Usage: amp tx ty tz roll pitch yaw" << std::endl;
      }
      return;
    }

    if (cmd == "freq")
    {
      if (ParseSixFloats(tokens, 1, &values))
      {
        ApplyFreqCommand(values);
        std::cout << "Updated motion profile." << std::endl;
      }
      else
      {
        std::cout << "Usage: freq fx fy fz fr fp fyaw" << std::endl;
      }
      return;
    }

    std::cout << "Unknown command: " << cmd << std::endl;
  }

  void CommandLoop()
  {
    int tty_fd = ::open("/dev/tty", O_RDONLY | O_NONBLOCK);
    if (tty_fd < 0)
    {
      RCLCPP_WARN(get_logger(), "Failed to open /dev/tty for interactive commands: %s",
                  std::strerror(errno));
      return;
    }

    std::string buffer;
    std::cout << "Type 'help' for commands." << std::endl;

    while (rclcpp::ok() && command_loop_running_.load())
    {
      pollfd pfd;
      pfd.fd = tty_fd;
      pfd.events = POLLIN;
      pfd.revents = 0;

      const int poll_ret = ::poll(&pfd, 1, 200);
      if (poll_ret < 0)
      {
        if (errno == EINTR)
        {
          continue;
        }
        RCLCPP_WARN(get_logger(), "poll(/dev/tty) failed: %s", std::strerror(errno));
        break;
      }
      if (poll_ret == 0)
      {
        continue;
      }

      char read_buf[256];
      const ssize_t n_read = ::read(tty_fd, read_buf, sizeof(read_buf));
      if (n_read < 0)
      {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        {
          continue;
        }
        RCLCPP_WARN(get_logger(), "read(/dev/tty) failed: %s", std::strerror(errno));
        break;
      }
      if (n_read == 0)
      {
        continue;
      }

      for (ssize_t i = 0; i < n_read; ++i)
      {
        const char ch = read_buf[i];
        if (ch == '\r' || ch == '\n')
        {
          if (!buffer.empty())
          {
            ProcessCommand(buffer);
            buffer.clear();
          }
        }
        else
        {
          buffer.push_back(ch);
        }
      }
    }

    ::close(tty_fd);
  }

  void OnTimer()
  {
    const Eigen::Matrix4f left_pose = CurrentPose();
    const Eigen::Matrix4f right_pose = right_from_left_ * left_pose;

    cv::Mat left_bgr = renderer_->RenderBgr(left_pose, left_k_);
    cv::Mat right_bgr = renderer_->RenderBgr(right_pose, right_k_);

    const rclcpp::Time now = this->now();
    builtin_interfaces::msg::Time stamp;
    const int64_t stamp_ns = now.nanoseconds();
    stamp.sec = static_cast<int32_t>(stamp_ns / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
    auto left_msg = MakeImageMsg(left_bgr, frame_left_, stamp);
    auto right_msg = MakeImageMsg(right_bgr, frame_right_, stamp);
    left_info_.header.stamp = stamp;
    right_info_.header.stamp = stamp;

    left_pub_->publish(left_msg);
    right_pub_->publish(right_msg);
    left_info_pub_->publish(left_info_);
    right_info_pub_->publish(right_info_);

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = frame_left_;
    pose_msg.pose.position.x = left_pose(0, 3);
    pose_msg.pose.position.y = left_pose(1, 3);
    pose_msg.pose.position.z = left_pose(2, 3);
    const Eigen::Quaternionf q = ToQuaternion(left_pose);
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    pose_pub_->publish(pose_msg);
  }

  std::string mesh_path_;
  std::string calibration_path_;
  std::string calibration_error_;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 30.0;
  bool interactive_ = true;
  std::atomic<bool> command_loop_running_{false};

  std::string left_image_topic_;
  std::string right_image_topic_;
  std::string left_info_topic_;
  std::string right_info_topic_;
  std::string frame_left_;
  std::string frame_right_;

  StereoCalibration calibration_;
  Eigen::Matrix3f left_k_ = Eigen::Matrix3f::Identity();
  Eigen::Matrix3f right_k_ = Eigen::Matrix3f::Identity();
  Eigen::Matrix4f right_from_left_ = Eigen::Matrix4f::Identity();

  sensor_msgs::msg::CameraInfo left_info_;
  sensor_msgs::msg::CameraInfo right_info_;

  std::mutex motion_mutex_;
  MotionProfile motion_;
  bool motion_running_ = false;
  double motion_elapsed_sec_ = 0.0;
  std::chrono::steady_clock::time_point motion_run_started_at_;
  std::thread command_thread_;
  std::shared_ptr<StereoMeshRenderer> renderer_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace sim_stereo_cpp

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<sim_stereo_cpp::StereoRenderNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception &e)
  {
    std::cerr << "stereo_render_cpp_node failed: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
