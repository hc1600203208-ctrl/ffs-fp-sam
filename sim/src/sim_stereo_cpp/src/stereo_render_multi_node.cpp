#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include <builtin_interfaces/msg/time.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/imgproc.hpp>
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
Eigen::Matrix3f ToEigen(const cv::Matx33d &input)
{
  Eigen::Matrix3f result;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      result(row, column) = static_cast<float>(input(row, column));
  return result;
}

sensor_msgs::msg::CameraInfo MakeCameraInfo(const Eigen::Matrix3f &k,
                                             const cv::Vec<double, 5> &distortion,
                                             const std::string &frame, int width, int height)
{
  sensor_msgs::msg::CameraInfo info;
  info.width = width;
  info.height = height;
  info.header.frame_id = frame;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
    {
      info.k[row * 3 + column] = k(row, column);
      info.r[row * 3 + column] = row == column ? 1.0 : 0.0;
      info.p[row * 4 + column] = k(row, column);
    }
  info.d.assign(distortion.val, distortion.val + 5);
  info.distortion_model = "plumb_bob";
  return info;
}

sensor_msgs::msg::Image MakeImage(const cv::Mat &image, const std::string &frame,
                                  const builtin_interfaces::msg::Time &stamp)
{
  sensor_msgs::msg::Image message;
  message.header.frame_id = frame;
  message.header.stamp = stamp;
  message.height = image.rows;
  message.width = image.cols;
  message.encoding = sensor_msgs::image_encodings::BGR8;
  message.step = image.cols * image.elemSize();
  message.data.assign(image.datastart, image.dataend);
  return message;
}

// Match the compositing rule in the existing App so dark texture pixels stay inside the silhouette.
cv::Mat ForegroundMask(const cv::Mat &image)
{
  cv::Mat gray, binary, mask = cv::Mat::zeros(image.size(), CV_8UC1);
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, binary, 2, 255, cv::THRESH_BINARY);
  cv::morphologyEx(binary, binary, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  for (const auto &contour : contours)
    if (cv::contourArea(contour) >= 4.0)
    {
      const std::vector<std::vector<cv::Point>> one{contour};
      cv::drawContours(mask, one, 0, cv::Scalar(255), cv::FILLED);
    }
  return mask;
}

void Overlay(cv::Mat &destination, const cv::Mat &rendered)
{
  rendered.copyTo(destination, ForegroundMask(rendered));
}

std::vector<double> SixValues(const std::vector<double> &values, size_t count,
                              const std::string &name)
{
  if (values.empty())
    return std::vector<double>(count, 0.0);
  if (values.size() != count ||
      !std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument(name + " must contain " + std::to_string(count) + " finite values");
  return values;
}
} // namespace

class StereoRenderMultiNode : public rclcpp::Node
{
public:
  StereoRenderMultiNode() : Node("stereo_render_multi_cpp_node")
  {
    const auto paths = declare_parameter<std::vector<std::string>>("mesh_paths", std::vector<std::string>{});
    const auto names = declare_parameter<std::vector<std::string>>("object_names", std::vector<std::string>{});
    const auto initial = declare_parameter<std::vector<double>>("initial_poses", std::vector<double>{});
    const auto amplitudes = declare_parameter<std::vector<double>>("amplitudes", std::vector<double>{});
    const auto frequencies = declare_parameter<std::vector<double>>("frequencies", std::vector<double>{});
    const auto calibration_path = declare_parameter<std::string>("calibration_path", "");
    const int width = declare_parameter<int>("width", 640);
    const int height = declare_parameter<int>("height", 480);
    const double fps = declare_parameter<double>("fps", 30.0);
    const bool interactive = declare_parameter<bool>("interactive", true);
    const auto left_topic = declare_parameter<std::string>("left_image_topic", "/left/image_raw");
    const auto right_topic = declare_parameter<std::string>("right_image_topic", "/right/image_raw");
    const auto left_info_topic = declare_parameter<std::string>("left_camera_info_topic", "/left/camera_info");
    const auto right_info_topic = declare_parameter<std::string>("right_camera_info_topic", "/right/camera_info");
    const auto left_frame = declare_parameter<std::string>("frame_id_left", "stereo_left_optical_frame");
    const auto right_frame = declare_parameter<std::string>("frame_id_right", "stereo_right_optical_frame");

    if (paths.size() < 2 || (!names.empty() && names.size() != paths.size()))
      throw std::invalid_argument("mesh_paths requires at least two meshes and matching object_names");
    if (width <= 0 || height <= 0 || !std::isfinite(fps) || fps <= 0.0)
      throw std::invalid_argument("width, height, and fps must be positive");
    if (initial.size() != 6 * paths.size() ||
        !std::all_of(initial.begin(), initial.end(), [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("initial_poses must contain six finite values per mesh");
    const auto amp = SixValues(amplitudes, 6, "amplitudes");
    const auto freq = SixValues(frequencies, 6, "frequencies");

    StereoCalibration calibration;
    std::string error;
    if (!LoadStereoCalibrationFromTxt(calibration_path, &calibration, &error))
      throw std::runtime_error("Failed to load stereo calibration: " + error);
    const Eigen::Matrix3f left_k = ToEigen(calibration.k_left);
    const Eigen::Matrix3f right_k = ToEigen(calibration.k_right);
    left_k_ = left_k;
    right_k_ = right_k;
    right_from_left_ = Eigen::Matrix4f::Identity();
    right_from_left_.block<3, 3>(0, 0) = ToEigen(calibration.rotation);
    for (int axis = 0; axis < 3; ++axis)
      right_from_left_(axis, 3) = calibration.translation[axis];
    left_info_ = MakeCameraInfo(left_k, calibration.dist_left, left_frame, width, height);
    right_info_ = MakeCameraInfo(right_k, calibration.dist_right, right_frame, width, height);

    for (size_t index = 0; index < paths.size(); ++index)
    {
      if (paths[index].empty())
        throw std::invalid_argument("mesh_paths cannot contain an empty path");
      const std::string name = names.empty() ? "object_" + std::to_string(index) : names[index];
      auto loader = detection_6d::CreateAssimpMeshLoader(name, paths[index]);
      renderers_.push_back(std::make_unique<StereoMeshRenderer>(loader, width, height));
      MotionProfile motion;
      const size_t offset = index * 6;
      motion.base_translation = Eigen::Vector3f(initial[offset], initial[offset + 1], initial[offset + 2]);
      motion.base_rpy_rad = Eigen::Vector3f(initial[offset + 3], initial[offset + 4], initial[offset + 5]) * (kPi / 180.0F);
      motion.trans_amp = Eigen::Vector3f(amp[0], amp[1], amp[2]);
      motion.rot_amp_rad = Eigen::Vector3f(amp[3], amp[4], amp[5]) * (kPi / 180.0F);
      motion.trans_freq = Eigen::Vector3f(freq[0], freq[1], freq[2]);
      motion.rot_freq = Eigen::Vector3f(freq[3], freq[4], freq[5]);
      motions_.push_back(motion);
    }

    left_pub_ = create_publisher<sensor_msgs::msg::Image>(left_topic, 10);
    right_pub_ = create_publisher<sensor_msgs::msg::Image>(right_topic, 10);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(left_info_topic, 10);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(right_info_topic, 10);
    left_frame_ = left_frame;
    right_frame_ = right_frame;
    width_ = width;
    height_ = height;
    running_ = !interactive;
    started_at_ = std::chrono::steady_clock::now();
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::duration<double>(1.0 / fps)),
                               [this] { PublishFrame(); });
    if (interactive)
      command_thread_ = std::thread([this] { CommandLoop(); });
    RCLCPP_INFO(get_logger(), "Multi-mesh renderer running with %zu objects", paths.size());
  }

  ~StereoRenderMultiNode() override
  {
    command_loop_running_ = false;
    if (command_thread_.joinable())
      command_thread_.join();
  }

private:
  static constexpr float kPi = 3.14159265358979323846F;

  void CommandLoop()
  {
    const int fd = ::open("/dev/tty", O_RDONLY | O_NONBLOCK);
    if (fd < 0)
    {
      RCLCPP_WARN(get_logger(), "Cannot open /dev/tty: %s", std::strerror(errno));
      return;
    }
    std::string line;
    while (rclcpp::ok() && command_loop_running_)
    {
      pollfd event{fd, POLLIN, 0};
      if (::poll(&event, 1, 200) <= 0)
        continue;
      char buffer[128];
      const ssize_t count = ::read(fd, buffer, sizeof(buffer));
      if (count <= 0)
        continue;
      for (ssize_t index = 0; index < count; ++index)
      {
        if (buffer[index] != '\n' && buffer[index] != '\r')
        {
          line += buffer[index];
          continue;
        }
        if (line == "start")
        {
          std::lock_guard<std::mutex> lock(motion_mutex_);
          if (!running_)
          {
            started_at_ = std::chrono::steady_clock::now();
            running_ = true;
          }
        }
        else if (line == "pause")
        {
          std::lock_guard<std::mutex> lock(motion_mutex_);
          if (running_)
          {
            elapsed_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at_).count();
            running_ = false;
          }
        }
        else if (line == "quit")
        {
          command_loop_running_ = false;
          rclcpp::shutdown();
        }
        line.clear();
      }
    }
    ::close(fd);
  }

  void PublishFrame()
  {
    double elapsed;
    {
      std::lock_guard<std::mutex> lock(motion_mutex_);
      elapsed = elapsed_;
      if (running_)
        elapsed += std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at_).count();
    }
    cv::Mat left(height_, width_, CV_8UC3, cv::Scalar::all(0));
    cv::Mat right(height_, width_, CV_8UC3, cv::Scalar::all(0));
    for (size_t index = 0; index < renderers_.size(); ++index)
    {
      const Eigen::Matrix4f pose = motions_[index].PoseAt(elapsed);
      Overlay(left, renderers_[index]->RenderBgr(pose, left_k_));
      Overlay(right, renderers_[index]->RenderBgr(right_from_left_ * pose, right_k_));
    }
    const builtin_interfaces::msg::Time stamp = now();
    left_info_.header.stamp = stamp;
    right_info_.header.stamp = stamp;
    left_pub_->publish(MakeImage(left, left_frame_, stamp));
    right_pub_->publish(MakeImage(right, right_frame_, stamp));
    left_info_pub_->publish(left_info_);
    right_info_pub_->publish(right_info_);
  }

  int width_ = 0;
  int height_ = 0;
  std::string left_frame_, right_frame_;
  Eigen::Matrix3f left_k_, right_k_;
  Eigen::Matrix4f right_from_left_;
  sensor_msgs::msg::CameraInfo left_info_, right_info_;
  std::vector<std::unique_ptr<StereoMeshRenderer>> renderers_;
  std::vector<MotionProfile> motions_;
  std::mutex motion_mutex_;
  std::atomic<bool> command_loop_running_{true};
  bool running_ = false;
  double elapsed_ = 0.0;
  std::chrono::steady_clock::time_point started_at_;
  std::thread command_thread_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_, right_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_, right_info_pub_;
};
} // namespace sim_stereo_cpp

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    rclcpp::spin(std::make_shared<sim_stereo_cpp::StereoRenderMultiNode>());
  }
  catch (const std::exception &error)
  {
    RCLCPP_ERROR(rclcpp::get_logger("stereo_render_multi_cpp_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
