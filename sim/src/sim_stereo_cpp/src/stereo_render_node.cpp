#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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
    }

    start_time_ = std::chrono::steady_clock::now();
    const auto period = std::chrono::duration<double>(1.0 / std::max(1e-3, fps_));
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                               std::bind(&StereoRenderNode::OnTimer, this));

    RCLCPP_INFO(get_logger(), "Stereo C++ renderer running at %.1f Hz", fps_);
  }

private:
  void OnTimer()
  {
    const auto wall_now = std::chrono::steady_clock::now();
    const double t_sec = std::chrono::duration<double>(wall_now - start_time_).count();
    const Eigen::Matrix4f left_pose = motion_.PoseAt(t_sec);
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

  MotionProfile motion_;
  std::chrono::steady_clock::time_point start_time_;
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
