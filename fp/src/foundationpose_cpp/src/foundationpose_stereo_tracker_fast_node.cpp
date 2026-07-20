// Minimal high-speed stereo FoundationPose tracker.
// Keeps only: stereo image sync -> stereo depth -> first mask registration -> direct tracking -> pose publish.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "estimator/fast_foundation_stereo_estimator.h"
#include "stereo_calibration_utils.hpp"
#include "trt_core/trt_core.h"

namespace
{

constexpr int kPoseBatchSize = 252;
constexpr int kCropHeight    = 160;
constexpr int kCropWidth     = 160;
constexpr float kRadiansToDegrees = 57.29577951308232F;

struct PreparedImage
{
  cv::Mat image;
  double  scale          = 1.0;
  int     pad_left       = 0;
  int     pad_top        = 0;
  int     resized_width  = 0;
  int     resized_height = 0;
};

struct CameraIntrinsics
{
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

bool CheckTimestamp(const builtin_interfaces::msg::Time &time1,
                    const builtin_interfaces::msg::Time &time2,
                    double                              max_delta_sec)
{
  const double timestamp1 =
      static_cast<double>(time1.sec) + static_cast<double>(time1.nanosec) / 1e9;
  const double timestamp2 =
      static_cast<double>(time2.sec) + static_cast<double>(time2.nanosec) / 1e9;
  return std::fabs(timestamp1 - timestamp2) <= max_delta_sec;
}

Eigen::Matrix3f BuildIntrinsicMatrix(double fx, double fy, double cx, double cy)
{
  if (fx <= 0.0 || fy <= 0.0)
  {
    throw std::invalid_argument("Camera intrinsics fx/fy must be positive.");
  }

  Eigen::Matrix3f intrinsic = Eigen::Matrix3f::Identity();
  intrinsic(0, 0)           = static_cast<float>(fx);
  intrinsic(1, 1)           = static_cast<float>(fy);
  intrinsic(0, 2)           = static_cast<float>(cx);
  intrinsic(1, 2)           = static_cast<float>(cy);
  return intrinsic;
}

cv::Mat RosImageToCvMat(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  return cv_bridge::toCvShare(msg, msg->encoding)->image;
}

cv::Mat ConvertPoseRgbImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  const cv::Mat image = RosImageToCvMat(msg);
  if (msg->encoding == sensor_msgs::image_encodings::RGB8)
  {
    return image;
  }

  cv::Mat rgb;
  if (msg->encoding == sensor_msgs::image_encodings::BGR8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    return rgb;
  }
  if (msg->encoding == sensor_msgs::image_encodings::RGBA8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_RGBA2RGB);
    return rgb;
  }
  if (msg->encoding == sensor_msgs::image_encodings::BGRA8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
    return rgb;
  }
  if (msg->encoding == sensor_msgs::image_encodings::MONO8 ||
      msg->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
  {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
  }

  throw std::runtime_error("Unsupported RGB image encoding: " + msg->encoding);
}

cv::Mat ConvertStereoInputImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  const cv::Mat image = RosImageToCvMat(msg);
  if (image.channels() == 1 || image.channels() == 3)
  {
    return image;
  }

  cv::Mat converted;
  if (image.channels() == 4)
  {
    cv::cvtColor(image, converted, cv::COLOR_BGRA2BGR);
    return converted;
  }

  throw std::runtime_error("Unsupported stereo image channel count.");
}

cv::Mat ConvertMaskImage(cv::Mat mask)
{
  if (mask.empty())
  {
    return mask;
  }

  if (mask.channels() == 3)
  {
    cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
  }
  else if (mask.channels() == 4)
  {
    cv::cvtColor(mask, mask, cv::COLOR_BGRA2GRAY);
  }
  else if (mask.channels() != 1)
  {
    throw std::runtime_error("Mask image must have 1, 3, or 4 channels.");
  }

  if (mask.type() == CV_16UC1)
  {
    mask.convertTo(mask, CV_8UC1, 1.0 / 256.0);
  }
  else if (mask.type() == CV_32FC1)
  {
    cv::patchNaNs(mask, 0.0);
    mask.convertTo(mask, CV_8UC1, 255.0);
  }
  else if (mask.type() != CV_8UC1)
  {
    mask.convertTo(mask, CV_8UC1);
  }

  cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
  return mask;
}

PreparedImage PrepareStereoInput(const cv::Mat &input, int target_width, int target_height)
{
  PreparedImage result;
  if (input.empty())
  {
    return result;
  }

  cv::Mat color;
  if (input.channels() == 1)
  {
    cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
  }
  else if (input.channels() == 3)
  {
    color = input;
  }
  else
  {
    throw std::runtime_error("Unsupported image channel count.");
  }

  result.scale = std::min(static_cast<double>(target_width) / static_cast<double>(color.cols),
                          static_cast<double>(target_height) / static_cast<double>(color.rows));
  result.resized_width  = std::max(1, static_cast<int>(std::round(color.cols * result.scale)));
  result.resized_height = std::max(1, static_cast<int>(std::round(color.rows * result.scale)));

  cv::Mat resized;
  cv::resize(color,
             resized,
             cv::Size(result.resized_width, result.resized_height),
             0,
             0,
             cv::INTER_LINEAR);

  const int pad_width  = target_width - result.resized_width;
  const int pad_height = target_height - result.resized_height;
  result.pad_left      = pad_width / 2;
  result.pad_top       = pad_height / 2;

  cv::copyMakeBorder(resized,
                     result.image,
                     result.pad_top,
                     pad_height - result.pad_top,
                     result.pad_left,
                     pad_width - result.pad_left,
                     cv::BORDER_REPLICATE);
  return result;
}

double ComputeRectifiedBaselineMeters(const StereoRectificationMaps &rectification_maps)
{
  if (rectification_maps.p2.empty())
  {
    return 0.0;
  }

  const double fx = rectification_maps.p2.at<double>(0, 0);
  if (std::abs(fx) <= 1e-9)
  {
    return 0.0;
  }

  return std::abs(rectification_maps.p2.at<double>(0, 3) / fx);
}

CameraIntrinsics ComputePreparedIntrinsics(const StereoRectificationMaps &rectification_maps,
                                           const PreparedImage           &prepared)
{
  CameraIntrinsics intrinsics;
  if (rectification_maps.p1.empty())
  {
    return intrinsics;
  }

  intrinsics.fx = rectification_maps.p1.at<double>(0, 0) * prepared.scale;
  intrinsics.fy = rectification_maps.p1.at<double>(1, 1) * prepared.scale;
  intrinsics.cx = rectification_maps.p1.at<double>(0, 2) * prepared.scale +
                  static_cast<double>(prepared.pad_left);
  intrinsics.cy = rectification_maps.p1.at<double>(1, 2) * prepared.scale +
                  static_cast<double>(prepared.pad_top);
  return intrinsics;
}

cv::Mat DisparityToDepthMeters(const cv::Mat &disparity,
                               float          fx,
                               float          baseline,
                               float          min_depth_meters,
                               float          max_depth_meters)
{
  cv::Mat     depth = cv::Mat::zeros(disparity.size(), CV_32FC1);
  const float numerator = fx * baseline;

  for (int y = 0; y < disparity.rows; ++y)
  {
    const float *disparity_row = disparity.ptr<float>(y);
    float       *depth_row     = depth.ptr<float>(y);
    for (int x = 0; x < disparity.cols; ++x)
    {
      const float disparity_value = disparity_row[x];
      if (!std::isfinite(disparity_value) || disparity_value <= 0.0F)
      {
        continue;
      }

      const float depth_value = numerator / disparity_value;
      if (std::isfinite(depth_value) && depth_value >= min_depth_meters &&
          depth_value <= max_depth_meters)
      {
        depth_row[x] = depth_value;
      }
    }
  }

  return depth;
}

cv::Mat RestoreToRectifiedResolution(const cv::Mat       &image,
                                     const PreparedImage &prepared,
                                     const cv::Size      &rectified_size,
                                     int                  interpolation)
{
  if (image.empty() || rectified_size.width <= 0 || rectified_size.height <= 0)
  {
    return {};
  }

  const cv::Rect valid_roi(prepared.pad_left,
                           prepared.pad_top,
                           std::min(prepared.resized_width, image.cols - prepared.pad_left),
                           std::min(prepared.resized_height, image.rows - prepared.pad_top));
  if (valid_roi.width <= 0 || valid_roi.height <= 0)
  {
    return {};
  }

  cv::Mat restored;
  cv::resize(image(valid_roi), restored, rectified_size, 0, 0, interpolation);
  return restored;
}

geometry_msgs::msg::PoseStamped PoseMatrixToPoseStamped(const Eigen::Matrix4f      &pose,
                                                        const std_msgs::msg::Header &header)
{
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = header;
  pose_msg.pose.position.x = pose(0, 3);
  pose_msg.pose.position.y = pose(1, 3);
  pose_msg.pose.position.z = pose(2, 3);

  Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
  quat.normalize();
  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  pose_msg.pose.orientation.w = quat.w();
  return pose_msg;
}

Eigen::Vector3f RotationMatrixToRpyDegrees(const Eigen::Matrix3f &rotation)
{
  const float sy = std::sqrt(rotation(0, 0) * rotation(0, 0) +
                             rotation(1, 0) * rotation(1, 0));
  const bool singular = sy < 1e-6F;

  float roll = 0.0F;
  float pitch = 0.0F;
  float yaw = 0.0F;
  if (!singular)
  {
    roll = std::atan2(rotation(2, 1), rotation(2, 2));
    pitch = std::atan2(-rotation(2, 0), sy);
    yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  }
  else
  {
    roll = std::atan2(-rotation(1, 2), rotation(1, 1));
    pitch = std::atan2(-rotation(2, 0), sy);
    yaw = 0.0F;
  }

  return Eigen::Vector3f(roll, pitch, yaw) * kRadiansToDegrees;
}

std::string PoseToCsvRow(size_t                      frame_index,
                         const std::string          &mode,
                         const std_msgs::msg::Header &header,
                         const std::string          &visualization_path,
                         const Eigen::Matrix4f      &pose,
                         const Eigen::Vector3f      &rotation_rpy_degrees,
                         const Eigen::Vector3f      &delta_rotation_rpy_degrees)
{
  Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
  quat.normalize();

  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(8);
  stream << frame_index << ',' << mode << ',' << header.stamp.sec << ',' << header.stamp.nanosec
         << ',' << header.frame_id << ',' << visualization_path << ',' << pose(0, 3) << ','
         << pose(1, 3) << ',' << pose(2, 3) << ',' << quat.x() << ',' << quat.y() << ','
         << quat.z() << ',' << quat.w() << ',' << rotation_rpy_degrees(0) << ','
         << rotation_rpy_degrees(1) << ',' << rotation_rpy_degrees(2) << ','
         << delta_rotation_rpy_degrees(0) << ',' << delta_rotation_rpy_degrees(1) << ','
         << delta_rotation_rpy_degrees(2);

  for (int row = 0; row < 4; ++row)
  {
    for (int col = 0; col < 4; ++col)
    {
      stream << ',' << pose(row, col);
    }
  }

  return stream.str();
}

void Draw3DBoundingBox(const Eigen::Matrix3f &intrinsic,
                       const Eigen::Matrix4f &pose,
                       const Eigen::Vector3f &dimension,
                       cv::Mat               &image)
{
  const float half_l = dimension(0) / 2.0F;
  const float half_w = dimension(1) / 2.0F;
  const float half_h = dimension(2) / 2.0F;

  const Eigen::Vector3f points[8] = {
      {-half_l, -half_w, half_h},  {half_l, -half_w, half_h},
      {half_l, half_w, half_h},    {-half_l, half_w, half_h},
      {-half_l, -half_w, -half_h}, {half_l, -half_w, -half_h},
      {half_l, half_w, -half_h},   {-half_l, half_w, -half_h}};

  Eigen::Vector4f transformed_points[8];
  for (int i = 0; i < 8; ++i)
  {
    transformed_points[i] = pose * Eigen::Vector4f(points[i](0), points[i](1), points[i](2), 1.0F);
  }

  std::vector<cv::Point2f> image_points;
  image_points.reserve(8);
  for (const auto &point : transformed_points)
  {
    const float z = point(2);
    if (z <= 1e-6F)
    {
      return;
    }

    image_points.emplace_back(intrinsic(0, 0) * (point(0) / z) + intrinsic(0, 2),
                              intrinsic(1, 1) * (point(1) / z) + intrinsic(1, 2));
  }

  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                                  {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (const auto &edge : edges)
  {
    // cv::line(image, image_points[edge.first], image_points[edge.second], cv::Scalar(0, 255, 0), 2);
  }

  const Eigen::Vector4f center = pose * Eigen::Vector4f(0.0F, 0.0F, 0.0F, 1.0F);
  if (center(2) <= 1e-6F)
  {
    return;
  }

  const float axis_length = (dimension(0) + dimension(1) + dimension(2)) / 6.0F;
  const std::vector<Eigen::Vector4f> axis_end_points = {
      pose * Eigen::Vector4f(axis_length, 0.0F, 0.0F, 1.0F),
      pose * Eigen::Vector4f(0.0F, 0.0F, axis_length, 1.0F),
      pose * Eigen::Vector4f(0.0F, axis_length, 0.0F, 1.0F)};
  const std::vector<cv::Scalar> axis_colors = {
      cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};

  const cv::Point center_pt(intrinsic(0, 0) * (center(0) / center(2)) + intrinsic(0, 2),
                            intrinsic(1, 1) * (center(1) / center(2)) + intrinsic(1, 2));

  for (size_t i = 0; i < axis_end_points.size(); ++i)
  {
    if (axis_end_points[i](2) <= 1e-6F)
    {
      continue;
    }

    const cv::Point end_pt(intrinsic(0, 0) * (axis_end_points[i](0) / axis_end_points[i](2)) +
                               intrinsic(0, 2),
                           intrinsic(1, 1) * (axis_end_points[i](1) / axis_end_points[i](2)) +
                               intrinsic(1, 2));
    cv::line(image, center_pt, end_pt, axis_colors[i], 3);
  }
}

} // namespace

class FoundationPoseStereoTrackerFastNode : public rclcpp::Node
{
public:
  FoundationPoseStereoTrackerFastNode()
      : Node("foundationpose_stereo_tracker_fast_node"),
        image_qos_profile_(rclcpp::QoS(10))
  {
    DeclareParameters();
    LoadParameters();
    BuildStereoEstimator();
    BuildPoseModel();
    SetupRosInterfaces();

    RCLCPP_INFO(this->get_logger(),
                "Fast stereo FoundationPose node ready: %s + %s -> %s",
                left_image_topic_.c_str(),
                right_image_topic_.c_str(),
                pose_topic_.c_str());
  }

private:
  using ImageMsg         = sensor_msgs::msg::Image;
  using StereoSyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
  using StereoSyncer     = message_filters::Synchronizer<StereoSyncPolicy>;
  using SubscriberT      = message_filters::Subscriber<ImageMsg>;

  void DeclareParameters()
  {
    this->declare_parameter<int>("image_reliability",
                                 static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));
    this->declare_parameter<std::string>("left_image_topic", "/left/image_raw");
    this->declare_parameter<std::string>("right_image_topic", "/right/image_raw");
    this->declare_parameter<int>("sync_queue_size", 10);
    this->declare_parameter<double>("max_sync_interval_sec", 0.07);

    this->declare_parameter<std::vector<std::string>>("stereo_engine_file_path",
                                                      std::vector<std::string>{});
    this->declare_parameter<std::string>("stereo_model_type", "FAST_FOUNDATION_STEREO");
    this->declare_parameter<int>("model_input_height", 448);
    this->declare_parameter<int>("model_input_width", 640);
    this->declare_parameter<double>("min_depth_meters", 0.1);
    this->declare_parameter<double>("max_depth_meters", 100.0);
    this->declare_parameter<std::string>("caminfo_path", "caminfo.txt");

    this->declare_parameter<std::string>("refiner_engine_path", "");
    this->declare_parameter<std::string>("scorer_engine_path", "");
    this->declare_parameter<std::string>("mesh_path", "");
    this->declare_parameter<std::string>("object_name", "target_object");
    this->declare_parameter<std::string>("mask_image_path", "");
    this->declare_parameter<std::string>("mask_image_directory", "");
    this->declare_parameter<std::string>("mask_image_name", "first_mask.png");
    this->declare_parameter<int>("mask_poll_interval_ms", 200);
    this->declare_parameter<bool>("resize_mask_to_input", false);

    this->declare_parameter<std::string>("pose_topic", "/foundationpose/pose");
    this->declare_parameter<std::string>("pose_frame_id", "");
    this->declare_parameter<std::string>("visualization_topic", "/foundationpose/visualization");
    this->declare_parameter<std::string>("visualization_window_name", "foundationpose_visualization");
    this->declare_parameter<bool>("publish_visualization", false);
    this->declare_parameter<bool>("show_visualization_window", false);
    this->declare_parameter<bool>("save_frame_outputs", false);
    this->declare_parameter<std::string>("frame_output_dir",
                                         "/home/bit/ffs+fp+sam/fp/stereo_tracker_fast_outputs");
    this->declare_parameter<double>("fx", 0.0);
    this->declare_parameter<double>("fy", 0.0);
    this->declare_parameter<double>("cx", 0.0);
    this->declare_parameter<double>("cy", 0.0);
    this->declare_parameter<int>("max_input_image_height", 1080);
    this->declare_parameter<int>("max_input_image_width", 1920);
    this->declare_parameter<int>("register_refine_iterations", 5);
    this->declare_parameter<int>("track_refine_iterations", 2);
  }

  void LoadParameters()
  {
    image_reliability_     = this->get_parameter("image_reliability").as_int();
    left_image_topic_      = this->get_parameter("left_image_topic").as_string();
    right_image_topic_     = this->get_parameter("right_image_topic").as_string();
    sync_queue_size_       = this->get_parameter("sync_queue_size").as_int();
    max_sync_interval_sec_ = this->get_parameter("max_sync_interval_sec").as_double();

    stereo_engine_file_path_ = this->get_parameter("stereo_engine_file_path").as_string_array();
    stereo_model_type_       = this->get_parameter("stereo_model_type").as_string();
    model_input_height_      = this->get_parameter("model_input_height").as_int();
    model_input_width_       = this->get_parameter("model_input_width").as_int();
    min_depth_meters_        = this->get_parameter("min_depth_meters").as_double();
    max_depth_meters_        = this->get_parameter("max_depth_meters").as_double();
    caminfo_path_            = this->get_parameter("caminfo_path").as_string();

    refiner_engine_path_ = this->get_parameter("refiner_engine_path").as_string();
    scorer_engine_path_  = this->get_parameter("scorer_engine_path").as_string();
    mesh_path_           = this->get_parameter("mesh_path").as_string();
    object_name_         = this->get_parameter("object_name").as_string();
    pose_topic_          = this->get_parameter("pose_topic").as_string();
    pose_frame_id_       = this->get_parameter("pose_frame_id").as_string();
    visualization_topic_ = this->get_parameter("visualization_topic").as_string();
    visualization_window_name_ = this->get_parameter("visualization_window_name").as_string();
    publish_visualization_ = this->get_parameter("publish_visualization").as_bool();
    show_visualization_window_ = this->get_parameter("show_visualization_window").as_bool();
    save_frame_outputs_ = this->get_parameter("save_frame_outputs").as_bool();
    frame_output_dir_ = this->get_parameter("frame_output_dir").as_string();
    mask_poll_interval_ms_ = this->get_parameter("mask_poll_interval_ms").as_int();
    resize_mask_to_input_  = this->get_parameter("resize_mask_to_input").as_bool();
    max_input_image_height_ = this->get_parameter("max_input_image_height").as_int();
    max_input_image_width_  = this->get_parameter("max_input_image_width").as_int();
    register_refine_iters_ =
        static_cast<size_t>(this->get_parameter("register_refine_iterations").as_int());
    track_refine_iters_ =
        static_cast<size_t>(this->get_parameter("track_refine_iterations").as_int());

    intrinsic_ = BuildIntrinsicMatrix(this->get_parameter("fx").as_double(),
                                      this->get_parameter("fy").as_double(),
                                      this->get_parameter("cx").as_double(),
                                      this->get_parameter("cy").as_double());
    mask_image_path_ = ResolveMaskImagePath();

    std::string calibration_error;
    if (!LoadStereoCalibrationFromTxt(caminfo_path_, &calibration_, &calibration_error))
    {
      throw std::runtime_error(calibration_error);
    }

    ValidateRequiredFile(refiner_engine_path_, "refiner_engine_path");
    ValidateRequiredFile(scorer_engine_path_, "scorer_engine_path");
    ValidateRequiredFile(mesh_path_, "mesh_path");
    if (mask_image_path_.empty())
    {
      throw std::invalid_argument(
          "Set mask_image_path, or set both mask_image_directory and mask_image_name.");
    }
    if (save_frame_outputs_ && frame_output_dir_.empty())
    {
      throw std::invalid_argument("frame_output_dir must not be empty when save_frame_outputs is true.");
    }
  }

  std::string ResolveMaskImagePath() const
  {
    const auto direct_path = this->get_parameter("mask_image_path").as_string();
    if (!direct_path.empty())
    {
      return direct_path;
    }

    const auto directory = this->get_parameter("mask_image_directory").as_string();
    const auto name      = this->get_parameter("mask_image_name").as_string();
    if (directory.empty() || name.empty())
    {
      return "";
    }

    return (std::filesystem::path(directory) / name).string();
  }

  void ValidateRequiredFile(const std::string &file_path, const std::string &param_name) const
  {
    if (file_path.empty())
    {
      throw std::invalid_argument("Required parameter is empty: " + param_name);
    }
    if (!std::filesystem::exists(file_path))
    {
      throw std::invalid_argument("Path from parameter '" + param_name + "' does not exist: " +
                                  file_path);
    }
  }

  void BuildStereoEstimator()
  {
    if (stereo_model_type_ != "FAST_FOUNDATION_STEREO")
    {
      throw std::runtime_error("Unsupported stereo model type: " + stereo_model_type_);
    }
    if (stereo_engine_file_path_.size() != 2)
    {
      throw std::runtime_error("FAST_FOUNDATION_STEREO expects exactly 2 stereo engine paths.");
    }

    ValidateRequiredFile(stereo_engine_file_path_[0], "stereo_engine_file_path[0]");
    ValidateRequiredFile(stereo_engine_file_path_[1], "stereo_engine_file_path[1]");
    stereo_estimator_ = std::make_unique<FastFoundationStereoEstimator>(stereo_engine_file_path_[0],
                                                                        stereo_engine_file_path_[1],
                                                                        model_input_height_,
                                                                        model_input_width_);
  }

  void BuildPoseModel()
  {
    auto refiner_core = inference_core::CreateTrtInferCore(
        refiner_engine_path_,
        {{"transf_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}},
         {"render_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}}},
        {{"trans", {kPoseBatchSize, 3}}, {"rot", {kPoseBatchSize, 3}}},
        1);

    auto scorer_core = inference_core::CreateTrtInferCore(
        scorer_engine_path_,
        {{"transf_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}},
         {"render_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}}},
        {{"scores", {kPoseBatchSize, 1}}},
        1);

    mesh_loader_ = detection_6d::CreateAssimpMeshLoader(object_name_, mesh_path_);
    foundation_pose_ =
        detection_6d::CreateFoundationPoseModel(refiner_core,
                                                scorer_core,
                                                {mesh_loader_},
                                                intrinsic_,
                                                max_input_image_height_,
                                                max_input_image_width_);
  }

  void SetupRosInterfaces()
  {
    image_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(image_reliability_));
    image_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    image_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, 10);
    if (publish_visualization_)
    {
      visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visualization_topic_, 10);
    }

    if (save_frame_outputs_)
    {
      std::filesystem::create_directories(std::filesystem::path(frame_output_dir_) / "visualization");
      pose_output_path_ = (std::filesystem::path(frame_output_dir_) / "poses.csv").string();
      pose_output_stream_.open(pose_output_path_, std::ios::out | std::ios::trunc);
      if (!pose_output_stream_.is_open())
      {
        throw std::runtime_error("Failed to open fast tracker pose output file: " + pose_output_path_);
      }

      pose_output_stream_
          << "frame_index,mode,stamp_sec,stamp_nanosec,frame_id,visualization_path,"
             "tx,ty,tz,qx,qy,qz,qw,rot_x_deg,rot_y_deg,rot_z_deg,"
             "delta_rot_x_deg,delta_rot_y_deg,delta_rot_z_deg,"
             "m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33\n";
      RCLCPP_INFO(this->get_logger(),
                  "Saving per-frame pose and visualization outputs to: %s",
                  frame_output_dir_.c_str());
    }

    left_image_sub_.subscribe(this, left_image_topic_, image_qos_profile_.get_rmw_qos_profile());
    right_image_sub_.subscribe(this, right_image_topic_, image_qos_profile_.get_rmw_qos_profile());

    stereo_sync_ = std::make_shared<StereoSyncer>(
        StereoSyncPolicy(sync_queue_size_), left_image_sub_, right_image_sub_);
    stereo_sync_->getPolicy()->setMaxIntervalDuration(
        rclcpp::Duration::from_seconds(max_sync_interval_sec_));
    stereo_sync_->registerCallback(std::bind(&FoundationPoseStereoTrackerFastNode::StereoCallback,
                                             this,
                                             std::placeholders::_1,
                                             std::placeholders::_2));
  }

  bool EnsureRectificationMaps(const cv::Size &image_size)
  {
    if (rectification_ready_ && rectification_maps_.image_size == image_size)
    {
      return true;
    }

    std::string rectification_error;
    if (!ComputeStereoRectificationMaps(calibration_, image_size, &rectification_maps_, &rectification_error))
    {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to compute stereo rectification maps: %s",
                   rectification_error.c_str());
      return false;
    }

    rectification_ready_ = true;
    return true;
  }

  bool BuildDepthFromStereo(const cv::Mat &left_stereo_input,
                            const cv::Mat &right_stereo_input,
                            cv::Mat       *depth_for_pose)
  {
    if (depth_for_pose == nullptr || left_stereo_input.size() != right_stereo_input.size())
    {
      return false;
    }
    if (!EnsureRectificationMaps(left_stereo_input.size()))
    {
      return false;
    }

    cv::Mat left_rectified;
    cv::Mat right_rectified;
    cv::remap(left_stereo_input,
              left_rectified,
              rectification_maps_.left_map_x,
              rectification_maps_.left_map_y,
              cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);
    cv::remap(right_stereo_input,
              right_rectified,
              rectification_maps_.right_map_x,
              rectification_maps_.right_map_y,
              cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);

    const PreparedImage prepared_left =
        PrepareStereoInput(left_rectified, model_input_width_, model_input_height_);
    const PreparedImage prepared_right =
        PrepareStereoInput(right_rectified, model_input_width_, model_input_height_);
    if (prepared_left.image.empty() || prepared_right.image.empty())
    {
      return false;
    }

    cv::Mat disparity;
    if (!stereo_estimator_->inference(prepared_left.image, prepared_right.image, disparity))
    {
      return false;
    }

    const double baseline_meters = ComputeRectifiedBaselineMeters(rectification_maps_);
    const CameraIntrinsics intrinsics = ComputePreparedIntrinsics(rectification_maps_, prepared_left);
    if (baseline_meters <= 0.0 || intrinsics.fx <= 0.0)
    {
      return false;
    }

    const cv::Mat depth_model = DisparityToDepthMeters(disparity,
                                                       static_cast<float>(intrinsics.fx),
                                                       static_cast<float>(baseline_meters),
                                                       static_cast<float>(min_depth_meters_),
                                                       static_cast<float>(max_depth_meters_));

    const cv::Mat depth_rectified_full = RestoreToRectifiedResolution(depth_model,
                                                                      prepared_left,
                                                                      left_rectified.size(),
                                                                      cv::INTER_NEAREST);
    *depth_for_pose = AlignRectifiedDepthToOriginalLeft(depth_rectified_full,
                                                       rectification_maps_,
                                                       cv::INTER_NEAREST);
    return !depth_for_pose->empty();
  }

  void StereoCallback(const ImageMsg::ConstSharedPtr &left_msg,
                      const ImageMsg::ConstSharedPtr &right_msg)
  {
    std::unique_lock<std::mutex> lock(process_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
      return;
    }

    if (!CheckTimestamp(left_msg->header.stamp, right_msg->header.stamp, max_sync_interval_sec_))
    {
      return;
    }

    try
    {
      const cv::Mat pose_rgb           = ConvertPoseRgbImage(left_msg);
      const cv::Mat left_stereo_input  = ConvertStereoInputImage(left_msg);
      const cv::Mat right_stereo_input = ConvertStereoInputImage(right_msg);

      cv::Mat depth;
      if (!BuildDepthFromStereo(left_stereo_input, right_stereo_input, &depth))
      {
        return;
      }
      if (pose_rgb.size() != depth.size())
      {
        return;
      }

      if (!has_pose_)
      {
        TryInitialRegistration(left_msg->header, pose_rgb, depth);
        return;
      }

      RunTracking(left_msg->header, pose_rgb, depth);
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000, "Fast stereo callback failed: %s", e.what());
    }
  }

  void TryInitialRegistration(const std_msgs::msg::Header &header,
                              const cv::Mat               &rgb,
                              const cv::Mat               &depth)
  {
    const auto now = std::chrono::steady_clock::now();
    if (last_mask_attempt_time_.time_since_epoch().count() != 0 &&
        now - last_mask_attempt_time_ < std::chrono::milliseconds(mask_poll_interval_ms_))
    {
      return;
    }
    last_mask_attempt_time_ = now;

    cv::Mat mask = cv::imread(mask_image_path_, cv::IMREAD_UNCHANGED);
    if (mask.empty())
    {
      return;
    }

    mask = ConvertMaskImage(mask);
    if (mask.size() != rgb.size())
    {
      if (!resize_mask_to_input_)
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             2000,
                             "Mask size %dx%d does not match RGBD size %dx%d.",
                             mask.cols,
                             mask.rows,
                             rgb.cols,
                             rgb.rows);
        return;
      }
      cv::resize(mask, mask, rgb.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }

    Eigen::Matrix4f pose;
    if (!foundation_pose_->Register(rgb, depth, mask, object_name_, pose, register_refine_iters_))
    {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000, "Initial FoundationPose registration failed.");
      return;
    }

    has_pose_  = true;
    last_pose_ = pose;
    PublishPose(header, last_pose_);
    SaveFrameOutput(header, rgb, last_pose_, "register");
    PublishVisualization(header, rgb, last_pose_);
    RCLCPP_INFO(this->get_logger(), "Initial registration succeeded. Fast direct tracking started.");
  }

  void RunTracking(const std_msgs::msg::Header &header, const cv::Mat &rgb, const cv::Mat &depth)
  {
    Eigen::Matrix4f pose;
    if (!foundation_pose_->Track(rgb, depth, last_pose_, object_name_, pose, track_refine_iters_))
    {
      return;
    }

    last_pose_ = pose;
    PublishPose(header, last_pose_);
    SaveFrameOutput(header, rgb, last_pose_, "track");
    PublishVisualization(header, rgb, last_pose_);
  }

  void PublishPose(std_msgs::msg::Header header, const Eigen::Matrix4f &pose)
  {
    if (!pose_frame_id_.empty())
    {
      header.frame_id = pose_frame_id_;
    }
    pose_pub_->publish(PoseMatrixToPoseStamped(pose, header));
  }

  void SaveFrameOutput(const std_msgs::msg::Header &header,
                       const cv::Mat               &rgb,
                       const Eigen::Matrix4f       &pose,
                       const std::string           &mode)
  {
    if (!save_frame_outputs_)
    {
      return;
    }

    const size_t      frame_index = saved_frame_index_++;
    const std::string frame_stem  = NextFrameStemFromIndex(frame_index);
    const auto        visualization_path =
        std::filesystem::path(frame_output_dir_) / "visualization" / (frame_stem + ".png");

    const cv::Mat visualization_bgr = BuildPoseVisualizationBgr(rgb, pose);
    if (!cv::imwrite(visualization_path.string(), visualization_bgr))
    {
      RCLCPP_ERROR_THROTTLE(this->get_logger(),
                            *this->get_clock(),
                            2000,
                            "Failed to save pose visualization frame: %s",
                            visualization_path.string().c_str());
      return;
    }

    if (pose_output_stream_.is_open())
    {
      auto output_header = header;
      if (!pose_frame_id_.empty())
      {
        output_header.frame_id = pose_frame_id_;
      }

      const Eigen::Vector3f rotation_rpy_degrees =
          RotationMatrixToRpyDegrees(pose.block<3, 3>(0, 0));
      Eigen::Vector3f delta_rotation_rpy_degrees = Eigen::Vector3f::Zero();
      if (has_last_saved_pose_)
      {
        const Eigen::Matrix3f relative_rotation =
            last_saved_pose_.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0);
        delta_rotation_rpy_degrees = RotationMatrixToRpyDegrees(relative_rotation);
      }

      pose_output_stream_ << PoseToCsvRow(frame_index,
                                          mode,
                                          output_header,
                                          visualization_path.string(),
                                          pose,
                                          rotation_rpy_degrees,
                                          delta_rotation_rpy_degrees)
                          << '\n';
      pose_output_stream_.flush();
      last_saved_pose_ = pose;
      has_last_saved_pose_ = true;
    }
  }

  std::string NextFrameStemFromIndex(size_t frame_index) const
  {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << frame_index;
    return stream.str();
  }

  cv::Mat BuildPoseVisualizationBgr(const cv::Mat &rgb, const Eigen::Matrix4f &pose) const
  {
    cv::Mat visualization_bgr;
    cv::cvtColor(rgb, visualization_bgr, cv::COLOR_RGB2BGR);

    const auto draw_pose = detection_6d::ConvertPoseMesh2BBox(pose, mesh_loader_);
    Draw3DBoundingBox(intrinsic_, draw_pose, mesh_loader_->GetObjectDimension(), visualization_bgr);
    return visualization_bgr;
  }

  void PublishVisualization(std_msgs::msg::Header header,
                            const cv::Mat        &rgb,
                            const Eigen::Matrix4f &pose)
  {
    if (!publish_visualization_ && !show_visualization_window_)
    {
      return;
    }

    cv::Mat visualization_bgr = BuildPoseVisualizationBgr(rgb, pose);

    if (publish_visualization_ && visualization_pub_ != nullptr)
    {
      auto image_msg =
          cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, visualization_bgr).toImageMsg();
      visualization_pub_->publish(*image_msg);
    }

    if (show_visualization_window_)
    {
      cv::imshow(visualization_window_name_, visualization_bgr);
      cv::waitKey(1);
    }
  }

  int         image_reliability_{1};
  std::string left_image_topic_;
  std::string right_image_topic_;
  int         sync_queue_size_{10};
  double      max_sync_interval_sec_{0.07};

  std::vector<std::string> stereo_engine_file_path_;
  std::string              stereo_model_type_;
  int                      model_input_height_{448};
  int                      model_input_width_{640};
  double                   min_depth_meters_{0.1};
  double                   max_depth_meters_{100.0};
  std::string              caminfo_path_;

  std::string refiner_engine_path_;
  std::string scorer_engine_path_;
  std::string mesh_path_;
  std::string object_name_;
  std::string mask_image_path_;
  std::string pose_topic_;
  std::string pose_frame_id_;
  std::string visualization_topic_;
  std::string visualization_window_name_;

  int    max_input_image_height_{1080};
  int    max_input_image_width_{1920};
  int    mask_poll_interval_ms_{200};
  bool   resize_mask_to_input_{false};
  bool   publish_visualization_{false};
  bool   show_visualization_window_{false};
  bool   save_frame_outputs_{false};
  size_t register_refine_iters_{5};
  size_t track_refine_iters_{2};
  size_t saved_frame_index_{0};
  bool has_last_saved_pose_{false};
  std::string frame_output_dir_;
  std::string pose_output_path_;
  std::ofstream pose_output_stream_;
  Eigen::Matrix4f last_saved_pose_{Eigen::Matrix4f::Identity()};

  rclcpp::QoS image_qos_profile_;
  StereoCalibration       calibration_;
  StereoRectificationMaps rectification_maps_;
  bool                    rectification_ready_{false};
  std::unique_ptr<StereoEstimator> stereo_estimator_;
  Eigen::Matrix3f                 intrinsic_{Eigen::Matrix3f::Identity()};

  bool            has_pose_{false};
  Eigen::Matrix4f last_pose_{Eigen::Matrix4f::Identity()};
  std::chrono::steady_clock::time_point last_mask_attempt_time_{};
  std::mutex process_mutex_;

  std::shared_ptr<detection_6d::BaseMeshLoader>         mesh_loader_;
  std::shared_ptr<detection_6d::Base6DofDetectionModel> foundation_pose_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         visualization_pub_;
  SubscriberT                                                   left_image_sub_;
  SubscriberT                                                   right_image_sub_;
  std::shared_ptr<StereoSyncer>                                 stereo_sync_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    rclcpp::spin(std::make_shared<FoundationPoseStereoTrackerFastNode>());
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("foundationpose_stereo_tracker_fast_node"),
                 "Node startup failed: %s",
                 e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
