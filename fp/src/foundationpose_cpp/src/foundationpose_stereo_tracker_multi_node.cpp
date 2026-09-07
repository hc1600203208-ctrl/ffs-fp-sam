#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
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

constexpr std::uint64_t kPoseBatchSize = 252;
constexpr std::uint64_t kTrackingBatchSize = 252;
constexpr int kCropHeight = 160;
constexpr int kCropWidth = 160;
constexpr float kRadiansToDegrees = 57.29577951308232F;

struct PreparedImage
{
  cv::Mat image;
  double scale{1.0};
  int pad_left{0};
  int pad_top{0};
  int resized_width{0};
  int resized_height{0};
};

struct CameraIntrinsics
{
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
};

std::string SanitizeName(const std::string &name)
{
  std::string result;
  result.reserve(name.size());
  for (const unsigned char character : name)
  {
    if (std::isalnum(character))
    {
      result.push_back(static_cast<char>(std::tolower(character)));
    }
    else if (character == '_' || character == '-')
    {
      result.push_back(static_cast<char>(character));
    }
    else
    {
      result.push_back('_');
    }
  }
  return result.empty() ? "object" : result;
}

bool CheckTimestamp(const builtin_interfaces::msg::Time &left,
                    const builtin_interfaces::msg::Time &right,
                    double max_delta_sec)
{
  const double left_seconds = static_cast<double>(left.sec) + static_cast<double>(left.nanosec) / 1e9;
  const double right_seconds = static_cast<double>(right.sec) + static_cast<double>(right.nanosec) / 1e9;
  return std::fabs(left_seconds - right_seconds) <= max_delta_sec;
}

Eigen::Matrix3f BuildIntrinsicMatrix(double fx, double fy, double cx, double cy)
{
  if (fx <= 0.0 || fy <= 0.0)
  {
    throw std::invalid_argument("Camera intrinsics fx/fy must be positive");
  }
  Eigen::Matrix3f intrinsic = Eigen::Matrix3f::Identity();
  intrinsic(0, 0) = static_cast<float>(fx);
  intrinsic(1, 1) = static_cast<float>(fy);
  intrinsic(0, 2) = static_cast<float>(cx);
  intrinsic(1, 2) = static_cast<float>(cy);
  return intrinsic;
}

cv::Mat RosImageToCvMat(const sensor_msgs::msg::Image::ConstSharedPtr &message)
{
  return cv_bridge::toCvShare(message, message->encoding)->image;
}

cv::Mat ConvertPoseRgbImage(const sensor_msgs::msg::Image::ConstSharedPtr &message)
{
  const cv::Mat image = RosImageToCvMat(message);
  if (message->encoding == sensor_msgs::image_encodings::RGB8)
  {
    return image;
  }

  cv::Mat rgb;
  if (message->encoding == sensor_msgs::image_encodings::BGR8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  }
  else if (message->encoding == sensor_msgs::image_encodings::RGBA8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_RGBA2RGB);
  }
  else if (message->encoding == sensor_msgs::image_encodings::BGRA8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  }
  else if (message->encoding == sensor_msgs::image_encodings::MONO8 ||
           message->encoding == sensor_msgs::image_encodings::TYPE_8UC1)
  {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  }
  else
  {
    throw std::runtime_error("Unsupported RGB image encoding: " + message->encoding);
  }
  return rgb;
}

cv::Mat ConvertStereoInputImage(const sensor_msgs::msg::Image::ConstSharedPtr &message)
{
  const cv::Mat image = RosImageToCvMat(message);
  if (image.channels() == 1 || image.channels() == 3)
  {
    return image;
  }
  if (image.channels() == 4)
  {
    cv::Mat converted;
    cv::cvtColor(image, converted, cv::COLOR_BGRA2BGR);
    return converted;
  }
  throw std::runtime_error("Unsupported stereo image channel count");
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
    throw std::runtime_error("Mask image must have 1, 3, or 4 channels");
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
    throw std::runtime_error("Unsupported image channel count");
  }

  result.scale = std::min(static_cast<double>(target_width) / color.cols,
                          static_cast<double>(target_height) / color.rows);
  result.resized_width = std::max(1, static_cast<int>(std::round(color.cols * result.scale)));
  result.resized_height = std::max(1, static_cast<int>(std::round(color.rows * result.scale)));

  cv::Mat resized;
  cv::resize(color, resized, cv::Size(result.resized_width, result.resized_height));
  const int pad_width = target_width - result.resized_width;
  const int pad_height = target_height - result.resized_height;
  result.pad_left = pad_width / 2;
  result.pad_top = pad_height / 2;
  cv::copyMakeBorder(resized, result.image, result.pad_top, pad_height - result.pad_top,
                     result.pad_left, pad_width - result.pad_left, cv::BORDER_REPLICATE);
  return result;
}

double ComputeRectifiedBaselineMeters(const StereoRectificationMaps &maps)
{
  if (maps.p2.empty())
  {
    return 0.0;
  }
  const double fx = maps.p2.at<double>(0, 0);
  return std::abs(fx) > 1e-9 ? std::abs(maps.p2.at<double>(0, 3) / fx) : 0.0;
}

CameraIntrinsics ComputePreparedIntrinsics(const StereoRectificationMaps &maps,
                                           const PreparedImage &prepared)
{
  CameraIntrinsics intrinsics;
  if (maps.p1.empty())
  {
    return intrinsics;
  }
  intrinsics.fx = maps.p1.at<double>(0, 0) * prepared.scale;
  intrinsics.fy = maps.p1.at<double>(1, 1) * prepared.scale;
  intrinsics.cx = maps.p1.at<double>(0, 2) * prepared.scale + prepared.pad_left;
  intrinsics.cy = maps.p1.at<double>(1, 2) * prepared.scale + prepared.pad_top;
  return intrinsics;
}

cv::Mat DisparityToDepthMeters(const cv::Mat &disparity, float fx, float baseline,
                               float min_depth_meters, float max_depth_meters)
{
  cv::Mat depth = cv::Mat::zeros(disparity.size(), CV_32FC1);
  const float numerator = fx * baseline;
  for (int y = 0; y < disparity.rows; ++y)
  {
    const float *disparity_row = disparity.ptr<float>(y);
    float *depth_row = depth.ptr<float>(y);
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

cv::Mat RestoreToRectifiedResolution(const cv::Mat &image, const PreparedImage &prepared,
                                     const cv::Size &rectified_size)
{
  if (image.empty())
  {
    return {};
  }
  const cv::Rect valid_roi(prepared.pad_left, prepared.pad_top,
                           std::min(prepared.resized_width, image.cols - prepared.pad_left),
                           std::min(prepared.resized_height, image.rows - prepared.pad_top));
  if (valid_roi.width <= 0 || valid_roi.height <= 0)
  {
    return {};
  }
  cv::Mat restored;
  cv::resize(image(valid_roi), restored, rectified_size, 0, 0, cv::INTER_NEAREST);
  return restored;
}

geometry_msgs::msg::PoseStamped PoseMatrixToPoseStamped(const Eigen::Matrix4f &pose,
                                                        const std_msgs::msg::Header &header)
{
  geometry_msgs::msg::PoseStamped message;
  message.header = header;
  message.pose.position.x = pose(0, 3);
  message.pose.position.y = pose(1, 3);
  message.pose.position.z = pose(2, 3);
  Eigen::Quaternionf quaternion(pose.block<3, 3>(0, 0));
  quaternion.normalize();
  message.pose.orientation.x = quaternion.x();
  message.pose.orientation.y = quaternion.y();
  message.pose.orientation.z = quaternion.z();
  message.pose.orientation.w = quaternion.w();
  return message;
}

Eigen::Vector3f RotationMatrixToRpyDegrees(const Eigen::Matrix3f &rotation)
{
  const float sy = std::sqrt(rotation(0, 0) * rotation(0, 0) + rotation(1, 0) * rotation(1, 0));
  const bool singular = sy < 1e-6F;
  const float roll = singular ? std::atan2(-rotation(1, 2), rotation(1, 1))
                              : std::atan2(rotation(2, 1), rotation(2, 2));
  const float pitch = std::atan2(-rotation(2, 0), sy);
  const float yaw = singular ? 0.0F : std::atan2(rotation(1, 0), rotation(0, 0));
  return Eigen::Vector3f(roll, pitch, yaw) * kRadiansToDegrees;
}

void DrawTextWithOutline(cv::Mat &image, const std::string &text, const cv::Point &origin,
                         double scale, const cv::Scalar &color)
{
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3,
              cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

void DrawFirstPoseOverlay(cv::Mat &image, const Eigen::Matrix4f &pose)
{
  const Eigen::Vector3f translation = pose.block<3, 1>(0, 3);
  const Eigen::Vector3f rpy = RotationMatrixToRpyDegrees(pose.block<3, 3>(0, 0));
  const std::array<std::string, 7> lines = {
      "Filtered pose", "X: " + cv::format("%.4f", translation.x()) + " m",
      "Y: " + cv::format("%.4f", translation.y()) + " m",
      "Z: " + cv::format("%.4f", translation.z()) + " m",
      "Roll X:  " + cv::format("%.2f", rpy.x()),
      "Pitch Y: " + cv::format("%.2f", rpy.y()),
      "Yaw Z:   " + cv::format("%.2f", rpy.z())};
  for (std::size_t index = 0; index < lines.size(); ++index)
  {
    DrawTextWithOutline(image, lines[index], cv::Point(12, 24 + static_cast<int>(index) * 24),
                        0.58, cv::Scalar(255, 255, 255));
  }
}

void Draw3DBoundingBox(const Eigen::Matrix3f &intrinsic, const Eigen::Matrix4f &pose,
                       const Eigen::Vector3f &dimension, const cv::Scalar &box_color,
                       const std::string &object_name, cv::Mat &image)
{
  const float half_length = dimension(0) / 2.0F;
  const float half_width = dimension(1) / 2.0F;
  const float half_height = dimension(2) / 2.0F;
  const std::array<Eigen::Vector3f, 8> points = {
      Eigen::Vector3f(-half_length, -half_width, half_height),
      Eigen::Vector3f(half_length, -half_width, half_height),
      Eigen::Vector3f(half_length, half_width, half_height),
      Eigen::Vector3f(-half_length, half_width, half_height),
      Eigen::Vector3f(-half_length, -half_width, -half_height),
      Eigen::Vector3f(half_length, -half_width, -half_height),
      Eigen::Vector3f(half_length, half_width, -half_height),
      Eigen::Vector3f(-half_length, half_width, -half_height)};

  std::array<cv::Point2f, 8> image_points;
  for (std::size_t index = 0; index < points.size(); ++index)
  {
    const Eigen::Vector4f point = pose * Eigen::Vector4f(points[index].x(), points[index].y(),
                                                           points[index].z(), 1.0F);
    if (point.z() <= 1e-6F)
    {
      return;
    }
    image_points[index] = cv::Point2f(intrinsic(0, 0) * point.x() / point.z() + intrinsic(0, 2),
                                       intrinsic(1, 1) * point.y() / point.z() + intrinsic(1, 2));
  }

  const std::array<std::pair<int, int>, 12> edges = {
      std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}, std::pair{3, 0},
      std::pair{4, 5}, std::pair{5, 6}, std::pair{6, 7}, std::pair{7, 4},
      std::pair{0, 4}, std::pair{1, 5}, std::pair{2, 6}, std::pair{3, 7}};
  for (const auto &[start, end] : edges)
  {
    cv::line(image, image_points[start], image_points[end], box_color, 2, cv::LINE_AA);
  }

  float minimum_x = image_points[0].x;
  float minimum_y = image_points[0].y;
  for (const auto &point : image_points)
  {
    minimum_x = std::min(minimum_x, point.x);
    minimum_y = std::min(minimum_y, point.y);
  }
  const cv::Point label_origin(std::max(0, static_cast<int>(minimum_x)),
                               std::max(18, static_cast<int>(minimum_y) - 6));
  DrawTextWithOutline(image, object_name, label_origin, 0.55, box_color);

  const Eigen::Vector4f center = pose * Eigen::Vector4f(0.0F, 0.0F, 0.0F, 1.0F);
  if (center.z() <= 1e-6F)
  {
    return;
  }
  const cv::Point center_point(intrinsic(0, 0) * center.x() / center.z() + intrinsic(0, 2),
                               intrinsic(1, 1) * center.y() / center.z() + intrinsic(1, 2));
  const float axis_length = (dimension(0) + dimension(1) + dimension(2)) / 6.0F;
  const std::array<Eigen::Vector4f, 3> axis_end_points = {
      pose * Eigen::Vector4f(axis_length, 0.0F, 0.0F, 1.0F),
      pose * Eigen::Vector4f(0.0F, 0.0F, axis_length, 1.0F),
      pose * Eigen::Vector4f(0.0F, axis_length, 0.0F, 1.0F)};
  const std::array<cv::Scalar, 3> axis_colors = {
      cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};
  for (std::size_t index = 0; index < axis_end_points.size(); ++index)
  {
    const auto &end = axis_end_points[index];
    if (end.z() <= 1e-6F)
    {
      continue;
    }
    const cv::Point end_point(intrinsic(0, 0) * end.x() / end.z() + intrinsic(0, 2),
                              intrinsic(1, 1) * end.y() / end.z() + intrinsic(1, 2));
    cv::line(image, center_point, end_point, axis_colors[index], 3, cv::LINE_AA);
  }
}

}  // namespace

class FoundationPoseStereoTrackerMultiNode final : public rclcpp::Node
{
public:
  FoundationPoseStereoTrackerMultiNode()
  : Node("foundationpose_stereo_tracker_multi_node"), image_qos_profile_(rclcpp::QoS(10))
  {
    DeclareParameters();
    LoadParameters();
    BuildStereoEstimator();
    BuildObjectTrackers();
    SetupRosInterfaces();
  }

  ~FoundationPoseStereoTrackerMultiNode() override
  {
    StopTrackingWorkers();
  }

private:
  using ImageMsg = sensor_msgs::msg::Image;
  using StereoSyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
  using StereoSyncer = message_filters::Synchronizer<StereoSyncPolicy>;
  using SubscriberT = message_filters::Subscriber<ImageMsg>;

  struct PoseResult
  {
    std::size_t object_index{0};
    bool success{false};
    Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
  };

  struct TrackingWorker
  {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable job_ready;
    std::condition_variable job_finished;
    bool stop{false};
    bool has_job{false};
    bool finished{false};
    cv::Mat rgb;
    cv::Mat depth;
    Eigen::Matrix4f hypothesis{Eigen::Matrix4f::Identity()};
    PoseResult result;
  };

  struct TrackedObject
  {
    std::string name;
    std::string mask_path;
    cv::Scalar box_color;
    std::shared_ptr<detection_6d::BaseMeshLoader> mesh_loader;
    std::shared_ptr<detection_6d::Base6DofDetectionModel> foundation_pose;
    std::unique_ptr<TrackingWorker> tracking_worker;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher;
    bool has_pose{false};
    bool has_smoothed_pose{false};
    Eigen::Matrix4f last_pose{Eigen::Matrix4f::Identity()};
    Eigen::Matrix4f smoothed_pose{Eigen::Matrix4f::Identity()};
  };

  void DeclareParameters()
  {
    declare_parameter<int>("image_reliability", static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));
    declare_parameter<std::string>("left_image_topic", "/left/image_raw");
    declare_parameter<std::string>("right_image_topic", "/right/image_raw");
    declare_parameter<int>("sync_queue_size", 10);
    declare_parameter<double>("max_sync_interval_sec", 0.07);
    declare_parameter<std::vector<std::string>>("stereo_engine_file_path", std::vector<std::string>{});
    declare_parameter<std::string>("stereo_model_type", "FAST_FOUNDATION_STEREO");
    declare_parameter<int>("model_input_height", 448);
    declare_parameter<int>("model_input_width", 640);
    declare_parameter<double>("min_depth_meters", 0.1);
    declare_parameter<double>("max_depth_meters", 100.0);
    declare_parameter<std::string>("caminfo_path", "caminfo.txt");
    declare_parameter<std::string>("refiner_engine_path", "");
    declare_parameter<std::string>("scorer_engine_path", "");
    declare_parameter<std::vector<std::string>>("tracked_object_names", std::vector<std::string>{});
    declare_parameter<std::vector<std::string>>("tracked_mesh_paths", std::vector<std::string>{});
    declare_parameter<std::string>("mask_image_directory", "");
    declare_parameter<std::vector<std::string>>("tracked_mask_image_names", std::vector<std::string>{});
    declare_parameter<int>("mask_poll_interval_ms", 200);
    declare_parameter<bool>("resize_mask_to_input", false);
    declare_parameter<std::string>("pose_topic", "/foundationpose/pose");
    declare_parameter<std::string>("pose_frame_id", "");
    declare_parameter<std::string>("visualization_topic", "/foundationpose/visualization");
    declare_parameter<std::string>("visualization_window_name", "foundationpose_multi_visualization");
    declare_parameter<bool>("publish_visualization", false);
    declare_parameter<bool>("show_visualization_window", false);
    declare_parameter<bool>("enable_pose_smoothing", true);
    declare_parameter<double>("pose_smoothing_translation_alpha", 0.45);
    declare_parameter<double>("pose_smoothing_rotation_alpha", 0.45);
    declare_parameter<double>("fx", 0.0);
    declare_parameter<double>("fy", 0.0);
    declare_parameter<double>("cx", 0.0);
    declare_parameter<double>("cy", 0.0);
    declare_parameter<int>("max_input_image_height", 480);
    declare_parameter<int>("max_input_image_width", 640);
    declare_parameter<int>("register_refine_iterations", 5);
    declare_parameter<int>("track_refine_iterations", 2);
  }

  void LoadParameters()
  {
    image_reliability_ = get_parameter("image_reliability").as_int();
    left_image_topic_ = get_parameter("left_image_topic").as_string();
    right_image_topic_ = get_parameter("right_image_topic").as_string();
    sync_queue_size_ = get_parameter("sync_queue_size").as_int();
    max_sync_interval_sec_ = get_parameter("max_sync_interval_sec").as_double();
    stereo_engine_file_paths_ = get_parameter("stereo_engine_file_path").as_string_array();
    stereo_model_type_ = get_parameter("stereo_model_type").as_string();
    model_input_height_ = get_parameter("model_input_height").as_int();
    model_input_width_ = get_parameter("model_input_width").as_int();
    min_depth_meters_ = get_parameter("min_depth_meters").as_double();
    max_depth_meters_ = get_parameter("max_depth_meters").as_double();
    caminfo_path_ = get_parameter("caminfo_path").as_string();
    refiner_engine_path_ = get_parameter("refiner_engine_path").as_string();
    scorer_engine_path_ = get_parameter("scorer_engine_path").as_string();
    object_names_ = get_parameter("tracked_object_names").as_string_array();
    mesh_paths_ = get_parameter("tracked_mesh_paths").as_string_array();
    mask_image_directory_ = get_parameter("mask_image_directory").as_string();
    mask_image_names_ = get_parameter("tracked_mask_image_names").as_string_array();
    mask_poll_interval_ms_ = get_parameter("mask_poll_interval_ms").as_int();
    resize_mask_to_input_ = get_parameter("resize_mask_to_input").as_bool();
    pose_topic_ = get_parameter("pose_topic").as_string();
    pose_frame_id_ = get_parameter("pose_frame_id").as_string();
    visualization_topic_ = get_parameter("visualization_topic").as_string();
    visualization_window_name_ = get_parameter("visualization_window_name").as_string();
    publish_visualization_ = get_parameter("publish_visualization").as_bool();
    show_visualization_window_ = get_parameter("show_visualization_window").as_bool();
    enable_pose_smoothing_ = get_parameter("enable_pose_smoothing").as_bool();
    translation_alpha_ = std::clamp(get_parameter("pose_smoothing_translation_alpha").as_double(), 0.0, 1.0);
    rotation_alpha_ = std::clamp(get_parameter("pose_smoothing_rotation_alpha").as_double(), 0.0, 1.0);
    max_input_image_height_ = get_parameter("max_input_image_height").as_int();
    max_input_image_width_ = get_parameter("max_input_image_width").as_int();
    register_refine_iterations_ = static_cast<std::size_t>(get_parameter("register_refine_iterations").as_int());
    track_refine_iterations_ = static_cast<std::size_t>(get_parameter("track_refine_iterations").as_int());
    intrinsic_ = BuildIntrinsicMatrix(get_parameter("fx").as_double(), get_parameter("fy").as_double(),
                                      get_parameter("cx").as_double(), get_parameter("cy").as_double());

    if (object_names_.empty() || object_names_.size() != mesh_paths_.size() ||
        object_names_.size() != mask_image_names_.size())
    {
      throw std::invalid_argument(
          "tracked_object_names, tracked_mesh_paths, and tracked_mask_image_names must have the same non-zero length");
    }
    if (mask_image_directory_.empty())
    {
      throw std::invalid_argument("mask_image_directory must not be empty");
    }
    if (pose_topic_.empty())
    {
      throw std::invalid_argument("pose_topic must not be empty");
    }
    std::unordered_set<std::string> names;
    for (std::size_t index = 0; index < object_names_.size(); ++index)
    {
      if (object_names_[index].empty() || mesh_paths_[index].empty() || mask_image_names_[index].empty())
      {
        throw std::invalid_argument("tracked object names, mesh paths, and mask names must not be empty");
      }
      if (!names.insert(SanitizeName(object_names_[index])).second)
      {
        throw std::invalid_argument("tracked_object_names must map to unique ROS topic suffixes");
      }
      ValidateRequiredFile(mesh_paths_[index], "tracked_mesh_paths[" + std::to_string(index) + "]");
    }
    ValidateRequiredFile(refiner_engine_path_, "refiner_engine_path");
    ValidateRequiredFile(scorer_engine_path_, "scorer_engine_path");
    std::string calibration_error;
    if (!LoadStereoCalibrationFromTxt(caminfo_path_, &calibration_, &calibration_error))
    {
      throw std::runtime_error(calibration_error);
    }
  }

  void ValidateRequiredFile(const std::string &path, const std::string &parameter_name) const
  {
    if (path.empty() || !std::filesystem::exists(path))
    {
      throw std::invalid_argument("Path from parameter '" + parameter_name + "' does not exist: " + path);
    }
  }

  void BuildStereoEstimator()
  {
    if (stereo_model_type_ != "FAST_FOUNDATION_STEREO" || stereo_engine_file_paths_.size() != 2)
    {
      throw std::runtime_error("FAST_FOUNDATION_STEREO requires exactly two stereo engine paths");
    }
    ValidateRequiredFile(stereo_engine_file_paths_[0], "stereo_engine_file_path[0]");
    ValidateRequiredFile(stereo_engine_file_paths_[1], "stereo_engine_file_path[1]");
    stereo_estimator_ = std::make_unique<FastFoundationStereoEstimator>(
        stereo_engine_file_paths_[0], stereo_engine_file_paths_[1], model_input_height_, model_input_width_);
  }

  std::shared_ptr<inference_core::BaseInferCore> CreateRefinerCore(std::uint64_t pose_batch_size) const
  {
    return inference_core::CreateTrtInferCore(
        refiner_engine_path_,
        {{"transf_input", {pose_batch_size, kCropHeight, kCropWidth, 6}},
         {"render_input", {pose_batch_size, kCropHeight, kCropWidth, 6}}},
        {{"trans", {pose_batch_size, 3}}, {"rot", {pose_batch_size, 3}}}, 1);
  }

  std::shared_ptr<inference_core::BaseInferCore> CreateScorerCore(std::uint64_t pose_batch_size) const
  {
    return inference_core::CreateTrtInferCore(
        scorer_engine_path_,
        {{"transf_input", {pose_batch_size, kCropHeight, kCropWidth, 6}},
         {"render_input", {pose_batch_size, kCropHeight, kCropWidth, 6}}},
        {{"scores", {pose_batch_size, 1}}}, 1);
  }

  std::shared_ptr<detection_6d::Base6DofDetectionModel> CreatePoseModel(
      const TrackedObject &object, std::uint64_t pose_batch_size) const
  {
    return detection_6d::CreateFoundationPoseModel(
        CreateRefinerCore(pose_batch_size), CreateScorerCore(pose_batch_size), {object.mesh_loader}, intrinsic_,
        max_input_image_height_, max_input_image_width_);
  }

  void BuildObjectTrackers()
  {
    static const std::array<cv::Scalar, 8> colors = {
        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
        cv::Scalar(0, 165, 255), cv::Scalar(203, 192, 255), cv::Scalar(128, 255, 0),
        cv::Scalar(255, 128, 0), cv::Scalar(0, 128, 255)};
    tracked_objects_.reserve(object_names_.size());
    for (std::size_t index = 0; index < object_names_.size(); ++index)
    {
      TrackedObject object;
      object.name = object_names_[index];
      object.mask_path = (std::filesystem::path(mask_image_directory_) / mask_image_names_[index]).string();
      object.box_color = colors[index % colors.size()];
      object.mesh_loader = detection_6d::CreateAssimpMeshLoader(object.name, mesh_paths_[index]);
      tracked_objects_.push_back(std::move(object));
    }
  }

  void SetupRosInterfaces()
  {
    image_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(image_reliability_));
    image_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    image_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);
    primary_pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, 10);
    for (auto &object : tracked_objects_)
    {
      object.pose_publisher = create_publisher<geometry_msgs::msg::PoseStamped>(
          pose_topic_ + "/" + SanitizeName(object.name), 10);
    }
    if (publish_visualization_)
    {
      visualization_publisher_ = create_publisher<sensor_msgs::msg::Image>(visualization_topic_, 10);
    }
    left_image_sub_.subscribe(this, left_image_topic_, image_qos_profile_.get_rmw_qos_profile());
    right_image_sub_.subscribe(this, right_image_topic_, image_qos_profile_.get_rmw_qos_profile());
    stereo_sync_ = std::make_shared<StereoSyncer>(StereoSyncPolicy(sync_queue_size_), left_image_sub_, right_image_sub_);
    stereo_sync_->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_sync_interval_sec_));
    stereo_sync_->registerCallback(std::bind(&FoundationPoseStereoTrackerMultiNode::StereoCallback, this,
                                             std::placeholders::_1, std::placeholders::_2));
  }

  bool EnsureRectificationMaps(const cv::Size &image_size)
  {
    if (rectification_ready_ && rectification_maps_.image_size == image_size)
    {
      return true;
    }
    std::string error;
    if (!ComputeStereoRectificationMaps(calibration_, image_size, &rectification_maps_, &error))
    {
      return false;
    }
    rectification_ready_ = true;
    return true;
  }

  bool BuildDepthFromStereo(const cv::Mat &left, const cv::Mat &right, cv::Mat *depth)
  {
    if (depth == nullptr || left.size() != right.size() || !EnsureRectificationMaps(left.size()))
    {
      return false;
    }
    cv::Mat left_rectified;
    cv::Mat right_rectified;
    cv::remap(left, left_rectified, rectification_maps_.left_map_x, rectification_maps_.left_map_y,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(right, right_rectified, rectification_maps_.right_map_x, rectification_maps_.right_map_y,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    const PreparedImage prepared_left = PrepareStereoInput(left_rectified, model_input_width_, model_input_height_);
    const PreparedImage prepared_right = PrepareStereoInput(right_rectified, model_input_width_, model_input_height_);
    if (prepared_left.image.empty() || prepared_right.image.empty())
    {
      return false;
    }
    cv::Mat disparity;
    if (!stereo_estimator_->inference(prepared_left.image, prepared_right.image, disparity))
    {
      return false;
    }
    const double baseline = ComputeRectifiedBaselineMeters(rectification_maps_);
    const CameraIntrinsics intrinsics = ComputePreparedIntrinsics(rectification_maps_, prepared_left);
    if (baseline <= 0.0 || intrinsics.fx <= 0.0)
    {
      return false;
    }
    const cv::Mat depth_model = DisparityToDepthMeters(
        disparity, static_cast<float>(intrinsics.fx), static_cast<float>(baseline),
        static_cast<float>(min_depth_meters_), static_cast<float>(max_depth_meters_));
    const cv::Mat depth_rectified = RestoreToRectifiedResolution(depth_model, prepared_left, left_rectified.size());
    *depth = AlignRectifiedDepthToOriginalLeft(depth_rectified, rectification_maps_, cv::INTER_NEAREST);
    return !depth->empty();
  }

  bool AllObjectsRegistered() const
  {
    return std::all_of(tracked_objects_.begin(), tracked_objects_.end(),
                       [](const TrackedObject &object) { return object.has_pose; });
  }

  void StartTrackingWorker(std::size_t index)
  {
    auto &object = tracked_objects_.at(index);
    object.tracking_worker = std::make_unique<TrackingWorker>();
    object.tracking_worker->thread = std::thread(
        [this, index]() { TrackingWorkerLoop(index); });
  }

  void StopTrackingWorkers() noexcept
  {
    for (auto &object : tracked_objects_)
    {
      if (object.tracking_worker == nullptr)
      {
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(object.tracking_worker->mutex);
        object.tracking_worker->stop = true;
      }
      object.tracking_worker->job_ready.notify_one();
    }
    for (auto &object : tracked_objects_)
    {
      if (object.tracking_worker != nullptr && object.tracking_worker->thread.joinable())
      {
        object.tracking_worker->thread.join();
      }
    }
  }

  void TrackingWorkerLoop(std::size_t index)
  {
    auto &object = tracked_objects_.at(index);
    auto &worker = *object.tracking_worker;
    std::unique_lock<std::mutex> lock(worker.mutex);
    while (true)
    {
      worker.job_ready.wait(lock, [&worker]() { return worker.stop || worker.has_job; });
      if (worker.stop)
      {
        return;
      }

      const cv::Mat rgb = worker.rgb;
      const cv::Mat depth = worker.depth;
      const Eigen::Matrix4f hypothesis = worker.hypothesis;
      worker.has_job = false;
      lock.unlock();

      PoseResult result;
      result.object_index = index;
      try
      {
        result.success = object.foundation_pose->Track(
            rgb, depth, hypothesis, object.name, result.pose, track_refine_iterations_);
      }
      catch (const std::exception &)
      {
        result.success = false;
      }

      lock.lock();
      worker.result = std::move(result);
      worker.finished = true;
      worker.job_finished.notify_one();
    }
  }

  void StereoCallback(const ImageMsg::ConstSharedPtr &left_message, const ImageMsg::ConstSharedPtr &right_message)
  {
    std::unique_lock<std::mutex> lock(process_mutex_, std::try_to_lock);
    if (!lock.owns_lock() ||
        !CheckTimestamp(left_message->header.stamp, right_message->header.stamp, max_sync_interval_sec_))
    {
      return;
    }
    try
    {
      const cv::Mat rgb = ConvertPoseRgbImage(left_message);
      cv::Mat depth;
      if (!BuildDepthFromStereo(ConvertStereoInputImage(left_message), ConvertStereoInputImage(right_message), &depth) ||
          rgb.size() != depth.size())
      {
        return;
      }
      if (!AllObjectsRegistered())
      {
        TryInitialRegistration(left_message->header, rgb, depth);
        return;
      }
      RunTracking(left_message->header, rgb, depth);
    }
    catch (const std::exception &error)
    {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Multi-object stereo callback failed: %s", error.what());
    }
  }

  bool LoadAllMasks(const cv::Size &image_size, std::vector<cv::Mat> *masks) const
  {
    masks->clear();
    masks->reserve(tracked_objects_.size());
    for (const auto &object : tracked_objects_)
    {
      cv::Mat mask = cv::imread(object.mask_path, cv::IMREAD_UNCHANGED);
      if (mask.empty())
      {
        return false;
      }
      mask = ConvertMaskImage(mask);
      if (mask.size() != image_size)
      {
        if (!resize_mask_to_input_)
        {
          return false;
        }
        cv::resize(mask, mask, image_size, 0.0, 0.0, cv::INTER_NEAREST);
      }
      if (cv::countNonZero(mask) == 0)
      {
        return false;
      }
      masks->push_back(mask);
    }
    return true;
  }

  void TryInitialRegistration(const std_msgs::msg::Header &header, const cv::Mat &rgb, const cv::Mat &depth)
  {
    const auto now = std::chrono::steady_clock::now();
    if (last_mask_attempt_time_.time_since_epoch().count() != 0 &&
        now - last_mask_attempt_time_ < std::chrono::milliseconds(mask_poll_interval_ms_))
    {
      return;
    }
    last_mask_attempt_time_ = now;
    std::vector<cv::Mat> masks;
    if (!LoadAllMasks(rgb.size(), &masks))
    {
      return;
    }

    bool updated = false;
    for (std::size_t index = 0; index < tracked_objects_.size(); ++index)
    {
      if (tracked_objects_[index].has_pose)
      {
        continue;
      }
      Eigen::Matrix4f pose;
      bool success = false;
      try
      {
        {
          const auto registration_model = CreatePoseModel(tracked_objects_[index], kPoseBatchSize);
          success = registration_model->Register(
              rgb, depth, masks[index], tracked_objects_[index].name, pose, register_refine_iterations_);
        }
        if (success)
        {
          tracked_objects_[index].foundation_pose = CreatePoseModel(tracked_objects_[index], kTrackingBatchSize);
          StartTrackingWorker(index);
        }
      }
      catch (const std::exception &)
      {
        success = false;
      }
      if (success)
      {
        tracked_objects_[index].has_pose = true;
        tracked_objects_[index].last_pose = pose;
        updated = true;
      }
    }
    if (updated)
    {
      PublishCurrentResults(header, rgb, "register");
    }
  }

  void RunTracking(const std_msgs::msg::Header &header, const cv::Mat &rgb, const cv::Mat &depth)
  {
    for (std::size_t index = 0; index < tracked_objects_.size(); ++index)
    {
      auto &worker = *tracked_objects_[index].tracking_worker;
      {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.rgb = rgb;
        worker.depth = depth;
        worker.hypothesis = tracked_objects_[index].last_pose;
        worker.finished = false;
        worker.has_job = true;
      }
      worker.job_ready.notify_one();
    }

    bool updated = false;
    for (std::size_t index = 0; index < tracked_objects_.size(); ++index)
    {
      auto &worker = *tracked_objects_[index].tracking_worker;
      std::unique_lock<std::mutex> lock(worker.mutex);
      worker.job_finished.wait(lock, [&worker]() { return worker.finished; });
      const PoseResult result = worker.result;
      if (result.success)
      {
        tracked_objects_[result.object_index].last_pose = result.pose;
        updated = true;
      }
    }
    if (updated)
    {
      PublishCurrentResults(header, rgb, "track");
    }
  }

  Eigen::Matrix4f SmoothPoseForDisplay(TrackedObject &object, const Eigen::Matrix4f &raw_pose) const
  {
    if (!enable_pose_smoothing_)
    {
      object.has_smoothed_pose = false;
      return raw_pose;
    }
    Eigen::Quaternionf raw_quaternion(raw_pose.block<3, 3>(0, 0));
    raw_quaternion.normalize();
    if (!object.has_smoothed_pose)
    {
      object.smoothed_pose = raw_pose;
      object.smoothed_pose.block<3, 3>(0, 0) = raw_quaternion.toRotationMatrix();
      object.has_smoothed_pose = true;
      return object.smoothed_pose;
    }
    Eigen::Quaternionf previous_quaternion(object.smoothed_pose.block<3, 3>(0, 0));
    previous_quaternion.normalize();
    if (previous_quaternion.dot(raw_quaternion) < 0.0F)
    {
      raw_quaternion.coeffs() *= -1.0F;
    }
    const Eigen::Vector3f previous_translation = object.smoothed_pose.block<3, 1>(0, 3);
    const Eigen::Vector3f raw_translation = raw_pose.block<3, 1>(0, 3);
    object.smoothed_pose.block<3, 1>(0, 3) = previous_translation +
        static_cast<float>(translation_alpha_) * (raw_translation - previous_translation);
    object.smoothed_pose.block<3, 3>(0, 0) =
        previous_quaternion.slerp(static_cast<float>(rotation_alpha_), raw_quaternion).normalized().toRotationMatrix();
    return object.smoothed_pose;
  }

  std_msgs::msg::Header PoseHeader(const std_msgs::msg::Header &header) const
  {
    auto output_header = header;
    if (!pose_frame_id_.empty())
    {
      output_header.frame_id = pose_frame_id_;
    }
    return output_header;
  }

  void PublishCurrentResults(const std_msgs::msg::Header &header, const cv::Mat &rgb, const std::string &mode)
  {
    const auto output_header = PoseHeader(header);
    for (std::size_t index = 0; index < tracked_objects_.size(); ++index)
    {
      auto &object = tracked_objects_[index];
      if (!object.has_pose)
      {
        continue;
      }
      const Eigen::Matrix4f display_pose = SmoothPoseForDisplay(object, object.last_pose);
      const auto pose_message = PoseMatrixToPoseStamped(object.last_pose, output_header);
      object.pose_publisher->publish(pose_message);
      if (index == 0)
      {
        primary_pose_publisher_->publish(pose_message);
        PrintFirstPose(output_header, display_pose, mode);
      }
    }
    PublishVisualization(output_header, rgb);
  }

  void PrintFirstPose(const std_msgs::msg::Header &header, const Eigen::Matrix4f &pose,
                      const std::string &mode) const
  {
    const Eigen::Vector3f rpy = RotationMatrixToRpyDegrees(pose.block<3, 3>(0, 0));
    Eigen::Quaternionf quaternion(pose.block<3, 3>(0, 0));
    quaternion.normalize();
    RCLCPP_INFO(get_logger(),
                "FILTERED_POSE object=%s mode=%s stamp=%d.%09u frame=%s xyz_m=[%.6f %.6f %.6f] "
                "rpy_deg=[%.3f %.3f %.3f] quat=[%.6f %.6f %.6f %.6f]",
                tracked_objects_.front().name.c_str(), mode.c_str(), header.stamp.sec, header.stamp.nanosec,
                header.frame_id.c_str(), pose(0, 3), pose(1, 3), pose(2, 3), rpy.x(), rpy.y(), rpy.z(),
                quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
  }

  cv::Mat BuildVisualization(const cv::Mat &rgb) const
  {
    cv::Mat visualization;
    cv::cvtColor(rgb, visualization, cv::COLOR_RGB2BGR);
    for (const auto &object : tracked_objects_)
    {
      if (!object.has_pose)
      {
        continue;
      }
      const auto draw_pose = detection_6d::ConvertPoseMesh2BBox(object.last_pose, object.mesh_loader);
      Draw3DBoundingBox(intrinsic_, draw_pose, object.mesh_loader->GetObjectDimension(), object.box_color,
                        object.name, visualization);
    }
    return visualization;
  }

  void PublishVisualization(const std_msgs::msg::Header &header, const cv::Mat &rgb)
  {
    if (!publish_visualization_ && !show_visualization_window_)
    {
      return;
    }
    cv::Mat visualization = BuildVisualization(rgb);
    if (publish_visualization_ && visualization_publisher_)
    {
      auto message = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, visualization).toImageMsg();
      visualization_publisher_->publish(*message);
    }
    if (show_visualization_window_)
    {
      if (!tracked_objects_.empty() && tracked_objects_.front().has_pose)
      {
        DrawFirstPoseOverlay(visualization, tracked_objects_.front().smoothed_pose);
      }
      cv::imshow(visualization_window_name_, visualization);
      cv::waitKey(1);
    }
  }

  int image_reliability_{1};
  std::string left_image_topic_;
  std::string right_image_topic_;
  int sync_queue_size_{10};
  double max_sync_interval_sec_{0.07};
  std::vector<std::string> stereo_engine_file_paths_;
  std::string stereo_model_type_;
  int model_input_height_{448};
  int model_input_width_{640};
  double min_depth_meters_{0.1};
  double max_depth_meters_{100.0};
  std::string caminfo_path_;
  std::string refiner_engine_path_;
  std::string scorer_engine_path_;
  std::vector<std::string> object_names_;
  std::vector<std::string> mesh_paths_;
  std::string mask_image_directory_;
  std::vector<std::string> mask_image_names_;
  int mask_poll_interval_ms_{200};
  bool resize_mask_to_input_{false};
  std::string pose_topic_;
  std::string pose_frame_id_;
  std::string visualization_topic_;
  std::string visualization_window_name_;
  bool publish_visualization_{false};
  bool show_visualization_window_{false};
  bool enable_pose_smoothing_{true};
  double translation_alpha_{0.45};
  double rotation_alpha_{0.45};
  int max_input_image_height_{480};
  int max_input_image_width_{640};
  std::size_t register_refine_iterations_{5};
  std::size_t track_refine_iterations_{2};
  Eigen::Matrix3f intrinsic_{Eigen::Matrix3f::Identity()};
  StereoCalibration calibration_;
  StereoRectificationMaps rectification_maps_;
  bool rectification_ready_{false};
  std::unique_ptr<StereoEstimator> stereo_estimator_;
  std::vector<TrackedObject> tracked_objects_;
  std::chrono::steady_clock::time_point last_mask_attempt_time_{};
  std::mutex process_mutex_;
  rclcpp::QoS image_qos_profile_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr primary_pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visualization_publisher_;
  SubscriberT left_image_sub_;
  SubscriberT right_image_sub_;
  std::shared_ptr<StereoSyncer> stereo_sync_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    rclcpp::spin(std::make_shared<FoundationPoseStereoTrackerMultiNode>());
  }
  catch (const std::exception &error)
  {
    RCLCPP_FATAL(rclcpp::get_logger("foundationpose_stereo_tracker_multi_node"),
                 "Node startup failed: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
