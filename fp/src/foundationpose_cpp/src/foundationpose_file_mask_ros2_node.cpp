#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "trt_core/trt_core.h"

namespace
{

constexpr int kPoseBatchSize = 252;
constexpr int kCropHeight    = 160;
constexpr int kCropWidth     = 160;

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

cv::Mat ConvertRgbImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  const auto cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
  const auto &image = cv_ptr->image;

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

  throw std::runtime_error("Unsupported RGB encoding: " + msg->encoding);
}

cv::Mat ConvertDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg, double depth_scale)
{
  const auto cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
  const auto &image = cv_ptr->image;

  cv::Mat depth;
  if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
      msg->encoding == sensor_msgs::image_encodings::MONO16)
  {
    image.convertTo(depth, CV_32FC1, depth_scale);
    return depth;
  }

  if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    depth = image.clone();
    cv::patchNaNs(depth, 0.0);
    return depth;
  }

  if (image.channels() != 1)
  {
    throw std::runtime_error("Depth image must be single-channel, got encoding: " + msg->encoding);
  }

  image.convertTo(depth, CV_32FC1, depth_scale);
  cv::patchNaNs(depth, 0.0);
  return depth;
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

geometry_msgs::msg::PoseStamped PoseMatrixToPoseStamped(const Eigen::Matrix4f &pose,
                                                        const std_msgs::msg::Header &header)
{
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = header;

  pose_msg.pose.position.x = pose(0, 3);
  pose_msg.pose.position.y = pose(1, 3);
  pose_msg.pose.position.z = pose(2, 3);

  Eigen::Matrix3f rotation = pose.block<3, 3>(0, 0);
  Eigen::Quaternionf quat(rotation);
  quat.normalize();

  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  pose_msg.pose.orientation.w = quat.w();
  return pose_msg;
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
      {-half_l, -half_w, half_h},  {half_l, -half_w, half_h},  {half_l, half_w, half_h},
      {-half_l, half_w, half_h},   {-half_l, -half_w, -half_h}, {half_l, -half_w, -half_h},
      {half_l, half_w, -half_h},   {-half_l, half_w, -half_h}
  };

  Eigen::Vector4f transformed_points[8];
  for (int i = 0; i < 8; ++i)
  {
    transformed_points[i] = pose * Eigen::Vector4f(points[i](0), points[i](1), points[i](2), 1.0F);
  }

  std::vector<cv::Point2f> image_points;
  image_points.reserve(8);
  for (int i = 0; i < 8; ++i)
  {
    const float z = transformed_points[i](2);
    if (z <= 1e-6F)
    {
      return;
    }

    const float x = transformed_points[i](0) / z;
    const float y = transformed_points[i](1) / z;
    const float u = intrinsic(0, 0) * x + intrinsic(0, 2);
    const float v = intrinsic(1, 1) * y + intrinsic(1, 2);
    image_points.emplace_back(u, v);
  }

  const std::vector<std::pair<int, int>> edges = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},
      {4, 5}, {5, 6}, {6, 7}, {7, 4},
      {0, 4}, {1, 5}, {2, 6}, {3, 7}
  };

  for (const auto &edge : edges)
  {
    cv::line(image, image_points[edge.first], image_points[edge.second], cv::Scalar(0, 255, 0), 2);
  }

  const Eigen::Vector4f center_world = pose * Eigen::Vector4f(0.0F, 0.0F, 0.0F, 1.0F);
  if (center_world(2) <= 1e-6F)
  {
    return;
  }

  const float axis_length = (dimension(0) + dimension(1) + dimension(2)) / 6.0F;
  const std::vector<Eigen::Vector4f> axis_end_points = {
      pose * Eigen::Vector4f(axis_length, 0.0F, 0.0F, 1.0F),
      pose * Eigen::Vector4f(0.0F, axis_length, 0.0F, 1.0F),
      pose * Eigen::Vector4f(0.0F, 0.0F, axis_length, 1.0F)
  };
  const std::vector<cv::Scalar> axis_colors = {
      cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)
  };

  const float center_u =
      intrinsic(0, 0) * (center_world(0) / center_world(2)) + intrinsic(0, 2);
  const float center_v =
      intrinsic(1, 1) * (center_world(1) / center_world(2)) + intrinsic(1, 2);
  const cv::Point center_pt(center_u, center_v);

  for (size_t i = 0; i < axis_end_points.size(); ++i)
  {
    if (axis_end_points[i](2) <= 1e-6F)
    {
      continue;
    }

    const float end_u =
        intrinsic(0, 0) * (axis_end_points[i](0) / axis_end_points[i](2)) + intrinsic(0, 2);
    const float end_v =
        intrinsic(1, 1) * (axis_end_points[i](1) / axis_end_points[i](2)) + intrinsic(1, 2);
    cv::line(image, center_pt, cv::Point(end_u, end_v), axis_colors[i], 3);
  }
}

} // namespace

class FoundationPoseFileMaskTrackerNode : public rclcpp::Node
{
public:
  FoundationPoseFileMaskTrackerNode()
      : Node("foundationpose_file_mask_tracker_node"),
        rgb_sub_(),
        depth_sub_()
  {
    DeclareParameters();
    LoadParameters();
    BuildModel();
    SetupRosInterfaces();

    RCLCPP_INFO(this->get_logger(),
                "FoundationPose file-mask ROS 2 node is ready. Waiting for synchronized "
                "RGB/Depth frames and mask file: %s",
                mask_image_path_.c_str());
  }

private:
  using ImageMsg        = sensor_msgs::msg::Image;
  using TrackSyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
  using TrackSyncer     = message_filters::Synchronizer<TrackSyncPolicy>;
  using SubscriberT     = message_filters::Subscriber<ImageMsg>;

  void DeclareParameters()
  {
    this->declare_parameter<std::string>("refiner_engine_path", "");
    this->declare_parameter<std::string>("scorer_engine_path", "");
    this->declare_parameter<std::string>("mesh_path", "");
    this->declare_parameter<std::string>("object_name", "target_object");
    this->declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
    this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    this->declare_parameter<std::string>("mask_image_path", "");
    this->declare_parameter<std::string>("mask_image_directory", "");
    this->declare_parameter<std::string>("mask_image_name", "mask.png");
    this->declare_parameter<std::string>("pose_topic", "/foundationpose/pose");
    this->declare_parameter<std::string>("pose_frame_id", "");
    this->declare_parameter<std::string>("visualization_topic", "/foundationpose/visualization");
    this->declare_parameter<std::string>("visualization_window_name", "foundationpose_visualization");

    this->declare_parameter<double>("fx", 0.0);
    this->declare_parameter<double>("fy", 0.0);
    this->declare_parameter<double>("cx", 0.0);
    this->declare_parameter<double>("cy", 0.0);
    this->declare_parameter<double>("depth_scale", 0.001);
    this->declare_parameter<double>("max_sync_interval_sec", 0.025);

    this->declare_parameter<int>("max_input_image_height", 1080);
    this->declare_parameter<int>("max_input_image_width", 1920);
    this->declare_parameter<int>("register_refine_iterations", 5);
    this->declare_parameter<int>("track_refine_iterations", 2);
    this->declare_parameter<int>("sync_queue_size", 5);
    this->declare_parameter<int>("mask_poll_interval_ms", 200);
    this->declare_parameter<bool>("resize_mask_to_input", false);
    this->declare_parameter<bool>("publish_visualization", true);
    this->declare_parameter<bool>("show_visualization_window", false);
  }

  void LoadParameters()
  {
    refiner_engine_path_ = this->get_parameter("refiner_engine_path").as_string();
    scorer_engine_path_  = this->get_parameter("scorer_engine_path").as_string();
    mesh_path_           = this->get_parameter("mesh_path").as_string();
    object_name_         = this->get_parameter("object_name").as_string();
    rgb_topic_           = this->get_parameter("rgb_topic").as_string();
    depth_topic_         = this->get_parameter("depth_topic").as_string();
    pose_topic_          = this->get_parameter("pose_topic").as_string();
    pose_frame_id_       = this->get_parameter("pose_frame_id").as_string();
    visualization_topic_ = this->get_parameter("visualization_topic").as_string();
    visualization_window_name_ = this->get_parameter("visualization_window_name").as_string();

    depth_scale_               = this->get_parameter("depth_scale").as_double();
    max_sync_interval_sec_     = this->get_parameter("max_sync_interval_sec").as_double();
    max_input_image_height_    = this->get_parameter("max_input_image_height").as_int();
    max_input_image_width_     = this->get_parameter("max_input_image_width").as_int();
    register_refine_iters_     = static_cast<size_t>(this->get_parameter("register_refine_iterations").as_int());
    track_refine_iters_        = static_cast<size_t>(this->get_parameter("track_refine_iterations").as_int());
    sync_queue_size_           = this->get_parameter("sync_queue_size").as_int();
    mask_poll_interval_ms_     = this->get_parameter("mask_poll_interval_ms").as_int();
    resize_mask_to_input_      = this->get_parameter("resize_mask_to_input").as_bool();
    publish_visualization_     = this->get_parameter("publish_visualization").as_bool();
    show_visualization_window_ = this->get_parameter("show_visualization_window").as_bool();

    const double fx = this->get_parameter("fx").as_double();
    const double fy = this->get_parameter("fy").as_double();
    const double cx = this->get_parameter("cx").as_double();
    const double cy = this->get_parameter("cy").as_double();
    intrinsic_       = BuildIntrinsicMatrix(fx, fy, cx, cy);

    mask_image_path_ = ResolveMaskImagePath();

    ValidateRequiredFile(refiner_engine_path_, "refiner_engine_path");
    ValidateRequiredFile(scorer_engine_path_, "scorer_engine_path");
    ValidateRequiredFile(mesh_path_, "mesh_path");
    if (mask_image_path_.empty())
    {
      throw std::invalid_argument(
          "Set mask_image_path, or set both mask_image_directory and mask_image_name.");
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

  void BuildModel()
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
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, 10);
    if (publish_visualization_)
    {
      visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visualization_topic_, 10);
    }

    rgb_sub_.subscribe(this, rgb_topic_, rmw_qos_profile_sensor_data);
    depth_sub_.subscribe(this, depth_topic_, rmw_qos_profile_sensor_data);

    track_sync_ = std::make_shared<TrackSyncer>(TrackSyncPolicy(sync_queue_size_), rgb_sub_, depth_sub_);
    auto *track_policy = track_sync_->getPolicy();
    track_policy->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_sync_interval_sec_));
    track_sync_->registerCallback(std::bind(&FoundationPoseFileMaskTrackerNode::ImageCallback,
                                            this,
                                            std::placeholders::_1,
                                            std::placeholders::_2));

    mask_poll_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(mask_poll_interval_ms_),
        std::bind(&FoundationPoseFileMaskTrackerNode::TryRegisterFromCachedRgbd, this));
  }

  void ImageCallback(const ImageMsg::ConstSharedPtr &rgb_msg,
                     const ImageMsg::ConstSharedPtr &depth_msg)
  {
    std::unique_lock<std::mutex> lock(process_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000, "Dropping frame because inference is busy.");
      return;
    }

    try
    {
      cv::Mat rgb   = ConvertRgbImage(rgb_msg);
      cv::Mat depth = ConvertDepthImage(depth_msg, depth_scale_);
      if (rgb.size() != depth.size())
      {
        throw std::runtime_error("RGB and depth image sizes do not match.");
      }

      if (!has_pose_)
      {
        latest_rgb_    = rgb.clone();
        latest_depth_  = depth.clone();
        latest_header_ = rgb_msg->header;
        has_cached_rgbd_ = true;
        TryRegisterFromCachedRgbdLocked();
        return;
      }

      RunTracking(rgb_msg->header, rgb, depth);
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000, "Image callback failed: %s", e.what());
    }
  }

  void TryRegisterFromCachedRgbd()
  {
    if (has_pose_)
    {
      return;
    }

    std::unique_lock<std::mutex> lock(process_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
      return;
    }

    TryRegisterFromCachedRgbdLocked();
  }

  void TryRegisterFromCachedRgbdLocked()
  {
    if (has_pose_)
    {
      return;
    }

    if (!has_cached_rgbd_)
    {
      RCLCPP_INFO_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           5000,
                           "Waiting for the first synchronized RGB/Depth frame.");
      return;
    }

    cv::Mat mask = cv::imread(mask_image_path_, cv::IMREAD_UNCHANGED);
    if (mask.empty())
    {
      RCLCPP_INFO_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "Waiting for a readable mask image at: %s",
                           mask_image_path_.c_str());
      return;
    }

    mask = ConvertMaskImage(mask);
    if (mask.empty())
    {
      RCLCPP_INFO_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "Mask file exists but decoded to an empty image: %s",
                           mask_image_path_.c_str());
      return;
    }

    if (mask.size() != latest_rgb_.size())
    {
      if (!resize_mask_to_input_)
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             2000,
                             "Mask image size %dx%d does not match latest RGBD size %dx%d. Enable "
                             "resize_mask_to_input to resize it automatically.",
                             mask.cols,
                             mask.rows,
                             latest_rgb_.cols,
                             latest_rgb_.rows);
        return;
      }

      const int original_mask_cols = mask.cols;
      const int original_mask_rows = mask.rows;
      cv::resize(mask, mask, latest_rgb_.size(), 0.0, 0.0, cv::INTER_NEAREST);
      RCLCPP_WARN(this->get_logger(),
                  "Resized file mask from %dx%d to %dx%d to match the latest RGBD frame.",
                  original_mask_cols,
                  original_mask_rows,
                  latest_rgb_.cols,
                  latest_rgb_.rows);
    }

    if (RunInitialRegistration(latest_header_, latest_rgb_, latest_depth_, mask))
    {
      if (mask_poll_timer_ != nullptr)
      {
        mask_poll_timer_->cancel();
        mask_poll_timer_.reset();
      }
    }
  }

  bool RunInitialRegistration(const std_msgs::msg::Header &header,
                              const cv::Mat               &rgb,
                              const cv::Mat               &depth,
                              const cv::Mat               &registration_mask)
  {
    Eigen::Matrix4f pose;
    const bool ok = foundation_pose_->Register(
        rgb, depth, registration_mask, object_name_, pose, register_refine_iters_);
    if (!ok)
    {
      RCLCPP_ERROR_THROTTLE(this->get_logger(),
                            *this->get_clock(),
                            2000,
                            "Initial FoundationPose registration failed. Waiting for another "
                            "RGBD frame and mask file retry.");
      return false;
    }

    has_pose_  = true;
    last_pose_ = pose;
    PublishPose(header, last_pose_);
    PublishVisualization(header, rgb, last_pose_);

    RCLCPP_INFO(this->get_logger(),
                "Initial registration succeeded using file mask '%s' and the latest synchronized "
                "RGB/Depth frame. The node is now tracking with RGB/Depth only.",
                mask_image_path_.c_str());
    return true;
  }

  void RunTracking(const std_msgs::msg::Header &header, const cv::Mat &rgb, const cv::Mat &depth)
  {
    Eigen::Matrix4f tracked_pose;
    const bool ok =
        foundation_pose_->Track(rgb, depth, last_pose_, object_name_, tracked_pose, track_refine_iters_);
    if (!ok)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "FoundationPose tracking failed on the current frame.");
      return;
    }

    last_pose_ = tracked_pose;
    PublishPose(header, last_pose_);
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

  void PublishVisualization(std_msgs::msg::Header header,
                            const cv::Mat        &rgb,
                            const Eigen::Matrix4f &pose)
  {
    if (!publish_visualization_ && !show_visualization_window_)
    {
      return;
    }

    cv::Mat visualization_bgr;
    cv::cvtColor(rgb, visualization_bgr, cv::COLOR_RGB2BGR);

    const auto draw_pose = detection_6d::ConvertPoseMesh2BBox(pose, mesh_loader_);
    Draw3DBoundingBox(intrinsic_, draw_pose, mesh_loader_->GetObjectDimension(), visualization_bgr);

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

private:
  std::string refiner_engine_path_;
  std::string scorer_engine_path_;
  std::string mesh_path_;
  std::string object_name_;
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string mask_image_path_;
  std::string pose_topic_;
  std::string pose_frame_id_;
  std::string visualization_topic_;
  std::string visualization_window_name_;

  double depth_scale_{0.001};
  double max_sync_interval_sec_{0.025};
  int    max_input_image_height_{1080};
  int    max_input_image_width_{1920};
  int    sync_queue_size_{5};
  int    mask_poll_interval_ms_{200};
  bool   resize_mask_to_input_{false};
  bool   publish_visualization_{true};
  bool   show_visualization_window_{false};

  size_t register_refine_iters_{5};
  size_t track_refine_iters_{2};

  Eigen::Matrix3f intrinsic_{Eigen::Matrix3f::Identity()};

  bool                 has_pose_{false};
  bool                 has_cached_rgbd_{false};
  cv::Mat              latest_rgb_;
  cv::Mat              latest_depth_;
  std_msgs::msg::Header latest_header_;
  Eigen::Matrix4f      last_pose_{Eigen::Matrix4f::Identity()};
  std::mutex           process_mutex_;

  std::shared_ptr<detection_6d::BaseMeshLoader>        mesh_loader_;
  std::shared_ptr<detection_6d::Base6DofDetectionModel> foundation_pose_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         visualization_pub_;
  SubscriberT                                                   rgb_sub_;
  SubscriberT                                                   depth_sub_;
  std::shared_ptr<TrackSyncer>                                  track_sync_;
  rclcpp::TimerBase::SharedPtr                                  mask_poll_timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    auto node = std::make_shared<FoundationPoseFileMaskTrackerNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("foundationpose_file_mask_tracker_node"),
                 "Node startup failed: %s",
                 e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
