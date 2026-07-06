#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "trt_core/trt_core.h"

namespace fs = std::filesystem;

// This node runs FoundationPose in an offline mode:
// 1) load a RGBD sequence from disk,
// 2) register the object on an initial frame using the provided mask,
// 3) track the object across the remaining frames,
// 4) publish poses, save CSV results, and optionally export visualizations.
// The implementation is designed for batch processing of a recorded dataset,
// rather than live ROS topics.
namespace
{

// Batch size used by the TensorRT inference backends.
constexpr int kPoseBatchSize = 252;
// Cropped image size expected by the pose refinement/scoring networks.
constexpr int kCropHeight    = 160;
constexpr int kCropWidth     = 160;

// Stores the RGB/depth pair information for one frame in the sequence.
struct FramePaths
{
  std::string id;      // File stem, used as the frame identifier.
  fs::path    rgb_path;
  fs::path    depth_path;
};

// Records the result of one registration or tracking attempt.
// The CSV writer uses this information to save a line for each frame.
struct PoseRecord
{
  std::string     mode;
  bool            success{false};
  Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
};

// Loads and normalizes the object mask for the initial registration frame.
// The mask is expected to indicate the visible object region and is used to
// constrain FoundationPose during the first pose estimation step.
cv::Mat LoadMaskImage(const std::string &mask_path)
{
  cv::Mat mask = cv::imread(mask_path, cv::IMREAD_UNCHANGED);
  if (mask.empty())
  {
    throw std::runtime_error("Failed to read first-frame mask: " + mask_path);
  }

  // Some datasets store masks as RGB(A) images; convert them to a single-channel mask.
  if (mask.channels() == 3)
  {
    cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
  }
  else if (mask.channels() == 4)
  {
    cv::cvtColor(mask, mask, cv::COLOR_BGRA2GRAY);
  }

  // Ensure a standard 8-bit binary mask.
  if (mask.type() != CV_8UC1)
  {
    mask.convertTo(mask, CV_8UC1);
  }

  // Convert any non-zero values to 255 so the mask is a clean binary foreground/background mask.
  cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
  return mask;
}

// Builds a camera intrinsic matrix from the camera parameters.
// The matrix is used to project 3D points into the image plane and to draw
// the object bounding box during visualization.
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

// Reads camera intrinsics from a plain text file.
// The expected format is a 3x3 matrix written row-major in the file.
Eigen::Matrix3f LoadIntrinsicMatrix(const fs::path &path)
{
  std::ifstream file(path);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open camera intrinsic file: " + path.string());
  }

  double k00, k01, k02;
  double k10, k11, k12;
  double k20, k21, k22;
  file >> k00 >> k01 >> k02 >> k10 >> k11 >> k12 >> k20 >> k21 >> k22;
  if (file.fail())
  {
    throw std::runtime_error("Failed to parse camera intrinsic file: " + path.string());
  }

  Eigen::Matrix3f intrinsic;
  intrinsic << static_cast<float>(k00), static_cast<float>(k01), static_cast<float>(k02),
      static_cast<float>(k10), static_cast<float>(k11), static_cast<float>(k12),
      static_cast<float>(k20), static_cast<float>(k21), static_cast<float>(k22);
  return intrinsic;
}

// Scans a sequence directory and pairs each RGB image with its matching depth image.
// Matching is done by file stem (for example, 000001.png in RGB and 000001.png in depth).
std::vector<FramePaths> CollectFramePaths(const fs::path    &sequence_dir,
                                          const std::string &rgb_subdir,
                                          const std::string &depth_subdir)
{
  const fs::path rgb_dir   = sequence_dir / rgb_subdir;
  const fs::path depth_dir = sequence_dir / depth_subdir;

  if (!fs::exists(rgb_dir) || !fs::is_directory(rgb_dir))
  {
    throw std::invalid_argument("RGB directory does not exist: " + rgb_dir.string());
  }

  if (!fs::exists(depth_dir) || !fs::is_directory(depth_dir))
  {
    throw std::invalid_argument("Depth directory does not exist: " + depth_dir.string());
  }

  // Build a map from depth file stem to its full path for fast lookup.
  std::unordered_map<std::string, fs::path> depth_by_stem;
  for (const auto &entry : fs::directory_iterator(depth_dir))
  {
    if (entry.is_regular_file())
    {
      depth_by_stem.emplace(entry.path().stem().string(), entry.path());
    }
  }

  std::vector<FramePaths> frames;
  for (const auto &entry : fs::directory_iterator(rgb_dir))
  {
    if (!entry.is_regular_file())
    {
      continue;
    }

    const std::string stem = entry.path().stem().string();
    const auto        iter = depth_by_stem.find(stem);
    if (iter == depth_by_stem.end())
    {
      continue;
    }

    frames.push_back(FramePaths{stem, entry.path(), iter->second});
  }

  // Sort by frame id so the sequence order is deterministic.
  std::sort(frames.begin(), frames.end(), [](const FramePaths &lhs, const FramePaths &rhs) {
    return lhs.id < rhs.id;
  });

  if (frames.empty())
  {
    throw std::runtime_error("No matching RGB/Depth frame pairs found in " + sequence_dir.string());
  }

  return frames;
}

// Loads RGB images and standardizes them to 3-channel RGB format.
// FoundationPose expects RGB input in a format that is easy to process by OpenCV.
cv::Mat LoadRgbImage(const fs::path &path)
{
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
  if (image.empty())
  {
    throw std::runtime_error("Failed to read RGB image: " + path.string());
  }

  cv::Mat rgb;
  if (image.channels() == 4)
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
    return rgb;
  }

  if (image.channels() == 3)
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    return rgb;
  }

  if (image.channels() == 1)
  {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
  }

  throw std::runtime_error("Unsupported RGB image channel count: " + std::to_string(image.channels()) +
                           " for " + path.string());
}

// Loads a depth image and normalizes it to meters (or the configured unit).
// The depth scale is applied so that raw integer pixel values are converted to a
// floating-point depth map that FoundationPose can use reliably.
cv::Mat LoadDepthImage(const fs::path &path, double depth_scale)
{
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
  if (image.empty())
  {
    throw std::runtime_error("Failed to read depth image: " + path.string());
  }

  if (image.channels() != 1)
  {
    throw std::runtime_error("Depth image must be single-channel: " + path.string());
  }

  cv::Mat depth;
  switch (image.type())
  {
  case CV_16UC1:
  case CV_16SC1:
  case CV_8UC1:
    image.convertTo(depth, CV_32FC1, depth_scale);
    break;
  case CV_32FC1:
    depth = image.clone();
    break;
  default:
    image.convertTo(depth, CV_32FC1, depth_scale);
    break;
  }

  // Replace invalid values with zero so that downstream geometry operations do not fail.
  cv::patchNaNs(depth, 0.0);
  return depth;
}

geometry_msgs::msg::PoseStamped PoseMatrixToPoseStamped(const Eigen::Matrix4f &pose,
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

std::string PoseToCsvRow(const std::string &frame_id, const PoseRecord &record)
{
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(8);

  stream << frame_id << ',' << record.mode << ',' << (record.success ? 1 : 0);
  if (!record.success)
  {
    stream << ",0,0,0,0,0,0,1";
    for (int row = 0; row < 4; ++row)
    {
      for (int col = 0; col < 4; ++col)
      {
        stream << ',' << (row == col ? 1.0F : 0.0F);
      }
    }
    return stream.str();
  }

  Eigen::Quaternionf quat(record.pose.block<3, 3>(0, 0));
  quat.normalize();

  stream << ',' << record.pose(0, 3) << ',' << record.pose(1, 3) << ',' << record.pose(2, 3) << ','
         << quat.x() << ',' << quat.y() << ',' << quat.z() << ',' << quat.w();

  for (int row = 0; row < 4; ++row)
  {
    for (int col = 0; col < 4; ++col)
    {
      stream << ',' << record.pose(row, col);
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

// ROS 2 node that runs FoundationPose on a prerecorded RGBD sequence.
// Unlike a live tracking node, this version reads all images from disk,
// performs initial registration once, and then tracks the object frame by frame.
class FoundationPoseOfflineRgbdNode : public rclcpp::Node
{
public:
  FoundationPoseOfflineRgbdNode()
      : Node("foundationpose_offline_rgbd_node",
             rclcpp::NodeOptions().allow_undeclared_parameters(true))
  {
    // The constructor follows a clear startup pipeline:
    // 1) declare all configurable parameters,
    // 2) load them and validate paths,
    // 3) initialize the FoundationPose model,
    // 4) prepare output publishers/writers,
    // 5) trigger the batch processing loop.
    DeclareParameters();
    LoadParameters();
    BuildModel();
    SetupOutputs();
    StartProcessing();
  }

private:
  // Declares every parameter used by the node.
  // This ensures the YAML config file can provide values for both the model backend
  // and the offline dataset-processing behavior.
  void DeclareParameters()
  {
    this->declare_parameter<std::string>("refiner_engine_path", "");
    this->declare_parameter<std::string>("scorer_engine_path", "");
    this->declare_parameter<std::string>("mesh_path", "");
    this->declare_parameter<std::string>("object_name", "target_object");
    this->declare_parameter<std::string>(
        "sequence_dir", "/home/hc/weizi/fp/src/foundationpose_cpp/test_data/mustard0");
    this->declare_parameter<std::string>("rgb_subdir", "rgb");
    this->declare_parameter<std::string>("depth_subdir", "depth");
    this->declare_parameter<std::string>("masks_subdir", "masks");
    this->declare_parameter<std::string>("initial_frame_id", "");
    this->declare_parameter<std::string>("initial_mask_path", "");
    this->declare_parameter<std::string>("camera_info_path", "");
    this->declare_parameter<std::string>("pose_topic", "/foundationpose/offline_pose");
    this->declare_parameter<std::string>("pose_frame_id", "camera_color_optical_frame");
    this->declare_parameter<std::string>("visualization_topic", "/foundationpose/offline_visualization");
    this->declare_parameter<std::string>(
        "output_pose_path", "/home/hc/weizi/fp/src/foundationpose_cpp/test_data/mustard0/offline_result/poses.csv");
    this->declare_parameter<std::string>(
        "visualization_output_dir",
        "/home/hc/weizi/fp/src/foundationpose_cpp/test_data/mustard0/offline_result/visualizations");
    this->declare_parameter<std::string>(
        "video_output_path",
        "/home/hc/weizi/fp/src/foundationpose_cpp/test_data/mustard0/offline_result/track.mp4");

    this->declare_parameter<double>("fx", 0.0);
    this->declare_parameter<double>("fy", 0.0);
    this->declare_parameter<double>("cx", 0.0);
    this->declare_parameter<double>("cy", 0.0);
    this->declare_parameter<double>("depth_scale", 0.001);
    this->declare_parameter<double>("video_fps", 30.0);

    this->declare_parameter<int>("max_input_image_height", 1080);
    this->declare_parameter<int>("max_input_image_width", 1920);
    this->declare_parameter<int>("register_refine_iterations", 5);
    this->declare_parameter<int>("track_refine_iterations", 2);
    this->declare_parameter<int>("visualization_wait_key_ms", 1);

    this->declare_parameter<bool>("resize_mask_to_input", false);
    this->declare_parameter<bool>("publish_pose", true);
    this->declare_parameter<bool>("publish_visualization", true);
    this->declare_parameter<bool>("save_visualizations", true);
    this->declare_parameter<bool>("show_visualization_window", false);
    this->declare_parameter<bool>("shutdown_when_done", true);
  }

  // Loads parameter values from ROS 2 and performs sanity checks.
  // This step also discovers the available frame pairs and determines the
  // initial registration frame before any heavy inference begins.
  void LoadParameters()
  {
    refiner_engine_path_ = this->get_parameter("refiner_engine_path").as_string();
    scorer_engine_path_  = this->get_parameter("scorer_engine_path").as_string();
    mesh_path_           = this->get_parameter("mesh_path").as_string();
    object_name_         = this->get_parameter("object_name").as_string();
    sequence_dir_        = this->get_parameter("sequence_dir").as_string();
    rgb_subdir_          = this->get_parameter("rgb_subdir").as_string();
    depth_subdir_        = this->get_parameter("depth_subdir").as_string();
    masks_subdir_        = this->get_parameter("masks_subdir").as_string();
    initial_frame_id_    = this->get_parameter("initial_frame_id").as_string();
    initial_mask_path_   = this->get_parameter("initial_mask_path").as_string();
    camera_info_path_    = this->get_parameter("camera_info_path").as_string();
    pose_topic_          = this->get_parameter("pose_topic").as_string();
    pose_frame_id_       = this->get_parameter("pose_frame_id").as_string();
    visualization_topic_ = this->get_parameter("visualization_topic").as_string();
    output_pose_path_    = this->get_parameter("output_pose_path").as_string();
    visualization_output_dir_ =
        this->get_parameter("visualization_output_dir").as_string();
    video_output_path_          = this->get_parameter("video_output_path").as_string();
    depth_scale_                = this->get_parameter("depth_scale").as_double();
    video_fps_                  = this->get_parameter("video_fps").as_double();
    max_input_image_height_     = this->get_parameter("max_input_image_height").as_int();
    max_input_image_width_      = this->get_parameter("max_input_image_width").as_int();
    register_refine_iters_      = static_cast<size_t>(this->get_parameter("register_refine_iterations").as_int());
    track_refine_iters_         = static_cast<size_t>(this->get_parameter("track_refine_iterations").as_int());
    visualization_wait_key_ms_  = this->get_parameter("visualization_wait_key_ms").as_int();
    resize_mask_to_input_       = this->get_parameter("resize_mask_to_input").as_bool();
    publish_pose_               = this->get_parameter("publish_pose").as_bool();
    publish_visualization_      = this->get_parameter("publish_visualization").as_bool();
    save_visualizations_        = this->get_parameter("save_visualizations").as_bool();
    show_visualization_window_  = this->get_parameter("show_visualization_window").as_bool();
    shutdown_when_done_         = this->get_parameter("shutdown_when_done").as_bool();

    ValidateRequiredFile(refiner_engine_path_, "refiner_engine_path");
    ValidateRequiredFile(scorer_engine_path_, "scorer_engine_path");
    ValidateRequiredFile(mesh_path_, "mesh_path");

    frames_ = CollectFramePaths(sequence_dir_, rgb_subdir_, depth_subdir_);
    if (initial_frame_id_.empty())
    {
      initial_frame_id_ = frames_.front().id;
    }

    initial_frame_index_ = FindFrameIndex(initial_frame_id_);

    if (initial_mask_path_.empty())
    {
      initial_mask_path_ =
          (fs::path(sequence_dir_) / masks_subdir_ / (initial_frame_id_ + ".png")).string();
    }
    ValidateRequiredFile(initial_mask_path_, "initial_mask_path");
    initial_mask_ = LoadMaskImage(initial_mask_path_);

    if (!camera_info_path_.empty())
    {
      intrinsic_ = LoadIntrinsicMatrix(camera_info_path_);
    }
    else
    {
      const double fx = this->get_parameter("fx").as_double();
      const double fy = this->get_parameter("fy").as_double();
      const double cx = this->get_parameter("cx").as_double();
      const double cy = this->get_parameter("cy").as_double();
      intrinsic_       = BuildIntrinsicMatrix(fx, fy, cx, cy);
    }

    if (video_fps_ <= 0.0)
    {
      throw std::invalid_argument("Parameter `video_fps` must be positive.");
    }
  }

  // Validates that a required path exists and is not empty.
  // This helper keeps the parameter loading logic concise and gives clear errors
  // when the user forgets to provide paths for the engine or mesh files.
  void ValidateRequiredFile(const std::string &file_path, const std::string &param_name) const
  {
    if (file_path.empty())
    {
      throw std::invalid_argument("Required parameter is empty: " + param_name);
    }

    if (!fs::exists(file_path))
    {
      throw std::invalid_argument("Path from parameter '" + param_name + "' does not exist: " +
                                  file_path);
    }
  }

  // Searches the sequence for the specified frame identifier.
  // The frame id is used both as the starting point for registration and as the
  // naming convention for pose outputs and visualization files.
  size_t FindFrameIndex(const std::string &frame_id) const
  {
    const auto iter = std::find_if(frames_.begin(), frames_.end(), [&frame_id](const FramePaths &frame) {
      return frame.id == frame_id;
    });
    if (iter == frames_.end())
    {
      throw std::invalid_argument("Initial frame id was not found in the RGB/Depth sequence: " + frame_id);
    }
    return static_cast<size_t>(std::distance(frames_.begin(), iter));
  }

  // Constructs the TensorRT inference backend and the FoundationPose model.
  // The refiner and scorer engines are loaded here, and the object mesh is attached
  // so that both registration and tracking can run using the same intrinsic parameters.
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

  // Prepares all outputs that may be used by the offline pipeline.
  // This includes ROS publishers, CSV pose export, visualization directories,
  // and optional video writing.
  void SetupOutputs()
  {
    if (publish_pose_)
    {
      pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, 10);
    }

    if (publish_visualization_)
    {
      visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visualization_topic_, 10);
    }

    if (!output_pose_path_.empty())
    {
      const fs::path output_path(output_pose_path_);
      if (!output_path.parent_path().empty())
      {
        fs::create_directories(output_path.parent_path());
      }
      pose_stream_.open(output_pose_path_, std::ios::out | std::ios::trunc);
      if (!pose_stream_.is_open())
      {
        throw std::runtime_error("Failed to open pose output file: " + output_pose_path_);
      }

      pose_stream_
          << "frame_id,mode,success,tx,ty,tz,qx,qy,qz,qw,"
             "m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33\n";
    }

    if (save_visualizations_ && !visualization_output_dir_.empty())
    {
      fs::create_directories(visualization_output_dir_);
    }

    if (!video_output_path_.empty())
    {
      const fs::path video_path(video_output_path_);
      if (!video_path.parent_path().empty())
      {
        fs::create_directories(video_path.parent_path());
      }
    }
  }

  // Starts the processing workflow by scheduling a single execution.
  // The timer is used only to defer the heavy work until the node is fully initialized.
  void StartProcessing()
  {
    processing_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1), std::bind(&FoundationPoseOfflineRgbdNode::ProcessSequence, this));

    RCLCPP_INFO(this->get_logger(),
                "Offline RGBD node is ready. %zu paired frames found in %s, starting from frame %s.",
                frames_.size(),
                sequence_dir_.c_str(),
                initial_frame_id_.c_str());
  }

  // Executes the complete offline pipeline once.
  // The sequence is processed in this order:
  // - perform initial object registration,
  // - track the object for all remaining frames,
  // - close writers and finalize logs.
  void ProcessSequence()
  {
    processing_timer_->cancel();
    const auto start_time = std::chrono::steady_clock::now();

    try
    {
      ProcessInitialFrame();

      size_t tracked_success_count = 0;
      for (size_t index = initial_frame_index_ + 1; index < frames_.size(); ++index)
      {
        if (ProcessTrackingFrame(index))
        {
          ++tracked_success_count;
        }
      }

      if (video_writer_.isOpened())
      {
        video_writer_.release();
      }

      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
      RCLCPP_INFO(this->get_logger(),
                  "Offline processing completed. Register: success, Track: %zu/%zu success, elapsed: %.3f s",
                  tracked_success_count,
                  frames_.size() - initial_frame_index_ - 1,
                  static_cast<double>(elapsed.count()) / 1000.0);
      if (!output_pose_path_.empty())
      {
        RCLCPP_INFO(this->get_logger(), "Pose results saved to: %s", output_pose_path_.c_str());
      }
      if (save_visualizations_ && !visualization_output_dir_.empty())
      {
        RCLCPP_INFO(this->get_logger(), "Visualization frames saved to: %s", visualization_output_dir_.c_str());
      }
      if (!video_output_path_.empty())
      {
        RCLCPP_INFO(this->get_logger(), "Visualization video saved to: %s", video_output_path_.c_str());
      }
    }
    catch (const std::exception &e)
    {
      if (video_writer_.isOpened())
      {
        video_writer_.release();
      }
      RCLCPP_ERROR(this->get_logger(), "Offline RGBD processing failed: %s", e.what());
    }

    if (pose_stream_.is_open())
    {
      pose_stream_.flush();
      pose_stream_.close();
    }

    if (shutdown_when_done_)
    {
      rclcpp::shutdown();
    }
  }

  // Registers the object on the first selected frame.
  // If the initial mask does not match the RGB image size and the option is enabled,
  // the mask is resized before calling FoundationPose::Register.
  void ProcessInitialFrame()
  {
    const FramePaths &frame = frames_.at(initial_frame_index_);
    cv::Mat           rgb   = LoadRgbImage(frame.rgb_path);
    cv::Mat           depth = LoadDepthImage(frame.depth_path, depth_scale_);

    if (rgb.size() != depth.size())
    {
      throw std::runtime_error("Initial RGB and depth image sizes do not match.");
    }

    cv::Mat registration_mask = initial_mask_;
    if (registration_mask.size() != rgb.size())
    {
      if (!resize_mask_to_input_)
      {
        throw std::runtime_error("Initial mask size does not match the selected RGB frame. Enable "
                                 "`resize_mask_to_input` to allow automatic resizing.");
      }

      cv::resize(registration_mask,
                 registration_mask,
                 rgb.size(),
                 0.0,
                 0.0,
                 cv::INTER_NEAREST);
    }

    Eigen::Matrix4f pose;
    const bool ok = foundation_pose_->Register(
        rgb, depth, registration_mask, object_name_, pose, register_refine_iters_);

    PoseRecord record;
    record.mode    = "register";
    record.success = ok;
    record.pose    = ok ? pose : Eigen::Matrix4f::Identity();
    SavePoseRecord(frame.id, record);

    if (!ok)
    {
      throw std::runtime_error("Initial FoundationPose registration failed on frame " + frame.id);
    }

    last_pose_ = pose;
    PublishPose(frame.id, pose);
    SaveVisualization(frame.id, rgb, pose);

    RCLCPP_INFO(this->get_logger(), "Initial registration succeeded on frame %s.", frame.id.c_str());
  }

  // Tracks the object on one frame using the previous valid pose as initialization.
  // A failure here does not stop the whole sequence; the node simply keeps the last
  // known pose and records the failure in the CSV output.
  bool ProcessTrackingFrame(size_t frame_index)
  {
    const FramePaths &frame = frames_.at(frame_index);
    cv::Mat           rgb   = LoadRgbImage(frame.rgb_path);
    cv::Mat           depth = LoadDepthImage(frame.depth_path, depth_scale_);

    if (rgb.size() != depth.size())
    {
      throw std::runtime_error("RGB and depth image sizes do not match for frame: " + frame.id);
    }

    Eigen::Matrix4f tracked_pose;
    const bool ok =
        foundation_pose_->Track(rgb, depth, last_pose_, object_name_, tracked_pose, track_refine_iters_);

    PoseRecord record;
    record.mode    = "track";
    record.success = ok;
    record.pose    = ok ? tracked_pose : last_pose_;
    SavePoseRecord(frame.id, record);

    if (!ok)
    {
      RCLCPP_WARN(this->get_logger(), "Tracking failed on frame %s, keeping the last valid pose.", frame.id.c_str());
      return false;
    }

    last_pose_ = tracked_pose;
    PublishPose(frame.id, last_pose_);
    SaveVisualization(frame.id, rgb, last_pose_);
    return true;
  }

  // Writes one result line into the pose CSV file.
  // The line contains the frame id, success flag, mode, pose, and full transformation matrix.
  void SavePoseRecord(const std::string &frame_id, const PoseRecord &record)
  {
    if (pose_stream_.is_open())
    {
      pose_stream_ << PoseToCsvRow(frame_id, record) << '\n';
    }
  }

  // Builds a ROS header for pose and visualization messages.
  // If the frame id is numeric, it is used as the timestamp; otherwise the current ROS time is used.
  std_msgs::msg::Header BuildHeader(const std::string &frame_id) const
  {
    std_msgs::msg::Header header;
    header.frame_id = pose_frame_id_;

    try
    {
      header.stamp = rclcpp::Time(std::stoll(frame_id));
    }
    catch (const std::exception &)
    {
      header.stamp = this->now();
    }

    return header;
  }

  // Publishes the estimated object pose as a ROS message.
  // The pose is converted from the Eigen matrix representation into a PoseStamped format.
  void PublishPose(const std::string &frame_id, const Eigen::Matrix4f &pose)
  {
    if (!publish_pose_ || pose_pub_ == nullptr)
    {
      return;
    }

    pose_pub_->publish(PoseMatrixToPoseStamped(pose, BuildHeader(frame_id)));
  }

  // Creates a visualization image for one frame and optionally publishes or saves it.
  // The object mesh is projected into the image plane so the user can inspect the estimated pose.
  void SaveVisualization(const std::string &frame_id, const cv::Mat &rgb, const Eigen::Matrix4f &pose)
  {
    if (!publish_visualization_ && !save_visualizations_ && !show_visualization_window_ &&
        video_output_path_.empty())
    {
      return;
    }

    cv::Mat visualization_bgr;
    cv::cvtColor(rgb, visualization_bgr, cv::COLOR_RGB2BGR);

    const auto draw_pose = detection_6d::ConvertPoseMesh2BBox(pose, mesh_loader_);
    Draw3DBoundingBox(intrinsic_, draw_pose, mesh_loader_->GetObjectDimension(), visualization_bgr);

    if (publish_visualization_ && visualization_pub_ != nullptr)
    {
      auto image_msg = cv_bridge::CvImage(
                           BuildHeader(frame_id), sensor_msgs::image_encodings::BGR8, visualization_bgr)
                           .toImageMsg();
      visualization_pub_->publish(*image_msg);
    }

    if (save_visualizations_ && !visualization_output_dir_.empty())
    {
      const fs::path output_path = fs::path(visualization_output_dir_) / (frame_id + ".png");
      if (!cv::imwrite(output_path.string(), visualization_bgr))
      {
        throw std::runtime_error("Failed to save visualization image: " + output_path.string());
      }
    }

    if (!video_output_path_.empty())
    {
      if (!video_writer_.isOpened())
      {
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        video_writer_.open(
            video_output_path_, fourcc, video_fps_, cv::Size(visualization_bgr.cols, visualization_bgr.rows));
        if (!video_writer_.isOpened())
        {
          throw std::runtime_error("Failed to open visualization video writer: " + video_output_path_);
        }
      }

      video_writer_.write(visualization_bgr);
    }

    if (show_visualization_window_)
    {
      cv::imshow("foundationpose_offline_visualization", visualization_bgr);
      cv::waitKey(visualization_wait_key_ms_);
    }
  }

private:
  std::string refiner_engine_path_;
  std::string scorer_engine_path_;
  std::string mesh_path_;
  std::string object_name_;
  std::string sequence_dir_;
  std::string rgb_subdir_;
  std::string depth_subdir_;
  std::string masks_subdir_;
  std::string initial_frame_id_;
  std::string initial_mask_path_;
  std::string camera_info_path_;
  std::string pose_topic_;
  std::string pose_frame_id_;
  std::string visualization_topic_;
  std::string output_pose_path_;
  std::string visualization_output_dir_;
  std::string video_output_path_;

  double depth_scale_{0.001};
  double video_fps_{30.0};
  int    max_input_image_height_{1080};
  int    max_input_image_width_{1920};
  int    visualization_wait_key_ms_{1};

  size_t register_refine_iters_{5};
  size_t track_refine_iters_{2};
  size_t initial_frame_index_{0};

  bool resize_mask_to_input_{false};
  bool publish_pose_{true};
  bool publish_visualization_{true};
  bool save_visualizations_{true};
  bool show_visualization_window_{false};
  bool shutdown_when_done_{true};

  Eigen::Matrix3f intrinsic_{Eigen::Matrix3f::Identity()};
  Eigen::Matrix4f last_pose_{Eigen::Matrix4f::Identity()};
  cv::Mat         initial_mask_;
  std::vector<FramePaths> frames_;
  std::ofstream           pose_stream_;
  cv::VideoWriter         video_writer_;

  std::shared_ptr<detection_6d::BaseMeshLoader>         mesh_loader_;
  std::shared_ptr<detection_6d::Base6DofDetectionModel> foundation_pose_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         visualization_pub_;
  rclcpp::TimerBase::SharedPtr                                  processing_timer_;
};

// Program entry point.
// The node is initialized here and kept alive until the offline process completes.
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    auto node = std::make_shared<FoundationPoseOfflineRgbdNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(
        rclcpp::get_logger("foundationpose_offline_rgbd_node"), "Node startup failed: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
