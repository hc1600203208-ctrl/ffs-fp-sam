// FoundationPose Stereo Tracker Node
// 整合立体视觉深度估计(FoundationStereo)和6DOF姿态跟踪(FoundationPose)的ROS2节点
// 完整流程: 订阅左右相机图像 -> 立体深度估计 -> 首帧配准 -> 实时跟踪 -> 发布6D姿态

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "adaptive_se3_pose_filter.h"
#include "depth_confidence.hpp"
#include "detection_6d_foundationpose/foundationpose.hpp"
#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "estimator/fast_foundation_stereo_estimator.h"
#include "stereo_calibration_utils.hpp"
#include "trt_core/trt_core.h"

namespace
{

// 姿态优化网络的批处理大小
constexpr int kPoseBatchSize = 252;
// 姿态优化的裁剪图像尺寸
constexpr int kCropHeight    = 160;
constexpr int kCropWidth     = 160;

// 预处理后的图像信息结构体，包含缩放、填充等信息
struct PreparedImage
{
  cv::Mat image;              // 处理后的图像
  double  scale          = 1.0;      // 相对于目标大小的缩放因子
  int     pad_left       = 0;        // 左侧填充像素
  int     pad_top        = 0;        // 顶部填充像素
  int     resized_width  = 0;        // 实际缩放后宽度
  int     resized_height = 0;        // 实际缩放后高度
};

// 相机内参结构体(焦距和主点)
struct CameraIntrinsics
{
  double fx = 0.0;  // x方向焦距
  double fy = 0.0;  // y方向焦距
  double cx = 0.0;  // x方向主点
  double cy = 0.0;  // y方向主点
};

// 检查两个ROS消息的时间戳是否在指定范围内同步
// 用于验证立体图像对是否足够同步
bool CheckTimestamp(const builtin_interfaces::msg::Time &time1,
                    const builtin_interfaces::msg::Time &time2,
                    double                              max_delta_sec,
                    double                             *cur_timestamp)
{
  const double timestamp1 =
      static_cast<double>(time1.sec) + static_cast<double>(time1.nanosec) / 1e9;
  const double timestamp2 =
      static_cast<double>(time2.sec) + static_cast<double>(time2.nanosec) / 1e9;

  if (cur_timestamp != nullptr)
  {
    *cur_timestamp = timestamp1;
  }

  return std::fabs(timestamp1 - timestamp2) <= max_delta_sec;
}

// 将ROS时间戳(秒+纳秒)转换为浮点秒数
double StampToSeconds(const builtin_interfaces::msg::Time &stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) / 1e9;
}

// 根据相机内参(焦距和主点)构建3x3内参矩阵
// 矩阵形式: [fx  0 cx]
//           [ 0 fy cy]
//           [ 0  0  1]
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

// 将ROS Image消息转换为OpenCV Mat格式，保持原始编码格式
cv::Mat RosImageToCvMat(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  return cv_bridge::toCvCopy(msg, msg->encoding)->image.clone();
}

// 将ROS图像消息转换为RGB格式
// 支持的输入格式: RGB8, BGR8, RGBA8, BGRA8, MONO8
// 输出格式: RGB8(3通道)
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

// 将立体输入图像转换为标准格式
// 支持1/3/4通道，4通道时转换为3通道(BGRA->BGR)
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

// 将掩码图像转换为标准的8位灰度二值图像
// 处理步骤: 彩色转灰度 -> 类型统一为8UC1 -> 二值化
cv::Mat ConvertMaskImage(cv::Mat mask)
{
  if (mask.empty())
  {
    return mask;
  }

  // 彩色转灰度
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

  // 数据类型转换
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

  // 二值化：确保只有0或255两个值
  cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
  return mask;
}

// 为立体模型推理准备输入图像
// 缩放图像以适应目标尺寸，同时保持宽高比，不足部分用边界复制填充
// 记录缩放和填充信息以便后续反解回原分辨率
PreparedImage PrepareStereoInput(const cv::Mat &input, int target_width, int target_height)
{
  PreparedImage result;
  if (input.empty())
  {
    return result;
  }

  // 确保输入为3通道彩色图
  cv::Mat color;
  if (input.channels() == 1)
  {
    cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
  }
  else if (input.channels() == 3)
  {
    color = input.clone();
  }
  else
  {
    throw std::runtime_error("Unsupported image channel count.");
  }

  // 计算缩放因子(保持宽高比，按较小的方向缩放)
  result.scale = std::min(static_cast<double>(target_width) / static_cast<double>(color.cols),
                          static_cast<double>(target_height) / static_cast<double>(color.rows));

  result.resized_width  = std::max(1, static_cast<int>(std::round(color.cols * result.scale)));
  result.resized_height = std::max(1, static_cast<int>(std::round(color.rows * result.scale)));

  // 缩放图像
  cv::Mat resized;
  cv::resize(color,
             resized,
             cv::Size(result.resized_width, result.resized_height),
             0,
             0,
             cv::INTER_LINEAR);

  // 计算填充大小(居中填充)
  const int pad_width  = target_width - result.resized_width;
  const int pad_height = target_height - result.resized_height;
  result.pad_left      = pad_width / 2;
  result.pad_top       = pad_height / 2;

  // 在四周填充，使最终图像尺寸为 target_width x target_height
  cv::copyMakeBorder(resized,
                     result.image,
                     result.pad_top,
                     pad_height - result.pad_top,
                     result.pad_left,
                     pad_width - result.pad_left,
                     cv::BORDER_REPLICATE);
  return result;
}

// 从立体校正后的投影矩阵P2中提取基线距离
// 基线 = |P2[0,3]| / fx，其中P2[0,3]是右相机投影矩阵的平移项
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

// 根据准备好的输入图像(包含缩放和填充)计算调整后的相机内参
// 需要考虑缩放因子和填充对主点位置的影响
CameraIntrinsics ComputePreparedIntrinsics(const StereoRectificationMaps &rectification_maps,
                                           const PreparedImage           &prepared)
{
  CameraIntrinsics intrinsics;
  if (rectification_maps.p1.empty())
  {
    return intrinsics;
  }

  // 焦距乘以缩放因子
  intrinsics.fx = rectification_maps.p1.at<double>(0, 0) * prepared.scale;
  intrinsics.fy = rectification_maps.p1.at<double>(1, 1) * prepared.scale;
  // 主点同时受缩放和填充的影响
  intrinsics.cx = rectification_maps.p1.at<double>(0, 2) * prepared.scale +
                  static_cast<double>(prepared.pad_left);
  intrinsics.cy = rectification_maps.p1.at<double>(1, 2) * prepared.scale +
                  static_cast<double>(prepared.pad_top);
  return intrinsics;
}

// 将视差图转换为深度值(米)
// 公式: depth = fx * baseline / disparity
// 超出有效范围的值设为0(无效)
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
      // 跳过无效的视差值
      if (!std::isfinite(disparity_value) || disparity_value <= 0.0F)
      {
        depth_row[x] = 0.0F;
        continue;
      }

      const float depth_value = numerator / disparity_value;
      // 深度值必须在有效范围内
      if (!std::isfinite(depth_value) || depth_value < min_depth_meters ||
          depth_value > max_depth_meters)
      {
        depth_row[x] = 0.0F;
        continue;
      }

      depth_row[x] = depth_value;
    }
  }

  return depth;
}

// 将经过缩放和填充的图像恢复到原始立体校正分辨率
// 步骤: 去掉填充 -> 缩放回原尺寸
cv::Mat RestoreToRectifiedResolution(const cv::Mat       &image,
                                     const PreparedImage &prepared,
                                     const cv::Size      &rectified_size,
                                     int                  interpolation)
{
  if (image.empty() || rectified_size.width <= 0 || rectified_size.height <= 0)
  {
    return {};
  }

  // 提取有效区域(去掉填充的部分)
  const cv::Rect valid_roi(prepared.pad_left,
                           prepared.pad_top,
                           std::min(prepared.resized_width, image.cols - prepared.pad_left),
                           std::min(prepared.resized_height, image.rows - prepared.pad_top));
  if (valid_roi.width <= 0 || valid_roi.height <= 0)
  {
    return {};
  }

  // 缩放回原始分辨率
  cv::Mat restored;
  cv::resize(image(valid_roi), restored, rectified_size, 0, 0, interpolation);
  return restored;
}

// 将4x4变换矩阵(SE(3))转换为ROS PoseStamped消息
// 平移: 矩阵最后一列的前3行
// 旋转: 矩阵左上角3x3矩阵 -> 四元数
geometry_msgs::msg::PoseStamped PoseMatrixToPoseStamped(const Eigen::Matrix4f     &pose,
                                                        const std_msgs::msg::Header &header)
{
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = header;

  // 平移
  pose_msg.pose.position.x = pose(0, 3);
  pose_msg.pose.position.y = pose(1, 3);
  pose_msg.pose.position.z = pose(2, 3);

  // 旋转矩阵转四元数
  Eigen::Matrix3f rotation = pose.block<3, 3>(0, 0);
  Eigen::Quaternionf quat(rotation);
  quat.normalize();

  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  pose_msg.pose.orientation.w = quat.w();
  return pose_msg;
}

// 在图像上绘制3D边界框和坐标轴
// 步骤:
// 1. 根据物体尺寸生成8个顶点
// 2. 用姿态矩阵变换顶点到相机坐标系
// 3. 用相机内参投影到图像平面
// 4. 绘制12条边(立方体)和3条坐标轴(RGB分别表示XYZ)
void Draw3DBoundingBox(const Eigen::Matrix3f &intrinsic,
                       const Eigen::Matrix4f &pose,
                       const Eigen::Vector3f &dimension,
                       cv::Mat               &image)
{
  // 生成立方体的8个顶点(物体坐标系，以质心为原点)
  const float half_l = dimension(0) / 2.0F;
  const float half_w = dimension(1) / 2.0F;
  const float half_h = dimension(2) / 2.0F;

  const Eigen::Vector3f points[8] = {
      {-half_l, -half_w, half_h},  {half_l, -half_w, half_h},
      {half_l, half_w, half_h},    {-half_l, half_w, half_h},
      {-half_l, -half_w, -half_h}, {half_l, -half_w, -half_h},
      {half_l, half_w, -half_h},   {-half_l, half_w, -half_h}};

  // 将顶点变换到相机坐标系
  Eigen::Vector4f transformed_points[8];
  for (int i = 0; i < 8; ++i)
  {
    transformed_points[i] = pose * Eigen::Vector4f(points[i](0), points[i](1), points[i](2), 1.0F);
  }

  // 投影到图像平面
  std::vector<cv::Point2f> image_points;
  image_points.reserve(8);
  for (const auto &point : transformed_points)
  {
    const float z = point(2);
    // 只投影相机前方的点
    if (z <= 1e-6F)
    {
      return;
    }

    image_points.emplace_back(intrinsic(0, 0) * (point(0) / z) + intrinsic(0, 2),
                              intrinsic(1, 1) * (point(1) / z) + intrinsic(1, 2));
  }

  // 绘制12条边(立方体)，绿色
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                                  {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (const auto &edge : edges)
  {
    cv::line(image, image_points[edge.first], image_points[edge.second], cv::Scalar(0, 255, 0), 2);
  }

  // 计算物体中心并投影
  const Eigen::Vector4f center_world = pose * Eigen::Vector4f(0.0F, 0.0F, 0.0F, 1.0F);
  if (center_world(2) <= 1e-6F)
  {
    return;
  }

  // 绘制坐标轴(长度为物体平均尺寸的1/6)
  const float axis_length = (dimension(0) + dimension(1) + dimension(2)) / 6.0F;
  const std::vector<Eigen::Vector4f> axis_end_points = {
      pose * Eigen::Vector4f(axis_length, 0.0F, 0.0F, 1.0F),  // X轴
      pose * Eigen::Vector4f(0.0F, axis_length, 0.0F, 1.0F),  // Y轴
      pose * Eigen::Vector4f(0.0F, 0.0F, axis_length, 1.0F)}; // Z轴
  // 坐标轴颜色: X红, Y绿, Z蓝(注意OpenCV使用BGR)
  const std::vector<cv::Scalar> axis_colors = {
      cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};

  const float center_u =
      intrinsic(0, 0) * (center_world(0) / center_world(2)) + intrinsic(0, 2);
  const float center_v =
      intrinsic(1, 1) * (center_world(1) / center_world(2)) + intrinsic(1, 2);
  const cv::Point center_pt(center_u, center_v);

  // 绘制从中心指向各轴方向的线
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

// ROS2节点：集成立体视觉深度估计和6D姿态跟踪
// 完整工作流程:
// 1. 订阅左右立体图像
// 2. 立体深度估计(FoundationStereo)
// 3. 首帧配准(初始化)
// 4. 实时跟踪和SE(3)滤波
// 5. 发布目标6D姿态
class FoundationPoseStereoTrackerNode : public rclcpp::Node
{
public:
  // 构造函数: 初始化节点、加载参数、构建推理引擎、配置ROS接口
  FoundationPoseStereoTrackerNode()
      : Node("foundationpose_stereo_tracker_node"),
        image_qos_profile_(rclcpp::QoS(10))
  {
    DeclareParameters();        // 声明所有ROS参数
    LoadParameters();           // 从参数服务器加载参数值
    BuildStereoEstimator();     // 初始化立体深度估计模型
    BuildPoseModel();           // 初始化姿态优化和评分模型
    SetupRosInterfaces();       // 配置订阅器、发布器和计时器

    RCLCPP_INFO(this->get_logger(),
                "Integrated FoundationStereo + FoundationPose node is ready. Subscribing to %s and "
                "%s; publishing poses to %s.",
                left_image_topic_.c_str(),
                right_image_topic_.c_str(),
                pose_topic_.c_str());
  }

private:
  using ImageMsg         = sensor_msgs::msg::Image;
  using StereoSyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
  using StereoSyncer     = message_filters::Synchronizer<StereoSyncPolicy>;
  using SubscriberT      = message_filters::Subscriber<ImageMsg>;

  // 声明所有ROS参数及其默认值
  // 参数分类: 图像订阅、立体模型配置、姿态模型配置、深度置信度参数、滤波器参数
  void DeclareParameters()
  {
    // 图像订阅参数
    this->declare_parameter<int>("image_reliability",
                                 static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));
    this->declare_parameter<std::string>("left_image_topic", "/left/image_raw");
    this->declare_parameter<std::string>("right_image_topic", "/right/image_raw");
    this->declare_parameter<int>("sync_queue_size", 10);
    this->declare_parameter<double>("max_sync_interval_sec", 0.07);

    this->declare_parameter<std::vector<std::string>>("stereo_engine_file_path", {});
    this->declare_parameter<std::string>("stereo_model_type", "FAST_FOUNDATION_STEREO");
    this->declare_parameter<int>("model_input_height", 448);
    this->declare_parameter<int>("model_input_width", 640);
    this->declare_parameter<double>("min_depth_meters", 0.1);
    this->declare_parameter<double>("max_depth_meters", 100.0);
    this->declare_parameter<bool>("use_filtered_depth_for_pose", false);
    this->declare_parameter<std::string>("caminfo_path", "caminfo.txt");

    this->declare_parameter<int>("patch_radius", confidence_params_.patch_radius);
    this->declare_parameter<int>("texture_ksize", confidence_params_.texture_ksize);
    this->declare_parameter<double>("sigma_photo", confidence_params_.sigma_photo);
    this->declare_parameter<double>("sigma_e", confidence_params_.sigma_e);
    this->declare_parameter<double>("sigma_t", confidence_params_.sigma_t);
    this->declare_parameter<double>("alpha_edge", confidence_params_.alpha_edge);
    this->declare_parameter<double>("w_photo", confidence_params_.w_photo);
    this->declare_parameter<double>("w_tex", confidence_params_.w_tex);
    this->declare_parameter<double>("w_grad", confidence_params_.w_grad);
    this->declare_parameter<double>("w_tmp", confidence_params_.w_tmp);
    this->declare_parameter<double>("conf_threshold", confidence_params_.conf_threshold);
    this->declare_parameter<double>("weight_gamma", confidence_params_.weight_gamma);

    this->declare_parameter<std::string>("refiner_engine_path", "");
    this->declare_parameter<std::string>("scorer_engine_path", "");
    this->declare_parameter<std::string>("mesh_path", "");
    this->declare_parameter<std::string>("object_name", "target_object");
    this->declare_parameter<std::string>("mask_image_path", "");
    this->declare_parameter<std::string>("mask_image_directory", "");
    this->declare_parameter<std::string>("mask_image_name", "first_mask.png");
    this->declare_parameter<std::string>("pose_topic", "/foundationpose/pose");
    this->declare_parameter<std::string>("pose_frame_id", "");
    this->declare_parameter<std::string>("visualization_topic", "/foundationpose/visualization");
    this->declare_parameter<std::string>("visualization_window_name", "foundationpose_visualization");

    this->declare_parameter<double>("fx", 0.0);
    this->declare_parameter<double>("fy", 0.0);
    this->declare_parameter<double>("cx", 0.0);
    this->declare_parameter<double>("cy", 0.0);
    this->declare_parameter<int>("max_input_image_height", 1080);
    this->declare_parameter<int>("max_input_image_width", 1920);
    this->declare_parameter<int>("register_refine_iterations", 5);
    this->declare_parameter<int>("track_refine_iterations", 2);
    this->declare_parameter<double>("pose_filter.default_dt", 1.0 / 30.0);
    this->declare_parameter<double>("pose_filter.q_pos", 1e-4);
    this->declare_parameter<double>("pose_filter.q_rot", 1e-4);
    this->declare_parameter<double>("pose_filter.q_vel", 1e-3);
    this->declare_parameter<double>("pose_filter.q_omega", 1e-3);
    this->declare_parameter<double>("pose_filter.sigma_pos_obs", 0.01);
    this->declare_parameter<double>("pose_filter.sigma_rot_obs", 0.05);
    this->declare_parameter<double>("pose_filter.sigma_pos_motion", 0.05);
    this->declare_parameter<double>("pose_filter.sigma_rot_motion", 0.35);
    this->declare_parameter<double>("pose_filter.min_quality", 0.01);
    this->declare_parameter<double>("pose_filter.gate_threshold", 16.81);
    this->declare_parameter<double>("pose_filter.reject_cov_increase", 1e-4);
    this->declare_parameter<bool>("pose_filter.enable_gating", true);
    this->declare_parameter<bool>("pose_filter.enable_adaptive_R", true);
    this->declare_parameter<int>("mask_poll_interval_ms", 200);
    this->declare_parameter<bool>("resize_mask_to_input", false);
    this->declare_parameter<bool>("publish_visualization", false);
    this->declare_parameter<bool>("show_visualization_window", false);
  }

  // 从参数服务器加载所有参数值到成员变量
  // 参数验证: 文件存在性检查、取值范围检查
  void LoadParameters()
  {
    // 图像订阅参数
    image_reliability_     = this->get_parameter("image_reliability").as_int();
    left_image_topic_      = this->get_parameter("left_image_topic").as_string();
    right_image_topic_     = this->get_parameter("right_image_topic").as_string();
    sync_queue_size_       = this->get_parameter("sync_queue_size").as_int();
    max_sync_interval_sec_ = this->get_parameter("max_sync_interval_sec").as_double();

    // 立体模型参数
    stereo_engine_file_path_ = this->get_parameter("stereo_engine_file_path").as_string_array();
    stereo_model_type_       = this->get_parameter("stereo_model_type").as_string();
    model_input_height_      = this->get_parameter("model_input_height").as_int();
    model_input_width_       = this->get_parameter("model_input_width").as_int();
    min_depth_meters_        = this->get_parameter("min_depth_meters").as_double();
    max_depth_meters_        = this->get_parameter("max_depth_meters").as_double();
    use_filtered_depth_for_pose_ = this->get_parameter("use_filtered_depth_for_pose").as_bool();
    caminfo_path_                = this->get_parameter("caminfo_path").as_string();

    // 深度置信度评估参数
    confidence_params_.patch_radius = this->get_parameter("patch_radius").as_int();
    confidence_params_.texture_ksize = this->get_parameter("texture_ksize").as_int();
    confidence_params_.sigma_photo =
        static_cast<float>(this->get_parameter("sigma_photo").as_double());
    confidence_params_.sigma_e = static_cast<float>(this->get_parameter("sigma_e").as_double());
    confidence_params_.sigma_t = static_cast<float>(this->get_parameter("sigma_t").as_double());
    confidence_params_.alpha_edge =
        static_cast<float>(this->get_parameter("alpha_edge").as_double());
    confidence_params_.w_photo = static_cast<float>(this->get_parameter("w_photo").as_double());
    confidence_params_.w_tex   = static_cast<float>(this->get_parameter("w_tex").as_double());
    confidence_params_.w_grad  = static_cast<float>(this->get_parameter("w_grad").as_double());
    confidence_params_.w_tmp   = static_cast<float>(this->get_parameter("w_tmp").as_double());
    confidence_params_.conf_threshold =
        static_cast<float>(this->get_parameter("conf_threshold").as_double());
    confidence_params_.weight_gamma =
        static_cast<float>(this->get_parameter("weight_gamma").as_double());

    // FoundationPose姿态模型参数
    refiner_engine_path_ = this->get_parameter("refiner_engine_path").as_string();
    scorer_engine_path_  = this->get_parameter("scorer_engine_path").as_string();
    mesh_path_           = this->get_parameter("mesh_path").as_string();
    object_name_         = this->get_parameter("object_name").as_string();
    pose_topic_          = this->get_parameter("pose_topic").as_string();
    pose_frame_id_       = this->get_parameter("pose_frame_id").as_string();
    visualization_topic_ = this->get_parameter("visualization_topic").as_string();
    visualization_window_name_ = this->get_parameter("visualization_window_name").as_string();
    max_input_image_height_    = this->get_parameter("max_input_image_height").as_int();
    max_input_image_width_     = this->get_parameter("max_input_image_width").as_int();
    register_refine_iters_ =
        static_cast<size_t>(this->get_parameter("register_refine_iterations").as_int());
    track_refine_iters_ =
        static_cast<size_t>(this->get_parameter("track_refine_iterations").as_int());

    // SE(3)滤波器参数
    pose_filter_default_dt_ = this->get_parameter("pose_filter.default_dt").as_double();
    pose_filter_params_.q_pos =
        static_cast<float>(this->get_parameter("pose_filter.q_pos").as_double());
    pose_filter_params_.q_rot =
        static_cast<float>(this->get_parameter("pose_filter.q_rot").as_double());
    pose_filter_params_.q_vel =
        static_cast<float>(this->get_parameter("pose_filter.q_vel").as_double());
    pose_filter_params_.q_omega =
        static_cast<float>(this->get_parameter("pose_filter.q_omega").as_double());
    pose_filter_params_.sigma_pos_obs =
        static_cast<float>(this->get_parameter("pose_filter.sigma_pos_obs").as_double());
    pose_filter_params_.sigma_rot_obs =
        static_cast<float>(this->get_parameter("pose_filter.sigma_rot_obs").as_double());
    pose_filter_params_.sigma_pos_motion =
        static_cast<float>(this->get_parameter("pose_filter.sigma_pos_motion").as_double());
    pose_filter_params_.sigma_rot_motion =
        static_cast<float>(this->get_parameter("pose_filter.sigma_rot_motion").as_double());
    pose_filter_params_.min_quality =
        static_cast<float>(this->get_parameter("pose_filter.min_quality").as_double());
    pose_filter_params_.gate_threshold =
        static_cast<float>(this->get_parameter("pose_filter.gate_threshold").as_double());
    pose_filter_params_.reject_cov_increase =
        static_cast<float>(this->get_parameter("pose_filter.reject_cov_increase").as_double());
    pose_filter_params_.enable_gating =
        this->get_parameter("pose_filter.enable_gating").as_bool();
    pose_filter_params_.enable_adaptive_R =
        this->get_parameter("pose_filter.enable_adaptive_R").as_bool();
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
  }

  // 解析掩码图像路径
  // 优先使用直接路径，若为空则组合目录+文件名
  std::string ResolveMaskImagePath() const
  {
    // 优先使用直接路径参数
    const auto direct_path = this->get_parameter("mask_image_path").as_string();
    if (!direct_path.empty())
    {
      return direct_path;
    }

    // 其次组合目录和文件名
    const auto directory = this->get_parameter("mask_image_directory").as_string();
    const auto name      = this->get_parameter("mask_image_name").as_string();
    if (directory.empty() || name.empty())
    {
      return "";
    }

    return (std::filesystem::path(directory) / name).string();
  }

  // 验证必需的文件是否存在
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

  // 构建立体视觉深度估计器
  // FastFoundationStereo需要两个TensorRT引擎(左右图处理)
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

  // 构建FoundationPose 6D姿态优化和评分模型
  // 包含两个TensorRT推理引擎:
  // 1. Refiner: 优化姿态估计(输入变换的源图/渲染图，输出平移/旋转增量)
  // 2. Scorer: 评估姿态好坏(输入同上，输出置信度分数)
  void BuildPoseModel()
  {
    // 姿态优化器(Refiner)引擎
    auto refiner_core = inference_core::CreateTrtInferCore(
        refiner_engine_path_,
        {{"transf_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}},
         {"render_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}}},
        {{"trans", {kPoseBatchSize, 3}}, {"rot", {kPoseBatchSize, 3}}},
        1);

    // 姿态评分器(Scorer)引擎
    auto scorer_core = inference_core::CreateTrtInferCore(
        scorer_engine_path_,
        {{"transf_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}},
         {"render_input", {kPoseBatchSize, kCropHeight, kCropWidth, 6}}},
        {{"scores", {kPoseBatchSize, 1}}},
        1);

    // 加载物体3D模型网格
    mesh_loader_ = detection_6d::CreateAssimpMeshLoader(object_name_, mesh_path_);
    // 创建FoundationPose模型(结合优化和评分引擎)
    foundation_pose_ =
        detection_6d::CreateFoundationPoseModel(refiner_core,
                                                scorer_core,
                                                {mesh_loader_},
                                                intrinsic_,
                                                max_input_image_height_,
                                                max_input_image_width_);
  }

  // 配置ROS订阅器、发布器和计时器
  // 1. 订阅器: 左右立体图像(近似时间同步)
  // 2. 发布器: 估计的6D姿态，可选的可视化图像
  // 3. 计时器: 定期尝试首帧配准(掩码加载可能延迟)
  void SetupRosInterfaces()
  {
    // 配置图像QoS策略
    image_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(image_reliability_));
    image_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    image_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);

    // 创建发布器
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, 10);
    if (publish_visualization_)
    {
      visualization_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visualization_topic_, 10);
    }

    // 订阅左右图像
    left_image_sub_.subscribe(this, left_image_topic_, image_qos_profile_.get_rmw_qos_profile());
    right_image_sub_.subscribe(this, right_image_topic_, image_qos_profile_.get_rmw_qos_profile());

    // 配置近似时间同步(允许小时间差的立体图像对)
    stereo_sync_ = std::make_shared<StereoSyncer>(
        StereoSyncPolicy(sync_queue_size_), left_image_sub_, right_image_sub_);
    stereo_sync_->getPolicy()->setMaxIntervalDuration(
        rclcpp::Duration::from_seconds(max_sync_interval_sec_));
    stereo_sync_->registerCallback(std::bind(&FoundationPoseStereoTrackerNode::StereoCallback,
                                             this,
                                             std::placeholders::_1,
                                             std::placeholders::_2));

    // 创建定时器: 定期尝试首帧配准(处理掩码文件加载可能延迟的情况)
    mask_poll_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(mask_poll_interval_ms_),
        std::bind(&FoundationPoseStereoTrackerNode::TryRegisterFromCachedRgbd, this));
  }

  // 确保立体校正映射已计算(若图像尺寸变化则重新计算)
  // 校正映射用于rectify立体图像对，消除相机畸变和极线偏移
  bool EnsureRectificationMaps(const cv::Size &image_size)
  {
    // 缓存检查: 若已为该尺寸计算过则直接返回
    if (rectification_ready_ && rectification_maps_.image_size == image_size)
    {
      return true;
    }

    // 从标定数据计算校正映射
    std::string rectification_error;
    if (!ComputeStereoRectificationMaps(calibration_, image_size, &rectification_maps_, &rectification_error))
    {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to compute stereo rectification maps: %s",
                   rectification_error.c_str());
      return false;
    }

    rectification_ready_ = true;
    RCLCPP_INFO(this->get_logger(),
                "Stereo rectification maps prepared for %dx%d input images.",
                image_size.width,
                image_size.height);
    return true;
  }

  // 从立体图像对生成深度图
  // 步骤:
  // 1. 图像校正(使用标定信息)
  // 2. 图像预处理(缩放到模型输入尺寸)
  // 3. FoundationStereo推理生成视差
  // 4. 视差转深度(考虑有效范围)
  // 5. 可选: 深度置信度滤波
  // 6. 恢复到原始分辨率
  bool BuildDepthFromStereo(const cv::Mat &left_stereo_input,
                            const cv::Mat &right_stereo_input,
                            cv::Mat       *depth_for_pose)
  {
    if (left_stereo_input.size() != right_stereo_input.size())
    {
      RCLCPP_ERROR(this->get_logger(), "Stereo image sizes do not match.");
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
      RCLCPP_ERROR(this->get_logger(), "Failed to prepare stereo images for inference.");
      return false;
    }

    const auto infer_start = std::chrono::high_resolution_clock::now();
    cv::Mat disparity;
    if (!stereo_estimator_->inference(prepared_left.image, prepared_right.image, disparity))
    {
      RCLCPP_ERROR(this->get_logger(), "FoundationStereo inference failed.");
      return false;
    }
    const auto infer_end = std::chrono::high_resolution_clock::now();
    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "FoundationStereo inference time: %ld ms.",
        std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count());

    const double baseline_meters = ComputeRectifiedBaselineMeters(rectification_maps_);
    const CameraIntrinsics intrinsics = ComputePreparedIntrinsics(rectification_maps_, prepared_left);
    if (baseline_meters <= 0.0 || intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "Skipping frame because rectified stereo intrinsics are invalid.");
      return false;
    }

    const cv::Mat raw_depth_model = DisparityToDepthMeters(disparity,
                                                           static_cast<float>(intrinsics.fx),
                                                           static_cast<float>(baseline_meters),
                                                           static_cast<float>(min_depth_meters_),
                                                           static_cast<float>(max_depth_meters_));

    cv::Mat depth_model_for_pose = raw_depth_model;
    if (use_filtered_depth_for_pose_)
    {
      const cv::Mat confidence_map =
          depth_confidence::compute_confidence_map(prepared_left.image,
                                                   prepared_right.image,
                                                   raw_depth_model,
                                                   static_cast<float>(intrinsics.fx),
                                                   static_cast<float>(intrinsics.fy),
                                                   static_cast<float>(intrinsics.cx),
                                                   static_cast<float>(intrinsics.cy),
                                                   static_cast<float>(baseline_meters),
                                                   previous_depth_,
                                                   confidence_params_);
      if (!confidence_map.empty())
      {
        depth_model_for_pose = depth_confidence::filter_depth_with_confidence(
            raw_depth_model, confidence_map, confidence_params_.conf_threshold);
      }
    }
    previous_depth_ = raw_depth_model.clone();

    const cv::Mat depth_rectified_full = RestoreToRectifiedResolution(depth_model_for_pose,
                                                                      prepared_left,
                                                                      left_rectified.size(),
                                                                      cv::INTER_NEAREST);
    *depth_for_pose = AlignRectifiedDepthToOriginalLeft(depth_rectified_full,
                                                       rectification_maps_,
                                                       cv::INTER_NEAREST);
    return depth_for_pose != nullptr && !depth_for_pose->empty();
  }

  // 立体图像对回调函数
  // 工作流程:
  // 1. 尝试获取互斥锁(如果推理忙碌则跳过)
  // 2. 检查时间戳同步
  // 3. 图像转换(RGB格式)
  // 4. 生成深度图
  // 5. 若无初始姿态，缓存RGBD并尝试配准；否则运行跟踪
  void StereoCallback(const ImageMsg::ConstSharedPtr &left_msg,
                      const ImageMsg::ConstSharedPtr &right_msg)
  {
    // 非阻塞式互斥锁，防止多帧堆积
    std::unique_lock<std::mutex> lock(process_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000, "Dropping frame because inference is busy.");
      return;
    }

    // 检查立体图像时间同步
    double timestamp = 0.0;
    if (!CheckTimestamp(left_msg->header.stamp,
                        right_msg->header.stamp,
                        max_sync_interval_sec_,
                        &timestamp))
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "Dropping stereo pair because timestamps differ by more than %.3f s.",
                           max_sync_interval_sec_);
      return;
    }

    try
    {
      // 图像格式转换
      const cv::Mat pose_rgb = ConvertPoseRgbImage(left_msg);
      const cv::Mat left_stereo_input = ConvertStereoInputImage(left_msg);
      const cv::Mat right_stereo_input = ConvertStereoInputImage(right_msg);

      // 从立体对生成深度图
      cv::Mat depth;
      if (!BuildDepthFromStereo(left_stereo_input, right_stereo_input, &depth))
      {
        return;
      }

      if (pose_rgb.size() != depth.size())
      {
        throw std::runtime_error("Left RGB and generated depth image sizes do not match.");
      }

      // 初始化阶段: 缓存RGBD，等待掩码文件进行首帧配准
      if (!has_pose_)
      {
        latest_rgb_        = pose_rgb.clone();
        latest_depth_      = depth.clone();
        latest_header_     = left_msg->header;
        has_cached_rgbd_   = true;
        TryRegisterFromCachedRgbdLocked();
        return;
      }

      // 跟踪阶段: 运行实时姿态跟踪
      RunTracking(left_msg->header, pose_rgb, depth);
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000, "Stereo callback failed: %s", e.what());
    }
  }

  // 定时器回调: 尝试从缓存的RGBD进行首帧配准(处理掩码延迟加载)
  // 非阻塞式检查，若推理繁忙则跳过
  void TryRegisterFromCachedRgbd()
  {
    // 若已有姿态，取消定时器(在RunInitialRegistration中)
    if (has_pose_)
    {
      return;
    }

    // 尝试获取互斥锁(不阻塞)
    std::unique_lock<std::mutex> lock(process_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
      return;
    }

    TryRegisterFromCachedRgbdLocked();
  }

  // 在持有互斥锁的情况下尝试首帧配准(内部函数)
  // 主要处理掩码文件的延迟加载
  void TryRegisterFromCachedRgbdLocked()
  {
    if (has_pose_)
    {
      return;
    }

    // 等待第一帧RGBD到达
    if (!has_cached_rgbd_)
    {
      RCLCPP_INFO_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           5000,
                           "Waiting for the first stereo RGBD frame.");
      return;
    }

    // 读取掩码文件
    cv::Mat mask = cv::imread(mask_image_path_, cv::IMREAD_UNCHANGED);
    if (mask.empty())
    {
      RCLCPP_INFO_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "Waiting for a readable first-frame mask at: %s",
                           mask_image_path_.c_str());
      return;
    }

    // 掩码格式转换(转为8bit二值图)
    mask = ConvertMaskImage(mask);

    // 检查掩码尺寸是否匹配
    if (mask.size() != latest_rgb_.size())
    {
      if (!resize_mask_to_input_)
      {
        RCLCPP_WARN_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             2000,
                             "Mask image size %dx%d does not match RGBD size %dx%d. Enable "
                             "resize_mask_to_input to resize it automatically.",
                             mask.cols,
                             mask.rows,
                             latest_rgb_.cols,
                             latest_rgb_.rows);
        return;
      }

      // 自动调整掩码尺寸
      cv::resize(mask, mask, latest_rgb_.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }

    // 尝试首帧配准，成功则取消定时器
    if (RunInitialRegistration(latest_header_, latest_rgb_, latest_depth_, mask))
    {
      if (mask_poll_timer_ != nullptr)
      {
        mask_poll_timer_->cancel();
        mask_poll_timer_.reset();
      }
    }
  }

  // 首帧配准: 从首次获得的RGBD和掩码估计初始6D姿态
  // 步骤:
  // 1. 使用FoundationPose Register进行首帧配准
  // 2. 初始化SE(3)滤波器
  // 3. 设置有姿态标志，开始跟踪阶段
  bool RunInitialRegistration(const std_msgs::msg::Header &header,
                              const cv::Mat               &rgb,
                              const cv::Mat               &depth,
                              const cv::Mat               &registration_mask)
  {
    // FoundationPose配准(初始6D姿态估计)
    Eigen::Matrix4f pose;
    const bool ok = foundation_pose_->Register(
        rgb, depth, registration_mask, object_name_, pose, register_refine_iters_);
    if (!ok)
    {
      RCLCPP_ERROR_THROTTLE(this->get_logger(),
                            *this->get_clock(),
                            2000,
                            "Initial FoundationPose registration failed. Waiting for another "
                            "stereo frame and mask retry.");
      return false;
    }

    // 初始化SE(3)滤波器(用于平滑和预测)
    const float init_dt = static_cast<float>(
        std::isfinite(pose_filter_default_dt_) && pose_filter_default_dt_ > 0.0
            ? pose_filter_default_dt_
            : 1.0 / 30.0);
    pose_filter_.initialize(pose, init_dt, pose_filter_params_);

    // 设置状态为有姿态，启动跟踪模式
    has_pose_                = true;
    last_pose_               = pose_filter_.getPose();
    has_last_pose_timestamp_ = true;
    last_pose_timestamp_sec_ = StampToSeconds(header.stamp);
    filter_frame_id_         = 0;
    PublishPose(header, last_pose_);
    PublishVisualization(header, rgb, last_pose_);

    RCLCPP_INFO(this->get_logger(),
                "Initial registration succeeded. Tracking now runs from raw stereo images without "
                "intermediate ROS depth or mask topics.");
    RCLCPP_INFO(this->get_logger(),
                "Pose filter frame=%lu quality=%.4f mahalanobis=%.4f accepted translation_residual=%.6f "
                "rotation_residual=%.6f",
                static_cast<unsigned long>(filter_frame_id_),
                pose_filter_.getLastQuality(),
                pose_filter_.getLastMahalanobisDistance(),
                pose_filter_.getLastTranslationResidualNorm(),
                pose_filter_.getLastRotationResidualNorm());
    return true;
  }

  // 实时姿态跟踪
  // 工作流程:
  // 1. SE(3)滤波器预测(基于前一帧和运动模型)
  // 2. FoundationPose从预测开始精细跟踪
  // 3. 若跟踪失败，使用滤波器预测；若成功，更新滤波器
  // 4. 发布最终姿态和可视化
  void RunTracking(const std_msgs::msg::Header &header, const cv::Mat &rgb, const cv::Mat &depth)
  {
    // 确保滤波器已初始化
    if (!pose_filter_.isInitialized())
    {
      pose_filter_.initialize(last_pose_,
                              static_cast<float>(pose_filter_default_dt_),
                              pose_filter_params_);
    }

    // SE(3)滤波器预测
    const float dt = ComputePoseFilterDt(header);
    const Eigen::Matrix4f predicted_pose = pose_filter_.predict(dt);

    // FoundationPose跟踪(从预测开始优化)
    Eigen::Matrix4f observed_pose;
    const bool ok =
        foundation_pose_->Track(rgb,
                                depth,
                                predicted_pose,
                                object_name_,
                                observed_pose,
                                track_refine_iters_);
    ++filter_frame_id_;

    // 跟踪失败: 使用滤波器预测，拒绝该帧观测
    if (!ok)
    {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "FoundationPose tracking failed on the current stereo frame. Publishing "
                           "the SE(3) filter prediction.");
      last_pose_ = pose_filter_.rejectObservation();
      UpdatePoseFilterTimestamp(header);
      PublishPose(header, last_pose_);
      PublishVisualization(header, rgb, last_pose_);
      RCLCPP_INFO(this->get_logger(),
                  "Pose filter frame=%lu quality=%.4f mahalanobis=%.4f rejected translation_residual=%.6f "
                  "rotation_residual=%.6f",
                  static_cast<unsigned long>(filter_frame_id_),
                  pose_filter_.getLastQuality(),
                  pose_filter_.getLastMahalanobisDistance(),
                  pose_filter_.getLastTranslationResidualNorm(),
                  pose_filter_.getLastRotationResidualNorm());
      return;
    }

    // 跟踪成功: 用观测值更新滤波器(自适应滤波可能拒绝异常值)
    last_pose_ = pose_filter_.update(observed_pose);
    UpdatePoseFilterTimestamp(header);
    RCLCPP_INFO(this->get_logger(),
                "Pose filter frame=%lu quality=%.4f mahalanobis=%.4f %s translation_residual=%.6f "
                "rotation_residual=%.6f",
                static_cast<unsigned long>(filter_frame_id_),
                pose_filter_.getLastQuality(),
                pose_filter_.getLastMahalanobisDistance(),
                pose_filter_.wasLastObservationAccepted() ? "accepted" : "rejected",
                pose_filter_.getLastTranslationResidualNorm(),
                pose_filter_.getLastRotationResidualNorm());
    PublishPose(header, last_pose_);
    PublishVisualization(header, rgb, last_pose_);
  }

  // 计算两帧间的时间步长(用于SE(3)滤波器)
  // 从ROS时间戳计算，若时间戳无效或间隔过大则使用默认值
  float ComputePoseFilterDt(const std_msgs::msg::Header &header) const
  {
    // 默认时间步长(通常为1/30秒)
    const double fallback_dt =
        std::isfinite(pose_filter_default_dt_) && pose_filter_default_dt_ > 0.0
            ? pose_filter_default_dt_
            : 1.0 / 30.0;

    const double stamp_sec = StampToSeconds(header.stamp);
    if (!has_last_pose_timestamp_ || stamp_sec <= 0.0)
    {
      return static_cast<float>(fallback_dt);
    }

    // 计算时间差，异常情况使用默认值(防止跳帧或时钟倒退)
    const double dt = stamp_sec - last_pose_timestamp_sec_;
    if (!std::isfinite(dt) || dt <= 0.0 || dt > 1.0)
    {
      return static_cast<float>(fallback_dt);
    }
    return static_cast<float>(dt);
  }

  // 更新上次处理帧的时间戳(用于下一帧的dt计算)
  void UpdatePoseFilterTimestamp(const std_msgs::msg::Header &header)
  {
    last_pose_timestamp_sec_ = StampToSeconds(header.stamp);
    has_last_pose_timestamp_ = true;
  }

  // 发布估计的6D姿态
  // 可选地覆盖frame_id(用于坐标系标识)
  void PublishPose(std_msgs::msg::Header header, const Eigen::Matrix4f &pose)
  {
    if (!pose_frame_id_.empty())
    {
      header.frame_id = pose_frame_id_;
    }

    pose_pub_->publish(PoseMatrixToPoseStamped(pose, header));
  }

  // 可视化函数: 在RGB图上绘制3D边界框并发布/显示
  // 可选: ROS话题发布或OpenCV窗口显示
  void PublishVisualization(std_msgs::msg::Header header,
                            const cv::Mat        &rgb,
                            const Eigen::Matrix4f &pose)
  {
    // 如果不需要可视化，直接返回
    if (!publish_visualization_ && !show_visualization_window_)
    {
      return;
    }

    // RGB转BGR(OpenCV标准格式)
    cv::Mat visualization_bgr;
    cv::cvtColor(rgb, visualization_bgr, cv::COLOR_RGB2BGR);

    // 从姿态和网格获取3D边界框
    const auto draw_pose = detection_6d::ConvertPoseMesh2BBox(pose, mesh_loader_);
    Draw3DBoundingBox(intrinsic_, draw_pose, mesh_loader_->GetObjectDimension(), visualization_bgr);

    // 发布到ROS话题
    if (publish_visualization_ && visualization_pub_ != nullptr)
    {
      auto image_msg =
          cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, visualization_bgr).toImageMsg();
      visualization_pub_->publish(*image_msg);
    }

    // 在OpenCV窗口显示
    if (show_visualization_window_)
    {
      cv::imshow(visualization_window_name_, visualization_bgr);
      cv::waitKey(1);
    }
  }

private:
  // ===== ROS图像订阅参数 =====
  int         image_reliability_{1};
  std::string left_image_topic_;
  std::string right_image_topic_;
  int         sync_queue_size_{10};
  double      max_sync_interval_sec_{0.07};

  // ===== 立体深度估计模型参数 =====
  std::vector<std::string> stereo_engine_file_path_;
  std::string              stereo_model_type_;
  int                      model_input_height_{448};
  int                      model_input_width_{640};
  double                   min_depth_meters_{0.1};
  double                   max_depth_meters_{100.0};
  bool                     use_filtered_depth_for_pose_{false};
  std::string              caminfo_path_;

  // ===== FoundationPose 6D姿态估计模型参数 =====
  std::string refiner_engine_path_;
  std::string scorer_engine_path_;
  std::string mesh_path_;
  std::string object_name_;
  std::string mask_image_path_;
  std::string pose_topic_;
  std::string pose_frame_id_;
  std::string visualization_topic_;
  std::string visualization_window_name_;

  // ===== 推理配置 =====
  int  max_input_image_height_{1080};
  int  max_input_image_width_{1920};
  int  mask_poll_interval_ms_{200};
  bool resize_mask_to_input_{false};
  bool publish_visualization_{false};
  bool show_visualization_window_{false};

  // ===== 优化参数 =====
  size_t register_refine_iters_{5};   // 首帧配准的迭代次数
  size_t track_refine_iters_{2};      // 跟踪阶段的迭代次数
  double pose_filter_default_dt_{1.0 / 30.0};
  foundationpose_filter::AdaptiveSE3PoseFilterParams pose_filter_params_;

  // ===== ROS配置 =====
  rclcpp::QoS image_qos_profile_;

  // ===== 相机标定和校正 =====
  StereoCalibration      calibration_;
  StereoRectificationMaps rectification_maps_;
  bool                   rectification_ready_{false};
  depth_confidence::ConfidenceParameters confidence_params_;
  cv::Mat previous_depth_;                          // 用于深度置信度计算
  std::unique_ptr<StereoEstimator> stereo_estimator_;

  // ===== 相机内参 =====
  Eigen::Matrix3f intrinsic_{Eigen::Matrix3f::Identity()};

  // ===== 姿态估计状态 =====
  bool                    has_pose_{false};        // 是否已完成首帧配准
  bool                    has_cached_rgbd_{false}; // 是否已缓存RGBD
  cv::Mat                 latest_rgb_;
  cv::Mat                 latest_depth_;
  std_msgs::msg::Header   latest_header_;
  Eigen::Matrix4f         last_pose_{Eigen::Matrix4f::Identity()};
  foundationpose_filter::AdaptiveSE3PoseFilter pose_filter_;
  bool                    has_last_pose_timestamp_{false};
  double                  last_pose_timestamp_sec_{0.0};
  uint64_t                filter_frame_id_{0};     // 滤波器处理帧计数

  // ===== 线程同步 =====
  std::mutex process_mutex_;                       // 保护推理过程，防止并发

  // ===== 推理引擎 =====
  std::shared_ptr<detection_6d::BaseMeshLoader>         mesh_loader_;
  std::shared_ptr<detection_6d::Base6DofDetectionModel> foundation_pose_;

  // ===== ROS消息接口 =====
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         visualization_pub_;
  SubscriberT                                                   left_image_sub_;
  SubscriberT                                                   right_image_sub_;
  std::shared_ptr<StereoSyncer>                                 stereo_sync_;
  rclcpp::TimerBase::SharedPtr                                  mask_poll_timer_;
};

// 主函数: 初始化ROS，创建节点，开始消息处理循环
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    // 创建FoundationPose立体跟踪节点
    auto node = std::make_shared<FoundationPoseStereoTrackerNode>();
    // 进入事件循环，处理ROS消息和定时器回调
    rclcpp::spin(node);
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("foundationpose_stereo_tracker_node"),
                 "Node startup failed: %s",
                 e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
