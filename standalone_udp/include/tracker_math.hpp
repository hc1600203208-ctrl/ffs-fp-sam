#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "stereo_calibration_utils.hpp"

namespace standalone_tracker {
constexpr std::uint64_t kPoseBatchSize = 252;
constexpr std::uint64_t kTrackingBatchSize = 252;
constexpr int kCropHeight = 160;
constexpr int kCropWidth = 160;
constexpr float kRadiansToDegrees = 57.29577951308232F;

struct PreparedImage
{
  // 保留缩放和补边参数，供视差深度回映射到原分辨率；image 是 FFS 输入尺寸。
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
Eigen::Matrix3f BuildIntrinsicMatrix(double fx, double fy, double cx, double cy)
{
  // FP 渲染/反投影使用原始左图的 K，不使用缩放后的 FFS 输入 K。
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
cv::Mat ConvertMaskImage(cv::Mat mask)
{
  // 磁盘 PNG 统一转 CV_8UC1 二值图；正像素为目标，零为背景。
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
  // 等比例缩放后补边到 engine 固定输入尺寸；不能直接拉伸视差几何。
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
  // OpenCV stereoRectify 的 P2(0,3)=-fx*baseline，取绝对值恢复米制基线。
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
  // P1 是整流左相机 K；对 resize 和居中补边分别修正焦距和主点。
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
  // 针孔双目公式 Z=fx*baseline/disparity；fx 是补边缩放后的像素焦距，baseline 为米。
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
  // 先去掉 FFS 输入补边，再恢复整流图尺寸；随后才可逆映射到原始左相机。
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
Eigen::Vector3f RotationMatrixToRpyDegrees(const Eigen::Matrix3f &rotation)
{
  // 仅用于人类可读的日志/叠字，ROS 消息仍发布四元数。
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
  // 双层 OpenCV putText 提高文字在复杂图像上的对比度。
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3,
              cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

void DrawFirstPoseOverlay(cv::Mat &image, const Eigen::Matrix4f &pose)
{
  // 只给首个目标显示平滑位姿数值，不修改 ROS pose topic。
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
  // 将 mesh 3D 包围盒用 K 投影到原始左图像素；OpenCV 绘图颜色按 BGR 指定。
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

} // namespace standalone_tracker
