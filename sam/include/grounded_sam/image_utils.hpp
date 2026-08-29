#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace grounded_sam {

struct Detection {
  cv::Rect2f xyxy;
  float score{0.0f};
};

struct DinoPreprocessResult {
  cv::Mat resized_rgb;
  int original_h{0};
  int original_w{0};
  int resized_h{0};
  int resized_w{0};
  float scale{1.0f};
  std::vector<float> chw;
  std::vector<std::uint8_t> mask;
};

struct SamPreprocessResult {
  int original_h{0};
  int original_w{0};
  int resized_h{0};
  int resized_w{0};
  float scale{1.0f};
  std::vector<float> chw;
};

cv::Mat readRgbImage(const std::string& path);
DinoPreprocessResult preprocessForDino(const cv::Mat& rgb, int target_h, int target_w);
SamPreprocessResult preprocessForSam(const cv::Mat& rgb, int target_side);
cv::Rect2f dinoCxcywhToOriginalXyxy(const float* box, int original_w, int original_h);
cv::Rect2f clampBox(const cv::Rect2f& box, int width, int height);
std::vector<std::string> listInputImages(const std::string& input);
void saveMaskPng(const std::string& path, const cv::Mat& mask);

}  // namespace grounded_sam
