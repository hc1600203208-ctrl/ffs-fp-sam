#include "grounded_sam/image_utils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace grounded_sam {

cv::Mat readRgbImage(const std::string& path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + path);
  }
  cv::Mat rgb;
  if (image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  } else {
    throw std::runtime_error("Unsupported image channel count: " + std::to_string(image.channels()));
  }
  return rgb;
}

DinoPreprocessResult preprocessForDino(const cv::Mat& rgb, int target_h, int target_w) {
  DinoPreprocessResult out;
  out.original_h = rgb.rows;
  out.original_w = rgb.cols;

  const int short_side = 800;
  const int max_side = 1333;
  const int min_orig = std::min(rgb.rows, rgb.cols);
  const int max_orig = std::max(rgb.rows, rgb.cols);
  int size = short_side;
  if (static_cast<float>(max_orig) / static_cast<float>(min_orig) * size > max_side) {
    size = static_cast<int>(std::round(static_cast<float>(max_side) * min_orig / max_orig));
  }
  if (rgb.cols < rgb.rows) {
    out.resized_w = size;
    out.resized_h = static_cast<int>(static_cast<float>(size) * rgb.rows / rgb.cols);
  } else {
    out.resized_h = size;
    out.resized_w = static_cast<int>(static_cast<float>(size) * rgb.cols / rgb.rows);
  }
  out.resized_h = std::min(out.resized_h, target_h);
  out.resized_w = std::min(out.resized_w, target_w);
  out.scale = static_cast<float>(out.resized_w) / static_cast<float>(rgb.cols);

  cv::resize(rgb, out.resized_rgb, cv::Size(out.resized_w, out.resized_h), 0, 0, cv::INTER_LINEAR);
  out.chw.assign(3 * target_h * target_w, 0.0f);
  out.mask.assign(target_h * target_w, 1);

  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stdv[3] = {0.229f, 0.224f, 0.225f};
  for (int y = 0; y < out.resized_h; ++y) {
    const auto* row = out.resized_rgb.ptr<cv::Vec3b>(y);
    for (int x = 0; x < out.resized_w; ++x) {
      out.mask[y * target_w + x] = 0;
      for (int c = 0; c < 3; ++c) {
        const float v = static_cast<float>(row[x][c]) / 255.0f;
        out.chw[c * target_h * target_w + y * target_w + x] = (v - mean[c]) / stdv[c];
      }
    }
  }
  return out;
}

SamPreprocessResult preprocessForSam(const cv::Mat& rgb, int target_side) {
  SamPreprocessResult out;
  out.original_h = rgb.rows;
  out.original_w = rgb.cols;
  out.scale = static_cast<float>(target_side) / static_cast<float>(std::max(rgb.rows, rgb.cols));
  out.resized_h = static_cast<int>(std::round(rgb.rows * out.scale));
  out.resized_w = static_cast<int>(std::round(rgb.cols * out.scale));

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(out.resized_w, out.resized_h), 0, 0, cv::INTER_LINEAR);
  out.chw.assign(3 * target_side * target_side, 0.0f);

  const float pixel_mean[3] = {123.675f, 116.28f, 103.53f};
  const float pixel_std[3] = {58.395f, 57.12f, 57.375f};

  for (int y = 0; y < out.resized_h; ++y) {
    const auto* row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < out.resized_w; ++x) {
      for (int c = 0; c < 3; ++c) {
        out.chw[c * target_side * target_side + y * target_side + x] =
            (static_cast<float>(row[x][c]) - pixel_mean[c]) / pixel_std[c];
      }
    }
  }
  return out;
}

cv::Rect2f dinoCxcywhToOriginalXyxy(const float* box, int original_w, int original_h) {
  const float cx = box[0] * original_w;
  const float cy = box[1] * original_h;
  const float w = box[2] * original_w;
  const float h = box[3] * original_h;
  return cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
}

cv::Rect2f clampBox(const cv::Rect2f& box, int width, int height) {
  const float x1 = std::clamp(box.x, 0.0f, static_cast<float>(width - 1));
  const float y1 = std::clamp(box.y, 0.0f, static_cast<float>(height - 1));
  const float x2 = std::clamp(box.x + box.width, 0.0f, static_cast<float>(width - 1));
  const float y2 = std::clamp(box.y + box.height, 0.0f, static_cast<float>(height - 1));
  return cv::Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));
}

std::vector<std::string> listInputImages(const std::string& input) {
  namespace fs = std::filesystem;
  std::vector<std::string> paths;
  const fs::path p(input);
  if (fs::is_regular_file(p)) {
    paths.push_back(p.string());
    return paths;
  }
  if (!fs::is_directory(p)) {
    throw std::runtime_error("Input is neither file nor directory: " + input);
  }
  for (const auto& entry : fs::directory_iterator(p)) {
    if (!entry.is_regular_file()) continue;
    const auto ext = entry.path().extension().string();
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

void saveMaskPng(const std::string& path, const cv::Mat& mask) {
  if (!cv::imwrite(path, mask)) {
    throw std::runtime_error("Failed to save mask: " + path);
  }
}

}  // namespace grounded_sam
