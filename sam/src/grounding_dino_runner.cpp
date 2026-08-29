#include "grounded_sam/grounding_dino_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <limits>

namespace grounded_sam {

namespace {

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

bool debugDinoEnabled() {
  const char* value = std::getenv("GS_DINO_DEBUG");
  if (!value) return false;
  const std::string normalized = value;
  return normalized == "1" || normalized == "true" || normalized == "TRUE" ||
         normalized == "yes" || normalized == "on";
}

void printTensorStats(const TensorBuffer& buffer, const char* label) {
  const auto values = tensorToFloatVector(buffer);
  std::size_t finite_count = 0;
  std::size_t nan_count = 0;
  std::size_t inf_count = 0;
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  for (float v : values) {
    if (std::isnan(v)) {
      ++nan_count;
      continue;
    }
    if (!std::isfinite(v)) {
      ++inf_count;
      continue;
    }
    ++finite_count;
    min_value = std::min(min_value, v);
    max_value = std::max(max_value, v);
  }
  std::cerr << "[GS_DINO_DEBUG] " << label << " name=" << buffer.name
            << " dtype=" << static_cast<int>(buffer.dtype)
            << " dims=" << dimsToString(buffer.dims)
            << " finite=" << finite_count
            << " nan=" << nan_count
            << " inf=" << inf_count;
  if (finite_count > 0) {
    std::cerr << " min=" << min_value << " max=" << max_value;
  }
  std::cerr << '\n';
  const std::size_t preview = std::min<std::size_t>(values.size(), 8);
  if (preview > 0) {
    std::cerr << "[GS_DINO_DEBUG] " << label << " preview:";
    for (std::size_t i = 0; i < preview; ++i) {
      std::cerr << ' ' << values[i];
    }
    std::cerr << '\n';
  }
}

}  // namespace

GroundingDinoRunner::GroundingDinoRunner(const std::string& engine_path)
    : engine_(std::make_unique<TrtEngine>(engine_path)) {
  engine_->setInputShape("images", nvinfer1::Dims4{1, 3, input_h_, input_w_});
  if (engine_->hasTensor("masks")) {
    nvinfer1::Dims mask_dims;
    mask_dims.nbDims = 3;
    mask_dims.d[0] = 1;
    mask_dims.d[1] = input_h_;
    mask_dims.d[2] = input_w_;
    engine_->setInputShape("masks", mask_dims);
  }
}

std::vector<Detection> GroundingDinoRunner::detect(const cv::Mat& rgb, float box_threshold,
                                                   int max_detections) {
  auto prep = preprocessForDino(rgb, input_h_, input_w_);
  auto& image_tensor = engine_->tensor("images");
  copyFloatToTensor(image_tensor, prep.chw.data(), prep.chw.size());

  if (engine_->hasTensor("masks")) {
    auto& mask_tensor = engine_->tensor("masks");
    copyBoolToTensor(mask_tensor, prep.mask.data(), prep.mask.size());
  }

  engine_->infer();

  const auto& logits_buf = engine_->tensor("pred_logits");
  const auto& boxes_buf = engine_->tensor("pred_boxes");
  if (debugDinoEnabled()) {
    printTensorStats(logits_buf, "pred_logits");
    printTensorStats(boxes_buf, "pred_boxes");
  }
  const std::vector<float> logits = tensorToFloatVector(logits_buf);
  const std::vector<float> boxes = tensorToFloatVector(boxes_buf);

  if (logits_buf.dims.nbDims != 3 || boxes_buf.dims.nbDims != 3 || boxes_buf.dims.d[2] != 4) {
    throw std::runtime_error("Unexpected DINO output shapes");
  }

  const int queries = logits_buf.dims.d[1];
  const int tokens = logits_buf.dims.d[2];
  std::vector<Detection> detections;
  detections.reserve(queries);

  for (int q = 0; q < queries; ++q) {
    float max_logit = -1e30f;
    for (int t = 0; t < tokens; ++t) {
      max_logit = std::max(max_logit, logits[static_cast<std::size_t>(q) * tokens + t]);
    }
    const float score = sigmoid(max_logit);
    if (score < box_threshold) continue;

    const cv::Rect2f box = clampBox(
        dinoCxcywhToOriginalXyxy(boxes.data() + static_cast<std::size_t>(q) * 4, rgb.cols, rgb.rows),
        rgb.cols, rgb.rows);
    if (box.width <= 1.0f || box.height <= 1.0f) continue;
    detections.push_back({box, score});
  }

  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (max_detections > 0 && static_cast<int>(detections.size()) > max_detections) {
    detections.resize(static_cast<std::size_t>(max_detections));
  }
  if (debugDinoEnabled()) {
    for (const auto& detection : detections) {
      std::cerr << "[GS_DINO_DEBUG] detection score=" << detection.score
                << " xyxy=" << detection.xyxy.x << "," << detection.xyxy.y << ","
                << detection.xyxy.x + detection.xyxy.width << ","
                << detection.xyxy.y + detection.xyxy.height << '\n';
    }
  }
  std::cerr << "GroundingDINO detections: " << detections.size() << '\n';
  return detections;
}

}  // namespace grounded_sam
