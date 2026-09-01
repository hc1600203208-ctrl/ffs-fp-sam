#include "grounded_sam/grounding_dino_runner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
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

std::string dimsToString(const std::vector<int64_t>& dims) {
  std::string out = "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i) out += "x";
    out += std::to_string(dims[i]);
  }
  out += "]";
  return out;
}

std::vector<int64_t> dimsToVector(const nvinfer1::Dims& dims) {
  std::vector<int64_t> out;
  out.reserve(static_cast<std::size_t>(dims.nbDims));
  for (int i = 0; i < dims.nbDims; ++i) {
    out.push_back(dims.d[i]);
  }
  return out;
}

void printFloatStats(const std::vector<float>& values, const std::vector<int64_t>& dims,
                     const char* label) {
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
  std::cerr << "[GS_DINO_DEBUG] " << label << " backend=onnxruntime"
            << " dims=" << dimsToString(dims)
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

bool pathLooksLikeOnnx(const std::string& path) {
  return std::filesystem::path(path).extension() == ".onnx";
}

std::vector<Detection> postprocessDinoOutputs(const std::vector<float>& logits,
                                              const std::vector<int64_t>& logits_shape,
                                              const std::vector<float>& boxes,
                                              const std::vector<int64_t>& boxes_shape,
                                              int image_w, int image_h,
                                              float box_threshold, int max_detections) {
  if (logits_shape.size() != 3 || boxes_shape.size() != 3 || boxes_shape[2] != 4) {
    throw std::runtime_error("Unexpected DINO output shapes: logits=" +
                             dimsToString(logits_shape) + " boxes=" +
                             dimsToString(boxes_shape));
  }
  if (logits_shape[0] != 1 || boxes_shape[0] != 1) {
    throw std::runtime_error("Only batch size 1 is supported for DINO postprocess");
  }
  const int queries = static_cast<int>(logits_shape[1]);
  const int tokens = static_cast<int>(logits_shape[2]);
  if (boxes_shape[1] != queries) {
    throw std::runtime_error("DINO logits/boxes query count mismatch");
  }
  if (logits.size() != static_cast<std::size_t>(queries * tokens) ||
      boxes.size() != static_cast<std::size_t>(queries * 4)) {
    throw std::runtime_error("DINO output element count mismatch");
  }

  std::vector<Detection> detections;
  detections.reserve(static_cast<std::size_t>(queries));

  for (int q = 0; q < queries; ++q) {
    float max_logit = -1e30f;
    for (int t = 0; t < tokens; ++t) {
      max_logit = std::max(max_logit, logits[static_cast<std::size_t>(q) * tokens + t]);
    }
    const float score = sigmoid(max_logit);
    if (score < box_threshold) continue;

    const cv::Rect2f box = clampBox(
        dinoCxcywhToOriginalXyxy(boxes.data() + static_cast<std::size_t>(q) * 4, image_w,
                                 image_h),
        image_w, image_h);
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

}  // namespace

GroundingDinoRunner::GroundingDinoRunner(const std::string& engine_path)
    : GroundingDinoRunner(engine_path, "") {}

GroundingDinoRunner::GroundingDinoRunner(const std::string& engine_path,
                                         const std::string& onnx_path) {
  const bool use_onnxruntime = !onnx_path.empty() || pathLooksLikeOnnx(engine_path);
  const std::string model_path = !onnx_path.empty() ? onnx_path : engine_path;

  if (use_onnxruntime) {
#ifdef GROUNDED_SAM_WITH_ONNXRUNTIME
    backend_ = Backend::OnnxRuntime;
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "grounded_sam_dino");
    ort_session_options_ = std::make_unique<Ort::SessionOptions>();
    ort_session_options_->SetIntraOpNumThreads(4);
    ort_session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ort_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(),
                                                  *ort_session_options_);
#else
    throw std::runtime_error("This build does not include ONNX Runtime support for DINO");
#endif
  } else {
    backend_ = Backend::TensorRT;
    engine_ = std::make_unique<TrtEngine>(engine_path);
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
}

std::vector<Detection> GroundingDinoRunner::detect(const cv::Mat& rgb, float box_threshold,
                                                   int max_detections) {
  if (backend_ == Backend::OnnxRuntime) {
    return detectWithOnnxRuntime(rgb, box_threshold, max_detections);
  }
  return detectWithTensorRt(rgb, box_threshold, max_detections);
}

std::vector<Detection> GroundingDinoRunner::detectWithTensorRt(const cv::Mat& rgb,
                                                               float box_threshold,
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

  return postprocessDinoOutputs(logits, dimsToVector(logits_buf.dims), boxes,
                                dimsToVector(boxes_buf.dims), rgb.cols, rgb.rows,
                                box_threshold, max_detections);
}

std::vector<Detection> GroundingDinoRunner::detectWithOnnxRuntime(const cv::Mat& rgb,
                                                                  float box_threshold,
                                                                  int max_detections) {
#ifdef GROUNDED_SAM_WITH_ONNXRUNTIME
  auto prep = preprocessForDino(rgb, input_h_, input_w_);
  std::array<int64_t, 4> input_shape{1, 3, input_h_, input_w_};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, prep.chw.data(), prep.chw.size(), input_shape.data(), input_shape.size());

  std::array<const char*, 1> input_names{"images"};
  std::array<const char*, 2> output_names{"pred_logits", "pred_boxes"};
  auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr}, input_names.data(),
                                          &input_tensor, input_names.size(),
                                          output_names.data(), output_names.size());
  if (output_tensors.size() != 2) {
    throw std::runtime_error("Unexpected ONNX Runtime output count");
  }

  const auto logits_info = output_tensors[0].GetTensorTypeAndShapeInfo();
  const auto boxes_info = output_tensors[1].GetTensorTypeAndShapeInfo();
  const auto logits_shape = logits_info.GetShape();
  const auto boxes_shape = boxes_info.GetShape();
  const float* logits_data = output_tensors[0].GetTensorData<float>();
  const float* boxes_data = output_tensors[1].GetTensorData<float>();
  std::vector<float> logits(logits_data, logits_data + logits_info.GetElementCount());
  std::vector<float> boxes(boxes_data, boxes_data + boxes_info.GetElementCount());

  if (debugDinoEnabled()) {
    printFloatStats(logits, logits_shape, "pred_logits");
    printFloatStats(boxes, boxes_shape, "pred_boxes");
  }
  return postprocessDinoOutputs(logits, logits_shape, boxes, boxes_shape, rgb.cols, rgb.rows,
                                box_threshold, max_detections);
#else
  (void)rgb;
  (void)box_threshold;
  (void)max_detections;
  throw std::runtime_error("This build does not include ONNX Runtime support for DINO");
#endif
}

}  // namespace grounded_sam
