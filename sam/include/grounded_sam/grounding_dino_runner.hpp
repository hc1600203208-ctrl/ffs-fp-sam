#pragma once

#include "grounded_sam/image_utils.hpp"
#include "grounded_sam/trt_utils.hpp"

#ifdef GROUNDED_SAM_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <memory>
#include <string>
#include <vector>

namespace grounded_sam {

class GroundingDinoRunner {
 public:
  explicit GroundingDinoRunner(const std::string& engine_path);
  GroundingDinoRunner(const std::string& engine_path, const std::string& onnx_path);
  std::vector<Detection> detect(const cv::Mat& rgb, float box_threshold, int max_detections);

 private:
  enum class Backend {
    TensorRT,
    OnnxRuntime,
  };

  std::vector<Detection> detectWithTensorRt(const cv::Mat& rgb, float box_threshold,
                                            int max_detections);
  std::vector<Detection> detectWithOnnxRuntime(const cv::Mat& rgb, float box_threshold,
                                               int max_detections);

  Backend backend_{Backend::TensorRT};
  std::unique_ptr<TrtEngine> engine_;
#ifdef GROUNDED_SAM_WITH_ONNXRUNTIME
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::SessionOptions> ort_session_options_;
  std::unique_ptr<Ort::Session> ort_session_;
#endif
  int input_h_{800};
  int input_w_{1066};
};

}  // namespace grounded_sam
