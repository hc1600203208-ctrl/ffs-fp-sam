#pragma once

#include "grounded_sam/image_utils.hpp"
#include "grounded_sam/trt_utils.hpp"

#include <memory>
#include <string>
#include <vector>

namespace grounded_sam {

class SamRunner {
 public:
  SamRunner(const std::string& image_encoder_engine, const std::string& mask_decoder_engine);
  cv::Mat segment(const cv::Mat& rgb, const std::vector<Detection>& detections);

 private:
  std::vector<float> encodeImage(const cv::Mat& rgb, SamPreprocessResult& prep);
  cv::Mat decodeOneBox(const std::vector<float>& embedding, const SamPreprocessResult& prep,
                       const cv::Rect2f& box);

  std::unique_ptr<TrtEngine> encoder_;
  std::unique_ptr<TrtEngine> decoder_;
  int input_side_{1024};
};

}  // namespace grounded_sam

