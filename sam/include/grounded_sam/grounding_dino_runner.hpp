#pragma once

#include "grounded_sam/image_utils.hpp"
#include "grounded_sam/trt_utils.hpp"

#include <memory>
#include <string>
#include <vector>

namespace grounded_sam {

class GroundingDinoRunner {
 public:
  explicit GroundingDinoRunner(const std::string& engine_path);
  std::vector<Detection> detect(const cv::Mat& rgb, float box_threshold, int max_detections);

 private:
  std::unique_ptr<TrtEngine> engine_;
  int input_h_{800};
  int input_w_{1066};
};

}  // namespace grounded_sam
