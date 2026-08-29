#pragma once

#include "grounded_sam/grounding_dino_runner.hpp"
#include "grounded_sam/sam_runner.hpp"

namespace grounded_sam {

struct PipelineOptions {
  std::string dino_engine;
  std::string sam_encoder_engine;
  std::string sam_decoder_engine;
  float box_threshold{0.3f};
  int max_detections{16};
};

class GroundedSamPipeline {
 public:
  explicit GroundedSamPipeline(const PipelineOptions& options);
  cv::Mat run(const cv::Mat& rgb);

 private:
  PipelineOptions options_;
  GroundingDinoRunner dino_;
  SamRunner sam_;
};

}  // namespace grounded_sam

