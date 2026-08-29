#include "grounded_sam/pipeline.hpp"

namespace grounded_sam {

GroundedSamPipeline::GroundedSamPipeline(const PipelineOptions& options)
    : options_(options),
      dino_(options.dino_engine),
      sam_(options.sam_encoder_engine, options.sam_decoder_engine) {}

cv::Mat GroundedSamPipeline::run(const cv::Mat& rgb) {
  const auto detections = dino_.detect(rgb, options_.box_threshold, options_.max_detections);
  return sam_.segment(rgb, detections);
}

}  // namespace grounded_sam

