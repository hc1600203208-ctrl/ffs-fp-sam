#include "grounded_sam/pipeline.hpp"

namespace grounded_sam {

// PipelineOptions 来自 ROS 节点；同一实例只负责一个目标的模型和掩码。

GroundedSamPipeline::GroundedSamPipeline(const PipelineOptions& options)
    : options_(options),
      dino_(options.dino_engine, options.dino_onnx),
      sam_(options.sam_encoder_engine, options.sam_decoder_engine) {}

std::vector<Detection> GroundedSamPipeline::detect(const cv::Mat& rgb) {
  return dino_.detect(rgb, options_.box_threshold, options_.max_detections);
}

cv::Mat GroundedSamPipeline::segment(const cv::Mat& rgb, const std::vector<Detection>& detections) {
  return sam_.segment(rgb, detections);
}

cv::Mat GroundedSamPipeline::run(const cv::Mat& rgb) {
  return segment(rgb, detect(rgb));
}

}  // namespace grounded_sam
