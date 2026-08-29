#include "grounded_sam/sam_runner.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace grounded_sam {

SamRunner::SamRunner(const std::string& image_encoder_engine, const std::string& mask_decoder_engine)
    : encoder_(std::make_unique<TrtEngine>(image_encoder_engine)),
      decoder_(std::make_unique<TrtEngine>(mask_decoder_engine)) {
  encoder_->setInputShape("image", nvinfer1::Dims4{1, 3, input_side_, input_side_});
  decoder_->setInputShape("image_embeddings", nvinfer1::Dims4{1, 256, 64, 64});
  decoder_->setInputShape("point_coords", nvinfer1::Dims3{1, 2, 2});
  decoder_->setInputShape("point_labels", nvinfer1::Dims2{1, 2});
  decoder_->setInputShape("mask_input", nvinfer1::Dims4{1, 1, 256, 256});
  nvinfer1::Dims has_mask_dims;
  has_mask_dims.nbDims = 1;
  has_mask_dims.d[0] = 1;
  decoder_->setInputShape("has_mask_input", has_mask_dims);
}

std::vector<float> SamRunner::encodeImage(const cv::Mat& rgb, SamPreprocessResult& prep) {
  prep = preprocessForSam(rgb, input_side_);
  auto& input = encoder_->tensor("image");
  copyFloatToTensor(input, prep.chw.data(), prep.chw.size());
  encoder_->infer();

  const auto& out = encoder_->tensor("image_embeddings");
  return tensorToFloatVector(out);
}

cv::Mat SamRunner::decodeOneBox(const std::vector<float>& embedding,
                                const SamPreprocessResult& prep,
                                const cv::Rect2f& box) {
  auto& embeddings = decoder_->tensor("image_embeddings");
  copyFloatToTensor(embeddings, embedding.data(), embedding.size());

  const float x1 = box.x * prep.scale;
  const float y1 = box.y * prep.scale;
  const float x2 = (box.x + box.width) * prep.scale;
  const float y2 = (box.y + box.height) * prep.scale;

  auto& point_coords_tensor = decoder_->tensor("point_coords");
  auto& point_labels_tensor = decoder_->tensor("point_labels");
  auto& mask_input_tensor = decoder_->tensor("mask_input");
  auto& has_mask_input_tensor = decoder_->tensor("has_mask_input");

  const std::array<float, 4> point_coords_values{x1, y1, x2, y2};
  const std::array<float, 2> point_labels_values{2.0f, 3.0f};
  copyFloatToTensor(point_coords_tensor, point_coords_values.data(), point_coords_values.size());
  copyFloatToTensor(point_labels_tensor, point_labels_values.data(), point_labels_values.size());
  std::fill(mask_input_tensor.host.begin(), mask_input_tensor.host.end(), 0);
  std::fill(has_mask_input_tensor.host.begin(), has_mask_input_tensor.host.end(), 0);

  decoder_->infer();

  const auto& mask_tensor = decoder_->tensor("low_res_masks");
  const std::vector<float> logits = tensorToFloatVector(mask_tensor);
  cv::Mat low_res(256, 256, CV_32FC1);
  std::memcpy(low_res.data, logits.data(), 256 * 256 * sizeof(float));

  cv::Mat upscaled;
  cv::resize(low_res, upscaled, cv::Size(input_side_, input_side_), 0, 0, cv::INTER_LINEAR);
  cv::Mat cropped = upscaled(cv::Rect(0, 0, prep.resized_w, prep.resized_h));
  cv::Mat original_logits;
  cv::resize(cropped, original_logits, cv::Size(prep.original_w, prep.original_h), 0, 0,
             cv::INTER_LINEAR);

  cv::Mat mask(prep.original_h, prep.original_w, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < prep.original_h; ++y) {
    const float* row = original_logits.ptr<float>(y);
    auto* out = mask.ptr<std::uint8_t>(y);
    for (int x = 0; x < prep.original_w; ++x) {
      out[x] = row[x] > 0.0f ? 255 : 0;
    }
  }
  return mask;
}

cv::Mat SamRunner::segment(const cv::Mat& rgb, const std::vector<Detection>& detections) {
  cv::Mat merged(rgb.rows, rgb.cols, CV_8UC1, cv::Scalar(0));
  if (detections.empty()) return merged;

  SamPreprocessResult prep;
  const std::vector<float> embedding = encodeImage(rgb, prep);
  for (const auto& det : detections) {
    cv::Mat mask = decodeOneBox(embedding, prep, det.xyxy);
    cv::bitwise_or(merged, mask, merged);
  }
  return merged;
}

}  // namespace grounded_sam
