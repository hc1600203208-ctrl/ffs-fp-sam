#ifndef DEPTH_CONFIDENCE_HPP
#define DEPTH_CONFIDENCE_HPP

#include <opencv2/opencv.hpp>

namespace depth_confidence {

struct ConfidenceParameters {
    int patch_radius = 2;
    int texture_ksize = 5;

    float sigma_photo = 0.12f;
    float sigma_e = 0.15f;
    float sigma_t = 0.02f;
    float alpha_edge = 0.8f;

    float w_photo = 0.45f;
    float w_tex = 0.20f;
    float w_grad = 0.20f;
    float w_tmp = 0.15f;

    float conf_threshold = 0.35f;
    float weight_gamma = 1.5f;
    float eps = 1e-6f;
};

cv::Mat compute_photometric_confidence(
    const cv::Mat& left_img,
    const cv::Mat& right_img,
    const cv::Mat& depth,
    float fx,
    float baseline,
    int patch_radius = 2,
    float sigma_photo = 0.12f);

cv::Mat compute_texture_confidence(
    const cv::Mat& left_img_gray,
    int ksize = 5);

cv::Mat compute_depth_edge_confidence(
    const cv::Mat& depth,
    const cv::Mat& left_img_gray,
    float alpha_edge = 0.8f,
    float sigma_e = 0.15f);

cv::Mat compute_temporal_confidence_simple(
    const cv::Mat& depth,
    const cv::Mat& depth_prev = cv::Mat(),
    float sigma_t = 0.02f);

cv::Mat compute_confidence_map(
    const cv::Mat& left_img,
    const cv::Mat& right_img,
    const cv::Mat& depth,
    float fx,
    float fy,
    float cx,
    float cy,
    float baseline,
    const cv::Mat& depth_prev = cv::Mat(),
    const ConfidenceParameters& params = ConfidenceParameters{});

cv::Mat filter_depth_with_confidence(
    const cv::Mat& depth,
    const cv::Mat& conf,
    float threshold = 0.35f);

cv::Mat confidence_to_weight(
    const cv::Mat& conf,
    float gamma = 1.5f);

}  // namespace depth_confidence

#endif  // DEPTH_CONFIDENCE_HPP
