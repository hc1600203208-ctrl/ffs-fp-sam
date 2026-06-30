#include "depth_confidence.hpp"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>

namespace depth_confidence {
namespace {

constexpr float kPercentile10 = 0.10f;
constexpr float kPercentile90 = 0.90f;

inline bool is_valid_depth_value(float value) {
    return std::isfinite(value) && value > 0.0f;
}

cv::Mat ensure_gray_float01(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported image channel count for grayscale conversion.");
    }

    cv::Mat gray_float;
    switch (gray.depth()) {
        case CV_8U:
            gray.convertTo(gray_float, CV_32F, 1.0 / 255.0);
            break;
        case CV_16U:
            gray.convertTo(gray_float, CV_32F, 1.0 / 65535.0);
            break;
        case CV_16S:
            gray.convertTo(gray_float, CV_32F, 1.0 / 32767.0);
            break;
        case CV_32F:
            gray_float = gray.clone();
            break;
        case CV_64F:
            gray.convertTo(gray_float, CV_32F);
            break;
        default:
            gray.convertTo(gray_float, CV_32F);
            break;
    }

    double max_value = 0.0;
    cv::minMaxLoc(gray_float, nullptr, &max_value);
    if (max_value > 1.0) {
        const double scale = (max_value <= 255.0) ? (1.0 / 255.0) : (1.0 / max_value);
        gray_float *= static_cast<float>(scale);
    }

    cv::min(gray_float, 1.0f, gray_float);
    cv::max(gray_float, 0.0f, gray_float);
    return gray_float;
}

std::vector<float> collect_finite_values(const cv::Mat& image, const cv::Mat& mask = cv::Mat()) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(image.rows) * image.cols);

    for (int y = 0; y < image.rows; ++y) {
        const float* row = image.ptr<float>(y);
        const uint8_t* mask_row = mask.empty() ? nullptr : mask.ptr<uint8_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (mask_row != nullptr && mask_row[x] == 0) {
                continue;
            }

            const float value = row[x];
            if (std::isfinite(value)) {
                values.push_back(value);
            }
        }
    }

    return values;
}

float compute_percentile(std::vector<float> values, float percentile) {
    if (values.empty()) {
        return 0.0f;
    }

    percentile = std::clamp(percentile, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(std::floor(percentile * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

cv::Mat valid_depth_mask(const cv::Mat& depth) {
    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8UC1);
    for (int y = 0; y < depth.rows; ++y) {
        const float* row = depth.ptr<float>(y);
        uint8_t* mask_row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < depth.cols; ++x) {
            mask_row[x] = is_valid_depth_value(row[x]) ? 255 : 0;
        }
    }
    return mask;
}

cv::Mat robust_normalize(
    const cv::Mat& input,
    float p_low,
    float p_high,
    const cv::Mat& mask = cv::Mat(),
    float eps = 1e-6f) {
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32FC1);
    std::vector<float> values = collect_finite_values(input, mask);
    if (values.empty()) {
        return output;
    }

    const float low = compute_percentile(values, p_low);
    const float high = compute_percentile(std::move(values), p_high);
    const float denom = std::max(high - low, eps);

    for (int y = 0; y < input.rows; ++y) {
        const float* src_row = input.ptr<float>(y);
        const uint8_t* mask_row = mask.empty() ? nullptr : mask.ptr<uint8_t>(y);
        float* dst_row = output.ptr<float>(y);
        for (int x = 0; x < input.cols; ++x) {
            if (mask_row != nullptr && mask_row[x] == 0) {
                dst_row[x] = 0.0f;
                continue;
            }

            const float value = src_row[x];
            if (!std::isfinite(value)) {
                dst_row[x] = 0.0f;
                continue;
            }

            dst_row[x] = std::clamp((value - low) / denom, 0.0f, 1.0f);
        }
    }

    return output;
}

bool bilinear_sample_gray(const cv::Mat& image, float x, float y, float* value) {
    if (value == nullptr || image.empty()) {
        return false;
    }

    if (x < 0.0f || y < 0.0f || x >= static_cast<float>(image.cols - 1) || y >= static_cast<float>(image.rows - 1)) {
        return false;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float ax = x - static_cast<float>(x0);
    const float ay = y - static_cast<float>(y0);

    const float v00 = image.at<float>(y0, x0);
    const float v01 = image.at<float>(y0, x1);
    const float v10 = image.at<float>(y1, x0);
    const float v11 = image.at<float>(y1, x1);

    *value = (1.0f - ax) * (1.0f - ay) * v00 +
             ax * (1.0f - ay) * v01 +
             (1.0f - ax) * ay * v10 +
             ax * ay * v11;
    return true;
}

cv::Mat clip_float01(const cv::Mat& input) {
    cv::Mat output;
    cv::max(input, 0.0f, output);
    cv::min(output, 1.0f, output);
    output.convertTo(output, CV_32F);
    return output;
}

}  // namespace

cv::Mat compute_photometric_confidence(
    const cv::Mat& left_img,
    const cv::Mat& right_img,
    const cv::Mat& depth,
    float fx,
    float baseline,
    int patch_radius,
    float sigma_photo) {
    if (left_img.empty() || right_img.empty() || depth.empty()) {
        return {};
    }

    patch_radius = std::max(0, patch_radius);

    cv::Mat left_gray = ensure_gray_float01(left_img);
    cv::Mat right_gray = ensure_gray_float01(right_img);

    cv::Mat confidence = cv::Mat::zeros(depth.size(), CV_32FC1);
    const float disp_scale = fx * baseline;
    const float eps = 1e-6f;
    const int patch_size = 2 * patch_radius + 1;
    const float norm = 1.0f / static_cast<float>(patch_size * patch_size);

    #pragma omp parallel for
    for (int y = 0; y < depth.rows; ++y) {
        const float* depth_row = depth.ptr<float>(y);
        float* conf_row = confidence.ptr<float>(y);

        for (int x = 0; x < depth.cols; ++x) {
            const float depth_value = depth_row[x];
            if (!is_valid_depth_value(depth_value)) {
                conf_row[x] = 0.0f;
                continue;
            }

            if (x - patch_radius < 0 || x + patch_radius >= depth.cols ||
                y - patch_radius < 0 || y + patch_radius >= depth.rows) {
                conf_row[x] = 0.0f;
                continue;
            }

            const float disparity = disp_scale / (depth_value + eps);
            float sad = 0.0f;
            bool valid = true;

            for (int dy = -patch_radius; dy <= patch_radius && valid; ++dy) {
                for (int dx = -patch_radius; dx <= patch_radius; ++dx) {
                    const int xl = x + dx;
                    const int yl = y + dy;
                    const float xr = static_cast<float>(xl) - disparity;
                    float sampled_right = 0.0f;
                    if (!bilinear_sample_gray(right_gray, xr, static_cast<float>(yl), &sampled_right)) {
                        valid = false;
                        break;
                    }

                    const float left_value = left_gray.at<float>(yl, xl);
                    sad += std::fabs(left_value - sampled_right);
                }
            }

            if (!valid) {
                conf_row[x] = 0.0f;
                continue;
            }

            const float photo_err = sad * norm;
            conf_row[x] = std::exp(-photo_err / std::max(sigma_photo, eps));
        }
    }

    return confidence;
}

cv::Mat compute_texture_confidence(const cv::Mat& left_img_gray, int ksize) {
    if (left_img_gray.empty()) {
        return {};
    }

    ksize = std::max(1, ksize);
    if (ksize % 2 == 0) {
        ++ksize;
    }

    cv::Mat gray = ensure_gray_float01(left_img_gray);

    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);

    cv::Mat texture;
    cv::boxFilter(
        grad_mag,
        texture,
        CV_32F,
        cv::Size(ksize, ksize),
        cv::Point(-1, -1),
        true,
        cv::BORDER_REFLECT101);

    return robust_normalize(texture, kPercentile10, kPercentile90);
}

cv::Mat compute_depth_edge_confidence(
    const cv::Mat& depth,
    const cv::Mat& left_img_gray,
    float alpha_edge,
    float sigma_e) {
    if (depth.empty() || left_img_gray.empty()) {
        return {};
    }

    const cv::Mat depth_mask = valid_depth_mask(depth);
    cv::Mat confidence = cv::Mat::zeros(depth.size(), CV_32FC1);
    if (cv::countNonZero(depth_mask) == 0) {
        return confidence;
    }

    cv::Mat gray = ensure_gray_float01(left_img_gray);

    cv::Mat img_grad_x;
    cv::Mat img_grad_y;
    cv::Sobel(gray, img_grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, img_grad_y, CV_32F, 0, 1, 3);
    cv::Mat img_grad_mag;
    cv::magnitude(img_grad_x, img_grad_y, img_grad_mag);

    cv::Mat sanitized_depth = cv::Mat::zeros(depth.size(), CV_32FC1);
    depth.copyTo(sanitized_depth, depth_mask);

    cv::Mat depth_grad_x;
    cv::Mat depth_grad_y;
    cv::Sobel(sanitized_depth, depth_grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(sanitized_depth, depth_grad_y, CV_32F, 0, 1, 3);
    cv::Mat depth_grad_mag;
    cv::magnitude(depth_grad_x, depth_grad_y, depth_grad_mag);

    cv::Mat igrad_norm = robust_normalize(img_grad_mag, 0.0f, kPercentile90, depth_mask);
    cv::Mat dgrad_norm = robust_normalize(depth_grad_mag, 0.0f, kPercentile90, depth_mask);

    const float eps = 1e-6f;
    for (int y = 0; y < depth.rows; ++y) {
        const uint8_t* mask_row = depth_mask.ptr<uint8_t>(y);
        const float* igrad_row = igrad_norm.ptr<float>(y);
        const float* dgrad_row = dgrad_norm.ptr<float>(y);
        float* conf_row = confidence.ptr<float>(y);

        for (int x = 0; x < depth.cols; ++x) {
            if (mask_row[x] == 0) {
                conf_row[x] = 0.0f;
                continue;
            }

            const float edge_error = std::max(0.0f, dgrad_row[x] - alpha_edge * igrad_row[x]);
            conf_row[x] = std::exp(-edge_error / std::max(sigma_e, eps));
        }
    }

    return confidence;
}

cv::Mat compute_temporal_confidence_simple(
    const cv::Mat& depth,
    const cv::Mat& depth_prev,
    float sigma_t) {
    if (depth.empty()) {
        return {};
    }

    cv::Mat confidence = cv::Mat::zeros(depth.size(), CV_32FC1);
    const cv::Mat depth_mask = valid_depth_mask(depth);
    const float eps = 1e-6f;

    if (depth_prev.empty() || depth_prev.size() != depth.size() || depth_prev.type() != depth.type()) {
        confidence.setTo(1.0f, depth_mask);
        return confidence;
    }

    for (int y = 0; y < depth.rows; ++y) {
        const float* depth_row = depth.ptr<float>(y);
        const float* prev_row = depth_prev.ptr<float>(y);
        const uint8_t* mask_row = depth_mask.ptr<uint8_t>(y);
        float* conf_row = confidence.ptr<float>(y);

        for (int x = 0; x < depth.cols; ++x) {
            if (mask_row[x] == 0 || !is_valid_depth_value(prev_row[x])) {
                conf_row[x] = 0.0f;
                continue;
            }

            const float err = std::fabs(depth_row[x] - prev_row[x]);
            conf_row[x] = std::exp(-err / std::max(sigma_t, eps));
        }
    }

    return confidence;
}

cv::Mat compute_confidence_map(
    const cv::Mat& left_img,
    const cv::Mat& right_img,
    const cv::Mat& depth,
    float fx,
    float fy,
    float cx,
    float cy,
    float baseline,
    const cv::Mat& depth_prev,
    const ConfidenceParameters& params) {
    static_cast<void>(fy);
    static_cast<void>(cx);
    static_cast<void>(cy);

    if (left_img.empty() || right_img.empty() || depth.empty()) {
        return {};
    }

    cv::Mat left_gray = ensure_gray_float01(left_img);
    cv::Mat photo = compute_photometric_confidence(
        left_gray,
        right_img,
        depth,
        fx,
        baseline,
        params.patch_radius,
        params.sigma_photo);
    cv::Mat texture = compute_texture_confidence(left_gray, params.texture_ksize);
    cv::Mat grad = compute_depth_edge_confidence(depth, left_gray, params.alpha_edge, params.sigma_e);
    cv::Mat temporal = compute_temporal_confidence_simple(depth, depth_prev, params.sigma_t);

    cv::Mat confidence(depth.size(), CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat depth_mask = valid_depth_mask(depth);

    const float eps = std::max(params.eps, 1e-6f);
    for (int y = 0; y < depth.rows; ++y) {
        const uint8_t* mask_row = depth_mask.ptr<uint8_t>(y);
        const float* photo_row = photo.ptr<float>(y);
        const float* tex_row = texture.ptr<float>(y);
        const float* grad_row = grad.ptr<float>(y);
        const float* tmp_row = temporal.ptr<float>(y);
        float* conf_row = confidence.ptr<float>(y);

        for (int x = 0; x < depth.cols; ++x) {
            if (mask_row[x] == 0) {
                conf_row[x] = 0.0f;
                continue;
            }

            const float c_photo = std::pow(std::clamp(photo_row[x], eps, 1.0f), params.w_photo);
            const float c_tex = std::pow(std::clamp(tex_row[x], eps, 1.0f), params.w_tex);
            const float c_grad = std::pow(std::clamp(grad_row[x], eps, 1.0f), params.w_grad);
            const float c_tmp = std::pow(std::clamp(tmp_row[x], eps, 1.0f), params.w_tmp);
            conf_row[x] = c_photo * c_tex * c_grad * c_tmp;
        }
    }

    cv::GaussianBlur(confidence, confidence, cv::Size(3, 3), 0.8, 0.8, cv::BORDER_REFLECT101);
    confidence.setTo(0.0f, ~depth_mask);
    return clip_float01(confidence);
}

cv::Mat filter_depth_with_confidence(
    const cv::Mat& depth,
    const cv::Mat& conf,
    float threshold) {
    if (depth.empty() || conf.empty()) {
        return {};
    }

    cv::Mat filtered = depth.clone();
    for (int y = 0; y < filtered.rows; ++y) {
        float* depth_row = filtered.ptr<float>(y);
        const float* conf_row = conf.ptr<float>(y);
        for (int x = 0; x < filtered.cols; ++x) {
            if (!is_valid_depth_value(depth_row[x]) || !std::isfinite(conf_row[x]) || conf_row[x] < threshold) {
                depth_row[x] = 0.0f;
            }
        }
    }
    return filtered;
}

cv::Mat confidence_to_weight(const cv::Mat& conf, float gamma) {
    if (conf.empty()) {
        return {};
    }

    cv::Mat clipped = clip_float01(conf);
    cv::Mat weight;
    cv::pow(clipped, gamma, weight);
    weight.convertTo(weight, CV_32F);
    return weight;
}

}  // namespace depth_confidence
