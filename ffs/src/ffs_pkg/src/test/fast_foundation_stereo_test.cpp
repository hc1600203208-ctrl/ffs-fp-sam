#include <iostream>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#include "../estimator/fast_foundation_stereo_estimator.h"

namespace {

cv::Mat prepareStereoInput(const cv::Mat& input, int target_width, int target_height) {
    if (input.empty()) {
        return {};
    }

    cv::Mat color;
    if (input.channels() == 1) {
        cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
    } else if (input.channels() == 3) {
        color = input.clone();
    } else {
        throw std::runtime_error("Unsupported image channel count.");
    }

    cv::Mat resized;
    const double scale = std::min(
        static_cast<double>(target_width) / static_cast<double>(color.cols),
        static_cast<double>(target_height) / static_cast<double>(color.rows));

    const int resized_width = std::max(1, static_cast<int>(std::round(color.cols * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(color.rows * scale)));

    cv::resize(color, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    const int pad_width = target_width - resized_width;
    const int pad_height = target_height - resized_height;
    const int pad_left = pad_width / 2;
    const int pad_right = pad_width - pad_left;
    const int pad_top = pad_height / 2;
    const int pad_bottom = pad_height - pad_top;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized, padded,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv::BORDER_REPLICATE);

    return padded;
}

void printDisparityStats(const cv::Mat& disparity) {
    if (disparity.empty()) {
        std::cout << "[Main] Disparity is empty." << std::endl;
        return;
    }

    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    size_t valid_count = 0;
    size_t zero_like_count = 0;
    size_t nan_count = 0;

    for (int y = 0; y < disparity.rows; ++y) {
        const float* row = disparity.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float value = row[x];
            if (!std::isfinite(value)) {
                ++nan_count;
                continue;
            }

            min_val = std::min(min_val, static_cast<double>(value));
            max_val = std::max(max_val, static_cast<double>(value));
            sum += value;
            ++valid_count;

            if (std::abs(value) < 1e-6f) {
                ++zero_like_count;
            }
        }
    }

    if (valid_count == 0) {
        std::cout << "[Main] Disparity has no finite values. NaN/Inf count: " << nan_count << std::endl;
        return;
    }

    std::cout << "[Main] Disparity stats"
              << " min=" << min_val
              << " max=" << max_val
              << " mean=" << (sum / static_cast<double>(valid_count))
              << " valid=" << valid_count
              << " zero_like=" << zero_like_count
              << " nan_or_inf=" << nan_count
              << std::endl;
}

cv::Mat visualizeDisparityRobust(const cv::Mat& disparity) {
    std::vector<float> valid_values;
    valid_values.reserve(static_cast<size_t>(disparity.rows) * disparity.cols);

    for (int y = 0; y < disparity.rows; ++y) {
        const float* row = disparity.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float value = row[x];
            if (std::isfinite(value) && value > 0.0f) {
                valid_values.push_back(value);
            }
        }
    }

    cv::Mat disp_vis = cv::Mat::zeros(disparity.size(), CV_8UC1);
    if (valid_values.empty()) {
        return disp_vis;
    }

    std::sort(valid_values.begin(), valid_values.end());
    const size_t lo_idx = static_cast<size_t>(0.02 * static_cast<double>(valid_values.size() - 1));
    const size_t hi_idx = static_cast<size_t>(0.98 * static_cast<double>(valid_values.size() - 1));
    const float lo = valid_values[lo_idx];
    const float hi = valid_values[hi_idx];

    if (hi - lo < 1e-6f) {
        return disp_vis;
    }

    for (int y = 0; y < disparity.rows; ++y) {
        const float* src_row = disparity.ptr<float>(y);
        uint8_t* dst_row = disp_vis.ptr<uint8_t>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float value = src_row[x];
            if (!std::isfinite(value) || value <= 0.0f) {
                dst_row[x] = 0;
                continue;
            }

            const float clamped = std::min(std::max(value, lo), hi);
            const float normalized = (clamped - lo) / (hi - lo);
            dst_row[x] = static_cast<uint8_t>(std::round(normalized * 255.0f));
        }
    }

    return disp_vis;
}

}  // namespace

int main(int argc, char** argv) {
    try {

        // std::string feature_engine_path = "/workspaces/data/20-26-39/feature_runner_fp16.engine";
        // std::string post_engine_path = "/workspaces/data/20-26-39/post_runner_fp16.engine";

        // std::string left_image_path = "/workspaces/data/left.png";
        // std::string right_image_path = "/workspaces/data/right.png";
        // std::string output_path = "/workspaces/data/1_disparity_gen.png";

        std::string feature_engine_path = "/home/hc/weizi/ffs/model/feature_runner_fp16_5060.engine";
        std::string post_engine_path = "/home/hc/weizi/ffs/model/post_runner_fp16_5060.engine";

        std::string left_image_path = "/home/hc/weizi/ffs+fp+sam/ffs/demo_data/left.png";
        std::string right_image_path = "/home/hc/weizi/ffs+fp+sam/ffs/demo_data/right.png";
        std::string output_path = "/home/hc/weizi/ffs+fp+sam/ffs/results/fast_foundation_stereo_test_disparity.png";

        // 1. Configuration (matches your ONNX export)
        const int target_height = 448;
        const int target_width = 640;

        if (argc >= 6) {
            feature_engine_path = argv[1];
            post_engine_path = argv[2];
            left_image_path = argv[3];
            right_image_path = argv[4];
            output_path = argv[5];
        } else if (argc != 1) {
            std::cerr << "Usage: " << argv[0]
                      << " [feature.engine post.engine left.png right.png output_vis.png]"
                      << std::endl;
            return -1;
        }

        std::cout << "[Main] feature_engine_path: " << feature_engine_path << std::endl;
        std::cout << "[Main] post_engine_path: " << post_engine_path << std::endl;
        std::cout << "[Main] left_image_path: " << left_image_path << std::endl;
        std::cout << "[Main] right_image_path: " << right_image_path << std::endl;
        std::cout << "[Main] output_path: " << output_path << std::endl;

        // 2. Initialize Engine
        std::cout << "[Main] Initializing Engine..." << std::endl;
        FastFoundationStereoEstimator estimator(feature_engine_path, post_engine_path, target_height, target_width);

        // 3. Load Images
        cv::Mat leftRaw = cv::imread(left_image_path, cv::IMREAD_UNCHANGED);
        cv::Mat rightRaw = cv::imread(right_image_path, cv::IMREAD_UNCHANGED);
        if (leftRaw.empty() || rightRaw.empty()) {
            std::cerr << "[Main] Error: Could not load images." << std::endl;
            return -1;
        }

        leftRaw = prepareStereoInput(leftRaw, target_width, target_height);
        rightRaw = prepareStereoInput(rightRaw, target_width, target_height);

    // 4. Resize/crop if necessary (The wrapper expects 448x640 input)
    //cv::Mat leftInput, rightInput;
    //cv::resize(leftRaw, leftInput, cv::Size(target_width, target_height));
    //cv::resize(rightRaw, rightInput, cv::Size(target_width, target_height));

    // 5. Inference
        cv::Mat disparity;
    
    // Warmup
        std::cout << "[Main] Warming up..." << std::endl;
        for(int i=0; i<3; ++i) {
            if (!estimator.inference(leftRaw, rightRaw, disparity)) {
                std::cerr << "[Main] Warmup inference failed on iteration " << i << std::endl;
                return -1;
            }
        }

    // Timing
        std::cout << "[Main] Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
    
        if (!estimator.inference(leftRaw, rightRaw, disparity)) {
            std::cerr << "[Main] Inference failed." << std::endl;
            return -1;
        }
    
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[Main] Inference Time: " << duration << " ms" << std::endl;
        printDisparityStats(disparity);


    // 6. Visualization
    // Disparity is usually float. Normalize to 0-255 for display.
    cv::Mat dispVis;
    dispVis = visualizeDisparityRobust(disparity);
    
    // Apply colormap (Jet or Plasma)
    cv::applyColorMap(dispVis, dispVis, cv::COLORMAP_JET);
    
    // Save result
    cv::imwrite(output_path, dispVis);
    
    std::cout << "[Main] Result saved to " << output_path << std::endl;
    
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Main] Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[Main] Unknown exception." << std::endl;
    }

    return 1;
}
