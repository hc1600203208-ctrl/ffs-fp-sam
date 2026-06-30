#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>

#include "depth_confidence.hpp"
#include "estimator/fast_foundation_stereo_estimator.h"
#include "estimator/stereo_estimator.h"
#include "stereo_calibration_utils.hpp"

namespace fs = std::filesystem;

namespace {

enum class OfflineStereoModel {
    FAST_FOUNDATION_STEREO,
};

OfflineStereoModel StringToOfflineStereoModel(const std::string& model) {
    static const std::unordered_map<std::string, OfflineStereoModel> model_map = {
        {"FAST_FOUNDATION_STEREO", OfflineStereoModel::FAST_FOUNDATION_STEREO},
    };

    const auto it = model_map.find(model);
    if (it != model_map.end()) {
        return it->second;
    }

    throw std::invalid_argument("Unknown OfflineStereoModel string: " + model);
}

struct PreparedImage {
    cv::Mat image;
    double scale = 1.0;
    int pad_left = 0;
    int pad_top = 0;
    int resized_width = 0;
    int resized_height = 0;
};

struct CameraIntrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

PreparedImage PrepareStereoInput(const cv::Mat& input, int target_width, int target_height) {
    PreparedImage result;
    if (input.empty()) {
        return result;
    }

    cv::Mat color;
    if (input.channels() == 1) {
        cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
    } else if (input.channels() == 3) {
        color = input.clone();
    } else if (input.channels() == 4) {
        cv::cvtColor(input, color, cv::COLOR_BGRA2BGR);
    } else {
        throw std::runtime_error("Unsupported image channel count.");
    }

    result.scale = std::min(
        static_cast<double>(target_width) / static_cast<double>(color.cols),
        static_cast<double>(target_height) / static_cast<double>(color.rows));

    const int resized_width = std::max(1, static_cast<int>(std::round(color.cols * result.scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(color.rows * result.scale)));
    result.resized_width = resized_width;
    result.resized_height = resized_height;

    cv::Mat resized;
    cv::resize(color, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    const int pad_width = target_width - resized_width;
    const int pad_height = target_height - resized_height;
    result.pad_left = pad_width / 2;
    const int pad_right = pad_width - result.pad_left;
    result.pad_top = pad_height / 2;
    const int pad_bottom = pad_height - result.pad_top;

    cv::copyMakeBorder(
        resized,
        result.image,
        result.pad_top,
        pad_bottom,
        result.pad_left,
        pad_right,
        cv::BORDER_REPLICATE);

    return result;
}

double ComputeRectifiedBaselineMeters(const StereoRectificationMaps& rectification_maps) {
    if (rectification_maps.p2.empty()) {
        return 0.0;
    }

    const double fx = rectification_maps.p2.at<double>(0, 0);
    if (std::abs(fx) <= 1e-9) {
        return 0.0;
    }

    return std::abs(rectification_maps.p2.at<double>(0, 3) / fx);
}

CameraIntrinsics ComputePreparedIntrinsics(
    const StereoRectificationMaps& rectification_maps,
    const PreparedImage& prepared) {
    CameraIntrinsics intrinsics;
    if (rectification_maps.p1.empty()) {
        return intrinsics;
    }

    intrinsics.fx = rectification_maps.p1.at<double>(0, 0) * prepared.scale;
    intrinsics.fy = rectification_maps.p1.at<double>(1, 1) * prepared.scale;
    intrinsics.cx = rectification_maps.p1.at<double>(0, 2) * prepared.scale +
                    static_cast<double>(prepared.pad_left);
    intrinsics.cy = rectification_maps.p1.at<double>(1, 2) * prepared.scale +
                    static_cast<double>(prepared.pad_top);
    return intrinsics;
}

cv::Mat DisparityToDepthMeters(
    const cv::Mat& disparity,
    float fx,
    float baseline,
    float min_depth_meters,
    float max_depth_meters) {
    cv::Mat depth = cv::Mat::zeros(disparity.size(), CV_32FC1);
    const float numerator = fx * baseline;

    for (int y = 0; y < disparity.rows; ++y) {
        const float* disparity_row = disparity.ptr<float>(y);
        float* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float disparity_value = disparity_row[x];
            if (!std::isfinite(disparity_value) || disparity_value <= 0.0f) {
                depth_row[x] = 0.0f;
                continue;
            }

            const float depth_value = numerator / disparity_value;
            if (!std::isfinite(depth_value) ||
                depth_value < min_depth_meters ||
                depth_value > max_depth_meters) {
                depth_row[x] = 0.0f;
                continue;
            }

            depth_row[x] = depth_value;
        }
    }

    return depth;
}

cv::Mat ConvertDepthMetersToUint16Millimeters(const cv::Mat& depth_meters, double depth_scale) {
    cv::Mat depth_u16(depth_meters.size(), CV_16UC1, cv::Scalar(0));
    const double max_output_value = static_cast<double>(std::numeric_limits<uint16_t>::max());

    for (int y = 0; y < depth_meters.rows; ++y) {
        const float* src_row = depth_meters.ptr<float>(y);
        uint16_t* dst_row = depth_u16.ptr<uint16_t>(y);
        for (int x = 0; x < depth_meters.cols; ++x) {
            const float value = src_row[x];
            if (!std::isfinite(value) || value <= 0.0f) {
                dst_row[x] = 0;
                continue;
            }

            const double scaled = static_cast<double>(value) * depth_scale;
            if (scaled <= 0.0) {
                dst_row[x] = 0;
                continue;
            }

            dst_row[x] = static_cast<uint16_t>(std::min(std::round(scaled), max_output_value));
        }
    }

    return depth_u16;
}

cv::Mat RestoreToRectifiedResolution(
    const cv::Mat& image,
    const PreparedImage& prepared,
    const cv::Size& rectified_size,
    int interpolation) {
    if (image.empty() || rectified_size.width <= 0 || rectified_size.height <= 0) {
        return {};
    }

    const cv::Rect valid_roi(
        prepared.pad_left,
        prepared.pad_top,
        std::min(prepared.resized_width, image.cols - prepared.pad_left),
        std::min(prepared.resized_height, image.rows - prepared.pad_top));
    if (valid_roi.width <= 0 || valid_roi.height <= 0) {
        return {};
    }

    const cv::Mat cropped = image(valid_roi);
    cv::Mat restored;
    cv::resize(cropped, restored, rectified_size, 0, 0, interpolation);
    return restored;
}

cv::Mat VisualizeDepthMeters(
    const cv::Mat& depth_meters,
    float min_depth_meters,
    float max_depth_meters) {
    if (depth_meters.empty()) {
        return {};
    }

    const float depth_range = std::max(max_depth_meters - min_depth_meters, 1e-6f);
    cv::Mat depth_vis(depth_meters.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat valid_mask = cv::Mat::zeros(depth_meters.size(), CV_8UC1);

    for (int y = 0; y < depth_meters.rows; ++y) {
        const float* depth_row = depth_meters.ptr<float>(y);
        uint8_t* vis_row = depth_vis.ptr<uint8_t>(y);
        uint8_t* mask_row = valid_mask.ptr<uint8_t>(y);
        for (int x = 0; x < depth_meters.cols; ++x) {
            const float depth_value = depth_row[x];
            if (!std::isfinite(depth_value) ||
                depth_value < min_depth_meters ||
                depth_value > max_depth_meters) {
                vis_row[x] = 0;
                mask_row[x] = 0;
                continue;
            }

            const float normalized =
                std::clamp((depth_value - min_depth_meters) / depth_range, 0.0f, 1.0f);
            vis_row[x] = static_cast<uint8_t>(std::round((1.0f - normalized) * 255.0f));
            mask_row[x] = 255;
        }
    }

    cv::Mat color_vis;
    cv::applyColorMap(depth_vis, color_vis, cv::COLORMAP_JET);
    color_vis.setTo(cv::Scalar(0, 0, 0), ~valid_mask);
    return color_vis;
}

fs::path ResolveOutputDirectory(const fs::path& dataset_root, const std::string& configured_path) {
    const fs::path path(configured_path);
    if (path.is_absolute()) {
        return path;
    }
    return dataset_root / path;
}

fs::path MakePngOutputPath(const fs::path& output_dir, const fs::path& input_path) {
    return output_dir / (input_path.stem().string() + ".png");
}

std::vector<std::string> CollectMatchedImageNames(
    const fs::path& left_dir,
    const fs::path& right_dir) {
    std::set<std::string> matched_names;

    if (!fs::exists(left_dir) || !fs::is_directory(left_dir)) {
        throw std::runtime_error("Left image directory does not exist: " + left_dir.string());
    }
    if (!fs::exists(right_dir) || !fs::is_directory(right_dir)) {
        throw std::runtime_error("Right image directory does not exist: " + right_dir.string());
    }

    for (const auto& entry : fs::directory_iterator(left_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path filename = entry.path().filename();
        if (fs::exists(right_dir / filename)) {
            matched_names.insert(filename.string());
        }
    }

    return std::vector<std::string>(matched_names.begin(), matched_names.end());
}

}  // namespace

class OfflineStereoDepthNode : public rclcpp::Node {
   public:
    OfflineStereoDepthNode() : rclcpp::Node("offline_stereo_depth_node") {
        this->declare_parameter("model_input_height", 448);
        this->declare_parameter("model_input_width", 640);
        this->declare_parameter("input_image_height", 1080);
        this->declare_parameter("input_image_width", 1920);
        this->declare_parameter("engine_file_path", std::vector<std::string>{""});
        this->declare_parameter("model_type", "FAST_FOUNDATION_STEREO");
        this->declare_parameter("caminfo_path", std::string("/home/hc/weizi/ffs/caminfo.txt"));
        this->declare_parameter("dataset_root", std::string("/home/hc/weizi/ffs/jr"));
        this->declare_parameter("left_subdir", std::string("rgb"));
        this->declare_parameter("right_subdir", std::string("camera2"));
        this->declare_parameter("output_subdir", std::string("depth"));
        this->declare_parameter("raw_depth_subdir", std::string("depth_raw"));
        this->declare_parameter("vis_subdir", std::string("vis"));
        this->declare_parameter("min_depth_meters", 0.1);
        this->declare_parameter("max_depth_meters", 100.0);
        this->declare_parameter("depth_scale", 1000.0);
        this->declare_parameter("save_input_resolution", true);
        this->declare_parameter("patch_radius", confidence_params_.patch_radius);
        this->declare_parameter("texture_ksize", confidence_params_.texture_ksize);
        this->declare_parameter("sigma_photo", confidence_params_.sigma_photo);
        this->declare_parameter("sigma_e", confidence_params_.sigma_e);
        this->declare_parameter("sigma_t", confidence_params_.sigma_t);
        this->declare_parameter("alpha_edge", confidence_params_.alpha_edge);
        this->declare_parameter("w_photo", confidence_params_.w_photo);
        this->declare_parameter("w_tex", confidence_params_.w_tex);
        this->declare_parameter("w_grad", confidence_params_.w_grad);
        this->declare_parameter("w_tmp", confidence_params_.w_tmp);
        this->declare_parameter("conf_threshold", confidence_params_.conf_threshold);
        this->declare_parameter("weight_gamma", confidence_params_.weight_gamma);

        model_input_height_ = this->get_parameter("model_input_height").as_int();
        model_input_width_ = this->get_parameter("model_input_width").as_int();
        input_image_height_ = this->get_parameter("input_image_height").as_int();
        input_image_width_ = this->get_parameter("input_image_width").as_int();
        caminfo_path_ = this->get_parameter("caminfo_path").as_string();
        dataset_root_ = fs::path(this->get_parameter("dataset_root").as_string());
        left_dir_ = dataset_root_ / this->get_parameter("left_subdir").as_string();
        right_dir_ = dataset_root_ / this->get_parameter("right_subdir").as_string();
        output_dir_ = ResolveOutputDirectory(
            dataset_root_,
            this->get_parameter("output_subdir").as_string());
        raw_depth_dir_ = ResolveOutputDirectory(
            dataset_root_,
            this->get_parameter("raw_depth_subdir").as_string());
        vis_dir_ = ResolveOutputDirectory(
            dataset_root_,
            this->get_parameter("vis_subdir").as_string());
        min_depth_meters_ = static_cast<float>(this->get_parameter("min_depth_meters").as_double());
        max_depth_meters_ = static_cast<float>(this->get_parameter("max_depth_meters").as_double());
        depth_scale_ = this->get_parameter("depth_scale").as_double();
        save_input_resolution_ = this->get_parameter("save_input_resolution").as_bool();
        confidence_params_.patch_radius = this->get_parameter("patch_radius").as_int();
        confidence_params_.texture_ksize = this->get_parameter("texture_ksize").as_int();
        confidence_params_.sigma_photo = static_cast<float>(this->get_parameter("sigma_photo").as_double());
        confidence_params_.sigma_e = static_cast<float>(this->get_parameter("sigma_e").as_double());
        confidence_params_.sigma_t = static_cast<float>(this->get_parameter("sigma_t").as_double());
        confidence_params_.alpha_edge = static_cast<float>(this->get_parameter("alpha_edge").as_double());
        confidence_params_.w_photo = static_cast<float>(this->get_parameter("w_photo").as_double());
        confidence_params_.w_tex = static_cast<float>(this->get_parameter("w_tex").as_double());
        confidence_params_.w_grad = static_cast<float>(this->get_parameter("w_grad").as_double());
        confidence_params_.w_tmp = static_cast<float>(this->get_parameter("w_tmp").as_double());
        confidence_params_.conf_threshold = static_cast<float>(this->get_parameter("conf_threshold").as_double());
        confidence_params_.weight_gamma = static_cast<float>(this->get_parameter("weight_gamma").as_double());

        const std::vector<std::string> engine_file_path =
            this->get_parameter("engine_file_path").as_string_array();
        const std::string model_type = this->get_parameter("model_type").as_string();

        std::string calibration_error;
        if (!LoadStereoCalibrationFromTxt(caminfo_path_, &calibration_, &calibration_error)) {
            throw std::runtime_error(calibration_error);
        }

        const OfflineStereoModel stereo_model = StringToOfflineStereoModel(model_type);
        switch (stereo_model) {
            case OfflineStereoModel::FAST_FOUNDATION_STEREO:
                if (engine_file_path.size() != 2) {
                    throw std::runtime_error("FAST_FOUNDATION_STEREO expects exactly 2 engine paths.");
                }
                estimator_ = std::make_unique<FastFoundationStereoEstimator>(
                    engine_file_path[0],
                    engine_file_path[1],
                    model_input_height_,
                    model_input_width_);
                break;
            default:
                throw std::runtime_error("Unsupported offline stereo model type: " + model_type);
        }

        run_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&OfflineStereoDepthNode::runDataset, this));
    }

   private:
    bool ensureRectificationMaps(const cv::Size& image_size) {
        if (rectification_ready_ && rectification_maps_.image_size == image_size) {
            return true;
        }

        std::string rectification_error;
        if (!ComputeStereoRectificationMaps(
                calibration_,
                image_size,
                &rectification_maps_,
                &rectification_error)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to compute rectification maps: %s", rectification_error.c_str());
            return false;
        }

        rectification_ready_ = true;
        RCLCPP_INFO(
            this->get_logger(),
            "Rectification maps prepared for %dx%d images.",
            image_size.width,
            image_size.height);
        return true;
    }

    bool preprocess(
        const cv::Mat& left_image,
        const cv::Mat& right_image,
        cv::Mat* prepared_left_image,
        cv::Mat* prepared_right_image,
        PreparedImage* prepared_left) {
        if (prepared_left_image == nullptr || prepared_right_image == nullptr || prepared_left == nullptr) {
            return false;
        }

        if (left_image.empty() || right_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Input stereo image is empty.");
            return false;
        }

        if (left_image.size() != right_image.size()) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Stereo image sizes do not match: %dx%d vs %dx%d.",
                left_image.cols,
                left_image.rows,
                right_image.cols,
                right_image.rows);
            return false;
        }

        if (!ensureRectificationMaps(left_image.size())) {
            return false;
        }

        cv::Mat left_rectified;
        cv::Mat right_rectified;
        cv::remap(
            left_image,
            left_rectified,
            rectification_maps_.left_map_x,
            rectification_maps_.left_map_y,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT);
        cv::remap(
            right_image,
            right_rectified,
            rectification_maps_.right_map_x,
            rectification_maps_.right_map_y,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT);

        if (left_rectified.rows != input_image_height_ || left_rectified.cols != input_image_width_) {
            RCLCPP_WARN_ONCE(
                this->get_logger(),
                "Rectified image size %dx%d does not match configured input_image_%dx%d.",
                left_rectified.cols,
                left_rectified.rows,
                input_image_width_,
                input_image_height_);
        }

        *prepared_left = PrepareStereoInput(left_rectified, model_input_width_, model_input_height_);
        PreparedImage prepared_right = PrepareStereoInput(right_rectified, model_input_width_, model_input_height_);

        if (prepared_left->image.empty() || prepared_right.image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to prepare stereo input images.");
            return false;
        }

        *prepared_left_image = prepared_left->image;
        *prepared_right_image = prepared_right.image;
        return true;
    }

    bool processSinglePair(
        const fs::path& left_path,
        const fs::path& right_path,
        const fs::path& raw_depth_output_path,
        const fs::path& output_path,
        const fs::path& vis_output_path) {
        const cv::Mat left_image = cv::imread(left_path.string(), cv::IMREAD_COLOR);
        const cv::Mat right_image = cv::imread(right_path.string(), cv::IMREAD_COLOR);

        if (left_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load left image: %s", left_path.c_str());
            return false;
        }
        if (right_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load right image: %s", right_path.c_str());
            return false;
        }

        cv::Mat prepared_left_image;
        cv::Mat prepared_right_image;
        PreparedImage prepared_left;
        if (!preprocess(
                left_image,
                right_image,
                &prepared_left_image,
                &prepared_right_image,
                &prepared_left)) {
            return false;
        }

        const auto start = std::chrono::high_resolution_clock::now();

        cv::Mat disparity;
        estimator_->inference(prepared_left_image, prepared_right_image, disparity);

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        const double baseline_meters = ComputeRectifiedBaselineMeters(rectification_maps_);
        const CameraIntrinsics intrinsics = ComputePreparedIntrinsics(rectification_maps_, prepared_left);
        if (baseline_meters <= 0.0 || intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Invalid rectified intrinsics for depth conversion: fx=%.6f fy=%.6f baseline=%.6f.",
                intrinsics.fx,
                intrinsics.fy,
                baseline_meters);
            return false;
        }

        const cv::Mat depth_meters = DisparityToDepthMeters(
            disparity,
            static_cast<float>(intrinsics.fx),
            static_cast<float>(baseline_meters),
            min_depth_meters_,
            max_depth_meters_);
        const cv::Mat raw_depth_rectified_full = RestoreToRectifiedResolution(
            depth_meters,
            prepared_left,
            left_image.size(),
            cv::INTER_NEAREST);
        if (raw_depth_rectified_full.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to restore raw depth to rectified resolution.");
            return false;
        }

        cv::Mat raw_depth_to_save = raw_depth_rectified_full;
        if (save_input_resolution_) {
            raw_depth_to_save = AlignRectifiedDepthToOriginalLeft(
                raw_depth_rectified_full,
                rectification_maps_,
                cv::INTER_NEAREST);
            if (raw_depth_to_save.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to align raw depth back to the original left image.");
                return false;
            }
        }
        const cv::Mat confidence_map = depth_confidence::compute_confidence_map(
            prepared_left_image,
            prepared_right_image,
            depth_meters,
            static_cast<float>(intrinsics.fx),
            static_cast<float>(intrinsics.fy),
            static_cast<float>(intrinsics.cx),
            static_cast<float>(intrinsics.cy),
            static_cast<float>(baseline_meters),
            previous_depth_,
            confidence_params_);
        const cv::Mat filtered_depth_meters = depth_confidence::filter_depth_with_confidence(
            depth_meters,
            confidence_map,
            confidence_params_.conf_threshold);
        previous_depth_ = depth_meters.clone();

        const cv::Mat depth_rectified_full = RestoreToRectifiedResolution(
            filtered_depth_meters,
            prepared_left,
            left_image.size(),
            cv::INTER_NEAREST);
        if (depth_rectified_full.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to restore depth to rectified resolution.");
            return false;
        }

        cv::Mat depth_to_save = depth_rectified_full;
        if (save_input_resolution_) {
            depth_to_save = AlignRectifiedDepthToOriginalLeft(
                depth_rectified_full,
                rectification_maps_,
                cv::INTER_NEAREST);
            if (depth_to_save.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to align depth back to the original left image.");
                return false;
            }
        }

        const cv::Mat depth_u16 = ConvertDepthMetersToUint16Millimeters(depth_to_save, depth_scale_);
        const cv::Mat raw_depth_u16 =
            ConvertDepthMetersToUint16Millimeters(raw_depth_to_save, depth_scale_);
        const cv::Mat depth_vis = VisualizeDepthMeters(
            depth_to_save,
            min_depth_meters_,
            max_depth_meters_);
        if (depth_vis.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to visualize filtered depth image.");
            return false;
        }

        if (!cv::imwrite(raw_depth_output_path.string(), raw_depth_u16)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save raw depth image: %s", raw_depth_output_path.c_str());
            return false;
        }
        if (!cv::imwrite(output_path.string(), depth_u16)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save depth image: %s", output_path.c_str());
            return false;
        }
        if (!cv::imwrite(vis_output_path.string(), depth_vis)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save depth visualization: %s", vis_output_path.c_str());
            return false;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Processed %s -> raw:%s filtered:%s vis:%s in %ld ms.",
            left_path.filename().c_str(),
            raw_depth_output_path.filename().c_str(),
            output_path.filename().c_str(),
            vis_output_path.filename().c_str(),
            duration_ms);
        return true;
    }

    void runDataset() {
        run_timer_->cancel();

        try {
            fs::create_directories(output_dir_);
            fs::create_directories(raw_depth_dir_);
            fs::create_directories(vis_dir_);
            const std::vector<std::string> filenames = CollectMatchedImageNames(left_dir_, right_dir_);
            if (filenames.empty()) {
                throw std::runtime_error(
                    "No matched stereo image pairs found under " + left_dir_.string() +
                    " and " + right_dir_.string() + ".");
            }

            RCLCPP_INFO(
                this->get_logger(),
                "Starting offline depth estimation for %zu stereo pairs. Raw depth output: %s, filtered depth output: %s, visualization output: %s",
                filenames.size(),
                raw_depth_dir_.c_str(),
                output_dir_.c_str(),
                vis_dir_.c_str());

            std::size_t success_count = 0;
            for (const std::string& filename : filenames) {
                const fs::path left_path = left_dir_ / filename;
                const fs::path right_path = right_dir_ / filename;
                const fs::path raw_depth_output_path = MakePngOutputPath(raw_depth_dir_, left_path);
                const fs::path output_path = MakePngOutputPath(output_dir_, left_path);
                const fs::path vis_output_path = MakePngOutputPath(vis_dir_, left_path);

                if (processSinglePair(
                        left_path,
                        right_path,
                        raw_depth_output_path,
                        output_path,
                        vis_output_path)) {
                    ++success_count;
                }
            }

            RCLCPP_INFO(
                this->get_logger(),
                "Offline depth estimation completed: %zu/%zu pairs succeeded.",
                success_count,
                filenames.size());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Offline depth estimation failed: %s", e.what());
        }

        rclcpp::shutdown();
    }

    int model_input_height_ = 448;
    int model_input_width_ = 640;
    int input_image_height_ = 1080;
    int input_image_width_ = 1920;

    std::string caminfo_path_;
    fs::path dataset_root_;
    fs::path left_dir_;
    fs::path right_dir_;
    fs::path output_dir_;
    fs::path raw_depth_dir_;
    fs::path vis_dir_;

    float min_depth_meters_ = 0.1f;
    float max_depth_meters_ = 100.0f;
    double depth_scale_ = 1000.0;
    bool save_input_resolution_ = true;

    StereoCalibration calibration_;
    StereoRectificationMaps rectification_maps_;
    bool rectification_ready_ = false;

    std::unique_ptr<StereoEstimator> estimator_;
    rclcpp::TimerBase::SharedPtr run_timer_;
    depth_confidence::ConfidenceParameters confidence_params_;
    cv::Mat previous_depth_;
};

void signal_handler(int sig) {
    RCLCPP_WARN(rclcpp::get_logger("OfflineStereoDepthNode"), "catch sig %d", sig);
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    std::signal(SIGINT, signal_handler);

    try {
        rclcpp::spin(std::make_shared<OfflineStereoDepthNode>());
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& e) {
        RCLCPP_FATAL(
            rclcpp::get_logger("OfflineStereoDepthNode"),
            "Unhandled exception: %s",
            e.what());
    } catch (...) {
        RCLCPP_FATAL(
            rclcpp::get_logger("OfflineStereoDepthNode"),
            "Unhandled non-standard exception.");
    }

    rclcpp::shutdown();
    return 1;
}
