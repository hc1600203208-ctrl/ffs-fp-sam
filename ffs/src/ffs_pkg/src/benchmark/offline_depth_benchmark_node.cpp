#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include "../depth_confidence.hpp"
#include "../estimator/fast_foundation_stereo_estimator.h"
#include "../estimator/stereo_estimator.h"
#include "../stereo_calibration_utils.hpp"

namespace fs = std::filesystem;

namespace {

enum class OfflineStereoModel {
    FAST_FOUNDATION_STEREO,
};

OfflineStereoModel StringToOfflineStereoModel(const std::string& model) {
    if (model == "FAST_FOUNDATION_STEREO") {
        return OfflineStereoModel::FAST_FOUNDATION_STEREO;
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

struct ConfidenceTerm {
    const cv::Mat* map = nullptr;
    float weight = 0.0f;
};

struct ConfidenceComponents {
    cv::Mat photo;
    cv::Mat texture;
    cv::Mat gradient;
    cv::Mat temporal;
    bool temporal_available = false;
};

std::string Trim(const std::string& input) {
    const auto first = input.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = input.find_last_not_of(" \t\n\r");
    return input.substr(first, last - first + 1);
}

std::vector<float> ParseThresholdList(std::string text) {
    for (char& c : text) {
        if (c == '[' || c == ']' || c == ';') {
            c = ',';
        }
    }

    std::vector<float> thresholds;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = Trim(token);
        if (token.empty()) {
            continue;
        }
        thresholds.push_back(std::stof(token));
    }
    return thresholds;
}

std::string FormatThresholdDirectory(float threshold) {
    std::ostringstream oss;
    oss << "threshold_" << std::fixed << std::setprecision(2) << threshold;
    std::string value = oss.str();
    std::replace(value.begin(), value.end(), '.', '_');
    std::replace(value.begin(), value.end(), '-', 'm');
    return value;
}

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

    result.resized_width = std::max(1, static_cast<int>(std::round(color.cols * result.scale)));
    result.resized_height = std::max(1, static_cast<int>(std::round(color.rows * result.scale)));

    cv::Mat resized;
    cv::resize(
        color,
        resized,
        cv::Size(result.resized_width, result.resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    const int pad_width = target_width - result.resized_width;
    const int pad_height = target_height - result.resized_height;
    result.pad_left = pad_width / 2;
    result.pad_top = pad_height / 2;
    const int pad_right = pad_width - result.pad_left;
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
    cv::resize(cropped, restored, rectified_size, 0.0, 0.0, interpolation);
    return restored;
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

cv::Mat CreateValidDepthMask(const cv::Mat& depth) {
    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8UC1);
    for (int y = 0; y < depth.rows; ++y) {
        const float* depth_row = depth.ptr<float>(y);
        uint8_t* mask_row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < depth.cols; ++x) {
            const float value = depth_row[x];
            mask_row[x] = (std::isfinite(value) && value > 0.0f) ? 255 : 0;
        }
    }
    return mask;
}

cv::Mat ClipFloat01(const cv::Mat& input) {
    cv::Mat output;
    cv::max(input, 0.0f, output);
    cv::min(output, 1.0f, output);
    output.convertTo(output, CV_32F);
    return output;
}

cv::Mat CombineConfidenceTerms(
    const cv::Mat& depth,
    const std::vector<ConfidenceTerm>& terms,
    float eps) {
    if (depth.empty() || terms.empty()) {
        return {};
    }

    float weight_sum = 0.0f;
    for (const ConfidenceTerm& term : terms) {
        if (term.map == nullptr || term.map->empty()) {
            return {};
        }
        if (term.map->size() != depth.size()) {
            return {};
        }
        weight_sum += std::max(term.weight, 0.0f);
    }

    if (weight_sum <= eps) {
        return {};
    }

    const cv::Mat depth_mask = CreateValidDepthMask(depth);
    cv::Mat confidence(depth.size(), CV_32FC1, cv::Scalar(1.0f));
    const float safe_eps = std::max(eps, 1e-6f);

    for (int y = 0; y < depth.rows; ++y) {
        const uint8_t* mask_row = depth_mask.ptr<uint8_t>(y);
        float* conf_row = confidence.ptr<float>(y);
        for (int x = 0; x < depth.cols; ++x) {
            if (mask_row[x] == 0) {
                conf_row[x] = 0.0f;
                continue;
            }

            float combined = 1.0f;
            for (const ConfidenceTerm& term : terms) {
                const float* term_row = term.map->ptr<float>(y);
                float value = term_row[x];
                if (!std::isfinite(value)) {
                    value = 0.0f;
                }
                value = std::clamp(value, safe_eps, 1.0f);
                combined *= std::pow(value, std::max(term.weight, 0.0f) / weight_sum);
            }
            conf_row[x] = combined;
        }
    }

    cv::GaussianBlur(confidence, confidence, cv::Size(3, 3), 0.8, 0.8, cv::BORDER_REFLECT101);
    confidence.setTo(0.0f, ~depth_mask);
    return ClipFloat01(confidence);
}

cv::Mat ApplyMedianDepthFilter(const cv::Mat& depth, int kernel_size) {
    if (depth.empty()) {
        return {};
    }

    if (kernel_size <= 1) {
        return depth.clone();
    }
    if (kernel_size % 2 == 0) {
        ++kernel_size;
    }
    kernel_size = std::clamp(kernel_size, 3, 5);

    cv::Mat filtered;
    cv::medianBlur(depth, filtered, kernel_size);

    cv::Mat output = cv::Mat::zeros(depth.size(), depth.type());
    filtered.copyTo(output, CreateValidDepthMask(depth));
    return output;
}

cv::Mat ApplyBilateralDepthFilter(
    const cv::Mat& depth,
    int diameter,
    double sigma_color,
    double sigma_space) {
    if (depth.empty()) {
        return {};
    }

    cv::Mat filtered;
    cv::bilateralFilter(
        depth,
        filtered,
        std::max(diameter, 1),
        sigma_color,
        sigma_space);

    cv::Mat output = cv::Mat::zeros(depth.size(), depth.type());
    filtered.copyTo(output, CreateValidDepthMask(depth));
    return output;
}

double ComputeValidRatio(const cv::Mat& depth) {
    if (depth.empty()) {
        return 0.0;
    }

    const cv::Mat valid_mask = CreateValidDepthMask(depth);
    return static_cast<double>(cv::countNonZero(valid_mask)) /
           static_cast<double>(depth.rows * depth.cols);
}

}  // namespace

class OfflineDepthBenchmarkNode : public rclcpp::Node {
   public:
    OfflineDepthBenchmarkNode() : rclcpp::Node("offline_depth_benchmark_node") {
        declareParameters();
        loadParameters();
        loadCalibration();
        createEstimator();

        run_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&OfflineDepthBenchmarkNode::runDataset, this));
    }

   private:
    void declareParameters() {
        this->declare_parameter("model_input_height", 448);
        this->declare_parameter("model_input_width", 640);
        this->declare_parameter("input_image_height", 1080);
        this->declare_parameter("input_image_width", 1920);
        this->declare_parameter("engine_file_path", std::vector<std::string>{""});
        this->declare_parameter("model_type", "FAST_FOUNDATION_STEREO");
        this->declare_parameter("caminfo_path", std::string("/home/hc/weizi/dataset/jrnew-blue/caminfo.txt"));
        this->declare_parameter("dataset_root", std::string("/home/hc/weizi/dataset/jrnew-blue"));
        this->declare_parameter("left_subdir", std::string("rgb"));
        this->declare_parameter("right_subdir", std::string("camera2"));
        this->declare_parameter("output_root", std::string("depth_benchmark"));
        this->declare_parameter("min_depth_meters", 0.1);
        this->declare_parameter("max_depth_meters", 100.0);
        this->declare_parameter("depth_scale", 1000.0);
        this->declare_parameter("save_input_resolution", true);
        this->declare_parameter("max_pairs", 0);
        this->declare_parameter("median_kernel_size", 5);
        this->declare_parameter("bilateral_d", 5);
        this->declare_parameter("bilateral_sigma_color", 0.05);
        this->declare_parameter("bilateral_sigma_space", 5.0);
        this->declare_parameter("use_temporal_confidence", true);
        this->declare_parameter("threshold_sweep", std::string("0.20,0.30,0.35,0.40,0.50"));
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
    }

    void loadParameters() {
        model_input_height_ = this->get_parameter("model_input_height").as_int();
        model_input_width_ = this->get_parameter("model_input_width").as_int();
        input_image_height_ = this->get_parameter("input_image_height").as_int();
        input_image_width_ = this->get_parameter("input_image_width").as_int();
        engine_file_path_ = this->get_parameter("engine_file_path").as_string_array();
        model_type_ = this->get_parameter("model_type").as_string();
        caminfo_path_ = this->get_parameter("caminfo_path").as_string();

        dataset_root_ = fs::path(this->get_parameter("dataset_root").as_string());
        left_dir_ = dataset_root_ / this->get_parameter("left_subdir").as_string();
        right_dir_ = dataset_root_ / this->get_parameter("right_subdir").as_string();
        output_root_ = ResolveOutputDirectory(
            dataset_root_,
            this->get_parameter("output_root").as_string());

        min_depth_meters_ = static_cast<float>(this->get_parameter("min_depth_meters").as_double());
        max_depth_meters_ = static_cast<float>(this->get_parameter("max_depth_meters").as_double());
        depth_scale_ = this->get_parameter("depth_scale").as_double();
        save_input_resolution_ = this->get_parameter("save_input_resolution").as_bool();
        max_pairs_ = this->get_parameter("max_pairs").as_int();
        median_kernel_size_ = this->get_parameter("median_kernel_size").as_int();
        bilateral_d_ = this->get_parameter("bilateral_d").as_int();
        bilateral_sigma_color_ = this->get_parameter("bilateral_sigma_color").as_double();
        bilateral_sigma_space_ = this->get_parameter("bilateral_sigma_space").as_double();
        use_temporal_confidence_ = this->get_parameter("use_temporal_confidence").as_bool();

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

        threshold_sweep_ = ParseThresholdList(this->get_parameter("threshold_sweep").as_string());
        if (threshold_sweep_.empty()) {
            threshold_sweep_ = {0.20f, 0.30f, 0.35f, 0.40f, 0.50f};
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Offline benchmark output root: %s",
            output_root_.string().c_str());
    }

    void loadCalibration() {
        std::string calibration_error;
        if (!LoadStereoCalibrationFromTxt(caminfo_path_, &calibration_, &calibration_error)) {
            throw std::runtime_error(calibration_error);
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Loaded stereo calibration from %s with baseline %.6f m.",
            caminfo_path_.c_str(),
            calibration_.baseline);
    }

    void createEstimator() {
        const OfflineStereoModel stereo_model = StringToOfflineStereoModel(model_type_);
        switch (stereo_model) {
            case OfflineStereoModel::FAST_FOUNDATION_STEREO:
                if (engine_file_path_.size() != 2) {
                    throw std::runtime_error("FAST_FOUNDATION_STEREO expects exactly 2 engine paths.");
                }
                RCLCPP_INFO(
                    this->get_logger(),
                    "feature_model_file: %s",
                    engine_file_path_[0].c_str());
                RCLCPP_INFO(
                    this->get_logger(),
                    "post_model_file: %s",
                    engine_file_path_[1].c_str());

                estimator_ = std::make_unique<FastFoundationStereoEstimator>(
                    engine_file_path_[0],
                    engine_file_path_[1],
                    model_input_height_,
                    model_input_width_);
                break;
            default:
                throw std::runtime_error("Unsupported offline stereo model type: " + model_type_);
        }
    }

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
            RCLCPP_ERROR(
                this->get_logger(),
                "Failed to compute rectification maps: %s",
                rectification_error.c_str());
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
        PreparedImage* prepared_left,
        cv::Size* rectified_size) {
        if (prepared_left_image == nullptr ||
            prepared_right_image == nullptr ||
            prepared_left == nullptr ||
            rectified_size == nullptr) {
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
        const PreparedImage prepared_right =
            PrepareStereoInput(right_rectified, model_input_width_, model_input_height_);

        if (prepared_left->image.empty() || prepared_right.image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to prepare stereo input images.");
            return false;
        }

        *prepared_left_image = prepared_left->image;
        *prepared_right_image = prepared_right.image;
        *rectified_size = left_rectified.size();
        return true;
    }

    ConfidenceComponents computeConfidenceComponents(
        const cv::Mat& prepared_left_image,
        const cv::Mat& prepared_right_image,
        const cv::Mat& depth_model,
        const CameraIntrinsics& intrinsics,
        float baseline_meters) {
        ConfidenceComponents components;
        components.photo = depth_confidence::compute_photometric_confidence(
            prepared_left_image,
            prepared_right_image,
            depth_model,
            static_cast<float>(intrinsics.fx),
            baseline_meters,
            confidence_params_.patch_radius,
            confidence_params_.sigma_photo);
        components.texture = depth_confidence::compute_texture_confidence(
            prepared_left_image,
            confidence_params_.texture_ksize);
        components.gradient = depth_confidence::compute_depth_edge_confidence(
            depth_model,
            prepared_left_image,
            confidence_params_.alpha_edge,
            confidence_params_.sigma_e);

        components.temporal_available =
            use_temporal_confidence_ &&
            !previous_depth_.empty() &&
            previous_depth_.size() == depth_model.size() &&
            previous_depth_.type() == depth_model.type();

        if (components.temporal_available) {
            components.temporal = depth_confidence::compute_temporal_confidence_simple(
                depth_model,
                previous_depth_,
                confidence_params_.sigma_t);
        }

        return components;
    }

    bool saveDepthResult(
        const cv::Mat& depth_model,
        const PreparedImage& prepared_left,
        const cv::Size& rectified_size,
        const fs::path& output_path) {
        const cv::Mat depth_rectified_full = RestoreToRectifiedResolution(
            depth_model,
            prepared_left,
            rectified_size,
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
        if (!cv::imwrite(output_path.string(), depth_u16)) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Failed to save depth image: %s",
                output_path.string().c_str());
            return false;
        }
        return true;
    }

    fs::path groupDirectory(const std::string& group_name) const {
        return output_root_ / group_name;
    }

    fs::path thresholdDirectory(float threshold) const {
        return output_root_ / "confidence_full_threshold_sweep" / FormatThresholdDirectory(threshold);
    }

    bool saveGroup(
        const std::string& group_name,
        const fs::path& left_path,
        const cv::Mat& depth_model,
        const PreparedImage& prepared_left,
        const cv::Size& rectified_size) {
        return saveDepthResult(
            depth_model,
            prepared_left,
            rectified_size,
            MakePngOutputPath(groupDirectory(group_name), left_path));
    }

    bool saveThresholdGroup(
        float threshold,
        const fs::path& left_path,
        const cv::Mat& depth_model,
        const PreparedImage& prepared_left,
        const cv::Size& rectified_size) {
        return saveDepthResult(
            depth_model,
            prepared_left,
            rectified_size,
            MakePngOutputPath(thresholdDirectory(threshold), left_path));
    }

    bool processSinglePair(const fs::path& left_path, const fs::path& right_path) {
        const cv::Mat left_image = cv::imread(left_path.string(), cv::IMREAD_COLOR);
        const cv::Mat right_image = cv::imread(right_path.string(), cv::IMREAD_COLOR);

        if (left_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load left image: %s", left_path.string().c_str());
            return false;
        }
        if (right_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load right image: %s", right_path.string().c_str());
            return false;
        }

        cv::Mat prepared_left_image;
        cv::Mat prepared_right_image;
        PreparedImage prepared_left;
        cv::Size rectified_size;
        if (!preprocess(
                left_image,
                right_image,
                &prepared_left_image,
                &prepared_right_image,
                &prepared_left,
                &rectified_size)) {
            return false;
        }

        const auto start = std::chrono::high_resolution_clock::now();

        cv::Mat disparity;
        estimator_->inference(prepared_left_image, prepared_right_image, disparity);

        const auto infer_end = std::chrono::high_resolution_clock::now();

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

        const cv::Mat raw_depth_model = DisparityToDepthMeters(
            disparity,
            static_cast<float>(intrinsics.fx),
            static_cast<float>(baseline_meters),
            min_depth_meters_,
            max_depth_meters_);
        const cv::Mat median_depth_model =
            ApplyMedianDepthFilter(raw_depth_model, median_kernel_size_);
        const cv::Mat bilateral_depth_model = ApplyBilateralDepthFilter(
            raw_depth_model,
            bilateral_d_,
            bilateral_sigma_color_,
            bilateral_sigma_space_);

        const ConfidenceComponents components = computeConfidenceComponents(
            prepared_left_image,
            prepared_right_image,
            raw_depth_model,
            intrinsics,
            static_cast<float>(baseline_meters));

        const cv::Mat conf_photo = CombineConfidenceTerms(
            raw_depth_model,
            {{&components.photo, confidence_params_.w_photo}},
            confidence_params_.eps);
        const cv::Mat conf_tex_grad = CombineConfidenceTerms(
            raw_depth_model,
            {
                {&components.texture, confidence_params_.w_tex},
                {&components.gradient, confidence_params_.w_grad},
            },
            confidence_params_.eps);
        const cv::Mat conf_photo_tex_grad = CombineConfidenceTerms(
            raw_depth_model,
            {
                {&components.photo, confidence_params_.w_photo},
                {&components.texture, confidence_params_.w_tex},
                {&components.gradient, confidence_params_.w_grad},
            },
            confidence_params_.eps);

        cv::Mat conf_full;
        if (components.temporal_available) {
            conf_full = CombineConfidenceTerms(
                raw_depth_model,
                {
                    {&components.photo, confidence_params_.w_photo},
                    {&components.texture, confidence_params_.w_tex},
                    {&components.gradient, confidence_params_.w_grad},
                    {&components.temporal, confidence_params_.w_tmp},
                },
                confidence_params_.eps);
        } else {
            conf_full = conf_photo_tex_grad.clone();
        }

        if (conf_photo.empty() ||
            conf_tex_grad.empty() ||
            conf_photo_tex_grad.empty() ||
            conf_full.empty()) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Failed to compute one or more confidence maps for %s.",
                left_path.filename().string().c_str());
            return false;
        }

        const cv::Mat depth_photo_only = depth_confidence::filter_depth_with_confidence(
            raw_depth_model,
            conf_photo,
            confidence_params_.conf_threshold);
        const cv::Mat depth_tex_grad = depth_confidence::filter_depth_with_confidence(
            raw_depth_model,
            conf_tex_grad,
            confidence_params_.conf_threshold);
        const cv::Mat depth_photo_tex_grad = depth_confidence::filter_depth_with_confidence(
            raw_depth_model,
            conf_photo_tex_grad,
            confidence_params_.conf_threshold);
        const cv::Mat depth_full = depth_confidence::filter_depth_with_confidence(
            raw_depth_model,
            conf_full,
            confidence_params_.conf_threshold);

        bool ok = true;
        ok = saveGroup("raw_depth", left_path, raw_depth_model, prepared_left, rectified_size) && ok;
        ok = saveGroup("median_filter", left_path, median_depth_model, prepared_left, rectified_size) && ok;
        ok = saveGroup("bilateral_filter", left_path, bilateral_depth_model, prepared_left, rectified_size) && ok;
        ok = saveGroup("confidence_photo_only", left_path, depth_photo_only, prepared_left, rectified_size) && ok;
        ok = saveGroup("confidence_tex_grad", left_path, depth_tex_grad, prepared_left, rectified_size) && ok;
        ok = saveGroup("confidence_photo_tex_grad", left_path, depth_photo_tex_grad, prepared_left, rectified_size) && ok;
        ok = saveGroup("confidence_full", left_path, depth_full, prepared_left, rectified_size) && ok;

        for (const float threshold : threshold_sweep_) {
            const cv::Mat swept_depth = depth_confidence::filter_depth_with_confidence(
                raw_depth_model,
                conf_full,
                threshold);
            ok = saveThresholdGroup(threshold, left_path, swept_depth, prepared_left, rectified_size) && ok;
        }

        previous_depth_ = raw_depth_model.clone();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto infer_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - start).count();
        const auto total_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        RCLCPP_INFO(
            this->get_logger(),
            "Processed %s in %ld ms (inference %ld ms, raw valid %.2f%%, temporal %s).",
            left_path.filename().string().c_str(),
            total_ms,
            infer_ms,
            ComputeValidRatio(raw_depth_model) * 100.0,
            components.temporal_available ? "on" : "fallback");
        return ok;
    }

    void createOutputDirectories() const {
        const std::vector<std::string> group_names = {
            "raw_depth",
            "median_filter",
            "bilateral_filter",
            "confidence_photo_only",
            "confidence_tex_grad",
            "confidence_photo_tex_grad",
            "confidence_full",
        };

        for (const std::string& group_name : group_names) {
            fs::create_directories(groupDirectory(group_name));
        }
        for (const float threshold : threshold_sweep_) {
            fs::create_directories(thresholdDirectory(threshold));
        }
    }

    void runDataset() {
        run_timer_->cancel();

        try {
            createOutputDirectories();

            std::vector<std::string> filenames = CollectMatchedImageNames(left_dir_, right_dir_);
            if (filenames.empty()) {
                throw std::runtime_error(
                    "No matched stereo image pairs found under " + left_dir_.string() +
                    " and " + right_dir_.string() + ".");
            }

            if (max_pairs_ > 0 && static_cast<std::size_t>(max_pairs_) < filenames.size()) {
                filenames.resize(static_cast<std::size_t>(max_pairs_));
            }

            RCLCPP_INFO(
                this->get_logger(),
                "Starting offline benchmark for %zu stereo pairs.",
                filenames.size());
            RCLCPP_INFO(
                this->get_logger(),
                "Result groups are saved under %s.",
                output_root_.string().c_str());

            std::size_t success_count = 0;
            for (const std::string& filename : filenames) {
                const fs::path left_path = left_dir_ / filename;
                const fs::path right_path = right_dir_ / filename;
                if (processSinglePair(left_path, right_path)) {
                    ++success_count;
                }
            }

            RCLCPP_INFO(
                this->get_logger(),
                "Offline benchmark completed: %zu/%zu pairs succeeded.",
                success_count,
                filenames.size());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Offline benchmark failed: %s", e.what());
        }

        rclcpp::shutdown();
    }

    int model_input_height_ = 448;
    int model_input_width_ = 640;
    int input_image_height_ = 1080;
    int input_image_width_ = 1920;

    std::vector<std::string> engine_file_path_;
    std::string model_type_;
    std::string caminfo_path_;

    fs::path dataset_root_;
    fs::path left_dir_;
    fs::path right_dir_;
    fs::path output_root_;

    float min_depth_meters_ = 0.1f;
    float max_depth_meters_ = 100.0f;
    double depth_scale_ = 1000.0;
    bool save_input_resolution_ = true;
    int max_pairs_ = 0;

    int median_kernel_size_ = 5;
    int bilateral_d_ = 5;
    double bilateral_sigma_color_ = 0.05;
    double bilateral_sigma_space_ = 5.0;
    bool use_temporal_confidence_ = true;
    std::vector<float> threshold_sweep_ = {0.20f, 0.30f, 0.35f, 0.40f, 0.50f};

    StereoCalibration calibration_;
    StereoRectificationMaps rectification_maps_;
    bool rectification_ready_ = false;

    std::unique_ptr<StereoEstimator> estimator_;
    rclcpp::TimerBase::SharedPtr run_timer_;
    depth_confidence::ConfidenceParameters confidence_params_;
    cv::Mat previous_depth_;
};

void signal_handler(int sig) {
    RCLCPP_WARN(rclcpp::get_logger("OfflineDepthBenchmarkNode"), "catch sig %d", sig);
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    std::signal(SIGINT, signal_handler);

    try {
        rclcpp::spin(std::make_shared<OfflineDepthBenchmarkNode>());
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& e) {
        RCLCPP_FATAL(
            rclcpp::get_logger("OfflineDepthBenchmarkNode"),
            "Unhandled exception: %s",
            e.what());
    } catch (...) {
        RCLCPP_FATAL(
            rclcpp::get_logger("OfflineDepthBenchmarkNode"),
            "Unhandled non-standard exception.");
    }

    rclcpp::shutdown();
    return 1;
}
