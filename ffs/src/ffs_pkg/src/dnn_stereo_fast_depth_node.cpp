#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "estimator/fast_foundation_stereo_estimator.h"
#include "estimator/stereo_estimator.h"
#include "stereo_calibration_utils.hpp"

namespace {

namespace fs = std::filesystem;

enum class StereoModel {
    FAST_FOUNDATION_STEREO,
};

StereoModel StringToStereoModel(const std::string& model) {
    static const std::unordered_map<std::string, StereoModel> model_map = {
        {"FAST_FOUNDATION_STEREO", StereoModel::FAST_FOUNDATION_STEREO},
    };

    const auto it = model_map.find(model);
    if (it != model_map.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unknown stereo model string: " + model);
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

bool CheckTimestamp(
    const builtin_interfaces::msg::Time& time1,
    const builtin_interfaces::msg::Time& time2,
    double max_delta_seconds) {
    const double timestamp1 =
        static_cast<double>(time1.sec) + static_cast<double>(time1.nanosec) / 1e9;
    const double timestamp2 =
        static_cast<double>(time2.sec) + static_cast<double>(time2.nanosec) / 1e9;
    return std::fabs(timestamp1 - timestamp2) <= max_delta_seconds;
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
        color = input;
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
    result.pad_left = (target_width - result.resized_width) / 2;
    result.pad_top = (target_height - result.resized_height) / 2;

    cv::Mat resized;
    if (result.resized_width == color.cols && result.resized_height == color.rows) {
        resized = color;
    } else {
        cv::resize(color, resized, cv::Size(result.resized_width, result.resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    }

    result.image = cv::Mat::zeros(target_height, target_width, resized.type());
    resized.copyTo(result.image(cv::Rect(result.pad_left, result.pad_top, resized.cols, resized.rows)));
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
    intrinsics.cx = rectification_maps.p1.at<double>(0, 2) * prepared.scale + prepared.pad_left;
    intrinsics.cy = rectification_maps.p1.at<double>(1, 2) * prepared.scale + prepared.pad_top;
    return intrinsics;
}

cv::Mat DisparityToDepthMeters(
    const cv::Mat& disparity,
    float fx,
    float baseline_meters,
    float min_depth_meters,
    float max_depth_meters) {
    cv::Mat depth(disparity.size(), CV_32FC1, cv::Scalar(0.0f));
    const float focal_baseline = fx * baseline_meters;

    for (int y = 0; y < disparity.rows; ++y) {
        const float* disparity_row = disparity.ptr<float>(y);
        float* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float disparity_value = disparity_row[x];
            if (!std::isfinite(disparity_value) || disparity_value <= 1e-6f) {
                continue;
            }

            const float depth_value = focal_baseline / disparity_value;
            if (std::isfinite(depth_value) &&
                depth_value >= min_depth_meters &&
                depth_value <= max_depth_meters) {
                depth_row[x] = depth_value;
            }
        }
    }

    return depth;
}

cv::Mat RestoreToRectifiedResolution(
    const cv::Mat& prepared_depth,
    const PreparedImage& prepared,
    const cv::Size& rectified_size) {
    if (prepared_depth.empty() ||
        prepared.resized_width <= 0 ||
        prepared.resized_height <= 0 ||
        prepared.scale <= 0.0) {
        return {};
    }

    const cv::Rect valid_roi(
        prepared.pad_left,
        prepared.pad_top,
        prepared.resized_width,
        prepared.resized_height);
    cv::Mat cropped = prepared_depth(valid_roi);
    cv::Mat restored;
    cv::resize(cropped, restored, rectified_size, 0.0, 0.0, cv::INTER_NEAREST);
    return restored;
}

cv::Mat AlignRectifiedDepthToOriginalLeftFast(
    const cv::Mat& rectified_depth,
    const StereoRectificationMaps& rectification_maps) {
    if (rectified_depth.empty() ||
        rectification_maps.left_inverse_map_x.empty() ||
        rectification_maps.left_inverse_map_y.empty()) {
        return {};
    }

    cv::Mat aligned_depth;
    cv::remap(
        rectified_depth,
        aligned_depth,
        rectification_maps.left_inverse_map_x,
        rectification_maps.left_inverse_map_y,
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));
    return aligned_depth;
}

void PublishDepth(
    const std_msgs::msg::Header& header,
    const cv::Mat& depth,
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher) {
    cv_bridge::CvImage out_msg;
    out_msg.header = header;
    out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    out_msg.image = depth;
    publisher->publish(*out_msg.toImageMsg());
}

cv::Mat ConvertDepthMetersToUint16Millimeters(const cv::Mat& depth_meters) {
    cv::Mat depth_mm(depth_meters.size(), CV_16UC1, cv::Scalar(0));
    for (int y = 0; y < depth_meters.rows; ++y) {
        const float* depth_row = depth_meters.ptr<float>(y);
        uint16_t* output_row = depth_mm.ptr<uint16_t>(y);
        for (int x = 0; x < depth_meters.cols; ++x) {
            const float value_m = depth_row[x];
            if (!std::isfinite(value_m) || value_m <= 0.0f) {
                continue;
            }

            const float value_mm = value_m * 1000.0f;
            if (value_mm <= 0.0f) {
                continue;
            }
            output_row[x] = static_cast<uint16_t>(
                std::min(value_mm, static_cast<float>(std::numeric_limits<uint16_t>::max())));
        }
    }
    return depth_mm;
}

std::string BuildFrameFilename(size_t frame_index) {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << frame_index << ".png";
    return stream.str();
}

}  // namespace

class DnnStereoFastDepthNode : public rclcpp::Node {
   public:
    explicit DnnStereoFastDepthNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node("dnn_stereo_fast_depth_node", options),
          image_qos_profile_(rclcpp::QoS(10)) {
        declare_parameter("image_reliability", static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));
        declare_parameter("model_input_height", 448);
        declare_parameter("model_input_width", 640);
        declare_parameter("min_depth_meters", 0.1);
        declare_parameter("max_depth_meters", 100.0);
        declare_parameter("max_timestamp_delta_seconds", 0.07);
        declare_parameter("sync_queue_size", 10);
        declare_parameter("caminfo_path", std::string("caminfo.txt"));
        declare_parameter("engine_file_path", std::vector<std::string>{""});
        declare_parameter("model_type", std::string("FAST_FOUNDATION_STEREO"));
        declare_parameter("save_frame_outputs", false);
        declare_parameter("frame_output_dir", std::string("/home/bit/ffs+fp+sam/ffs/fast_depth_outputs"));

        configure();
        activate();
    }

    ~DnnStereoFastDepthNode() override {
        sync_.reset();
        left_image_sub_.reset();
        right_image_sub_.reset();
    }

   private:
    using ApproximateSyncPolicy =
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image,
            sensor_msgs::msg::Image>;

    void configure() {
        const int image_reliability = get_parameter("image_reliability").as_int();
        image_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(image_reliability));
        image_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
        image_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);

        model_input_height_ = get_parameter("model_input_height").as_int();
        model_input_width_ = get_parameter("model_input_width").as_int();
        min_depth_meters_ = get_parameter("min_depth_meters").as_double();
        max_depth_meters_ = get_parameter("max_depth_meters").as_double();
        max_timestamp_delta_seconds_ = get_parameter("max_timestamp_delta_seconds").as_double();
        sync_queue_size_ = get_parameter("sync_queue_size").as_int();
        caminfo_path_ = get_parameter("caminfo_path").as_string();
        save_frame_outputs_ = get_parameter("save_frame_outputs").as_bool();
        frame_output_dir_ = get_parameter("frame_output_dir").as_string();

        std::string calibration_error;
        if (!LoadStereoCalibrationFromTxt(caminfo_path_, &calibration_, &calibration_error)) {
            throw std::runtime_error(calibration_error);
        }

        if (save_frame_outputs_) {
            if (frame_output_dir_.empty()) {
                throw std::runtime_error("frame_output_dir must not be empty when save_frame_outputs is true.");
            }
            fs::create_directories(fs::path(frame_output_dir_) / "left");
            fs::create_directories(fs::path(frame_output_dir_) / "right");
            fs::create_directories(fs::path(frame_output_dir_) / "depth");
            RCLCPP_INFO(get_logger(), "Saving stereo frames and depth images to: %s", frame_output_dir_.c_str());
        }

        const std::vector<std::string> engine_file_path = get_parameter("engine_file_path").as_string_array();
        const std::string model_type = get_parameter("model_type").as_string();

        switch (StringToStereoModel(model_type)) {
            case StereoModel::FAST_FOUNDATION_STEREO:
                if (engine_file_path.size() != 2) {
                    throw std::runtime_error("FAST_FOUNDATION_STEREO expects exactly 2 engine paths.");
                }
                estimator_ = std::make_unique<FastFoundationStereoEstimator>(
                    engine_file_path[0],
                    engine_file_path[1],
                    model_input_height_,
                    model_input_width_);
                break;
        }

        depth_pub_ = create_publisher<sensor_msgs::msg::Image>("depth_image", 1);
        RCLCPP_INFO(
            get_logger(),
            "Fast depth-only node configured. Publishing raw metric depth on /depth_image.");
    }

    void activate() {
        left_image_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this,
            "left_ir_image",
            image_qos_profile_.get_rmw_qos_profile());
        right_image_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this,
            "right_ir_image",
            image_qos_profile_.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
            ApproximateSyncPolicy(sync_queue_size_));
        sync_->connectInput(*left_image_sub_, *right_image_sub_);
        sync_->setAgePenalty(0.20);
        sync_->registerCallback(
            std::bind(
                &DnnStereoFastDepthNode::onStereoSyncCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2));
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
            RCLCPP_ERROR(get_logger(), "Failed to compute rectification maps: %s", rectification_error.c_str());
            return false;
        }

        rectification_ready_ = true;
        RCLCPP_INFO(
            get_logger(),
            "Rectification maps prepared for %dx%d input images.",
            image_size.width,
            image_size.height);
        return true;
    }

    void onStereoSyncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) {
        if (left_msg == nullptr || right_msg == nullptr) {
            return;
        }

        if (!CheckTimestamp(left_msg->header.stamp, right_msg->header.stamp, max_timestamp_delta_seconds_)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Skipping stereo pair because timestamps are too far apart.");
            return;
        }

        cv_bridge::CvImageConstPtr left_cv_ptr;
        cv_bridge::CvImageConstPtr right_cv_ptr;
        try {
            left_cv_ptr = cv_bridge::toCvShare(left_msg, left_msg->encoding);
            right_cv_ptr = cv_bridge::toCvShare(right_msg, right_msg->encoding);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to convert stereo images: %s", e.what());
            return;
        }

        const cv::Mat& left_image = left_cv_ptr->image;
        const cv::Mat& right_image = right_cv_ptr->image;
        if (left_image.empty() || right_image.empty() || left_image.size() != right_image.size()) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid stereo image pair.");
            return;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        if (!ensureRectificationMaps(left_image.size())) {
            return;
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

        const PreparedImage prepared_left = PrepareStereoInput(left_rectified, model_input_width_, model_input_height_);
        const PreparedImage prepared_right = PrepareStereoInput(right_rectified, model_input_width_, model_input_height_);
        if (prepared_left.image.empty() || prepared_right.image.empty()) {
            RCLCPP_ERROR(get_logger(), "Failed to prepare stereo input.");
            return;
        }

        cv::Mat disparity;
        estimator_->inference(prepared_left.image, prepared_right.image, disparity);
        if (disparity.empty()) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Estimator returned empty disparity.");
            return;
        }

        const double baseline_meters = ComputeRectifiedBaselineMeters(rectification_maps_);
        const CameraIntrinsics intrinsics = ComputePreparedIntrinsics(rectification_maps_, prepared_left);
        if (baseline_meters <= 0.0 || intrinsics.fx <= 0.0) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid rectified stereo intrinsics.");
            return;
        }

        const cv::Mat depth_rectified_model = DisparityToDepthMeters(
            disparity,
            static_cast<float>(intrinsics.fx),
            static_cast<float>(baseline_meters),
            static_cast<float>(min_depth_meters_),
            static_cast<float>(max_depth_meters_));
        const cv::Mat depth_rectified_full = RestoreToRectifiedResolution(
            depth_rectified_model,
            prepared_left,
            left_rectified.size());
        const cv::Mat depth_aligned = AlignRectifiedDepthToOriginalLeftFast(
            depth_rectified_full,
            rectification_maps_);
        if (depth_aligned.empty()) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to align depth image.");
            return;
        }

        PublishDepth(left_msg->header, depth_aligned, depth_pub_);
        SaveFrameOutputs(left_image, right_image, depth_aligned);

        const auto end = std::chrono::high_resolution_clock::now();
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        RCLCPP_INFO(get_logger(), "Depth-only stereo frame processed in %ld ms.", total_ms);
    }

    void SaveFrameOutputs(
        const cv::Mat& left_image,
        const cv::Mat& right_image,
        const cv::Mat& depth_meters) {
        if (!save_frame_outputs_) {
            return;
        }

        const size_t frame_index = saved_frame_index_++;
        const std::string filename = BuildFrameFilename(frame_index);
        const fs::path output_root(frame_output_dir_);
        const fs::path left_path = output_root / "left" / filename;
        const fs::path right_path = output_root / "right" / filename;
        const fs::path depth_path = output_root / "depth" / filename;

        if (!cv::imwrite(left_path.string(), left_image)) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 2000, "Failed to save left image: %s", left_path.c_str());
        }
        if (!cv::imwrite(right_path.string(), right_image)) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 2000, "Failed to save right image: %s", right_path.c_str());
        }

        const cv::Mat depth_mm = ConvertDepthMetersToUint16Millimeters(depth_meters);
        if (!cv::imwrite(depth_path.string(), depth_mm)) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 2000, "Failed to save depth image: %s", depth_path.c_str());
        }
    }

    rclcpp::QoS image_qos_profile_;
    int model_input_height_ = 448;
    int model_input_width_ = 640;
    int sync_queue_size_ = 10;
    double min_depth_meters_ = 0.1;
    double max_depth_meters_ = 100.0;
    double max_timestamp_delta_seconds_ = 0.07;
    std::string caminfo_path_;
    bool save_frame_outputs_ = false;
    std::string frame_output_dir_;
    size_t saved_frame_index_ = 1;

    StereoCalibration calibration_;
    StereoRectificationMaps rectification_maps_;
    bool rectification_ready_ = false;

    std::unique_ptr<StereoEstimator> estimator_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_image_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> right_image_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
};

void signal_handler(int sig) {
    RCLCPP_WARN(rclcpp::get_logger("DnnStereoFastDepthNode"), "catch sig %d", sig);
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    rclcpp::init(argc, argv);

    try {
        rclcpp::spin(std::make_shared<DnnStereoFastDepthNode>());
    } catch (const std::exception& e) {
        RCLCPP_FATAL(
            rclcpp::get_logger("DnnStereoFastDepthNode"),
            "Unhandled exception: %s",
            e.what());
    }

    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    return 0;
}
