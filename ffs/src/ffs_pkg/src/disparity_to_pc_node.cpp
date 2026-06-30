#include <csignal>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "depth_confidence.hpp"
#include "stereo_calibration_utils.hpp"

// 本节点将视差图转换为深度图，并基于双目图像计算置信度、加权图像和滤波后的深度。
// 主要流程：
// 1. 读取视差图和双目图像；
// 2. 校正双目图像；
// 3. 准备输入尺寸；
// 4. 将视差转为深度；
// 5. 计算置信度并滤波；
// 6. 发布多路浮点图像。
namespace {

struct PreparedImage {
    cv::Mat image;
    double scale = 1.0;
    int pad_left = 0;
    int pad_top = 0;
};

struct CameraIntrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

// 将输入图像转换到目标尺寸，保持纵横比并进行边界复制填充。
// 返回调整后的图像、缩放比例和填充偏移量，用于后续相机内参修正。
PreparedImage prepare_stereo_input(const cv::Mat& input, int target_width, int target_height) {
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

// 根据图像缩放和填充，调整相机内参以匹配准备后的图像。
// 这对于后续从视差计算深度时使用正确的焦距和主点位置非常重要。
CameraIntrinsics adjust_camera_intrinsics(
    const StereoCalibration& calibration,
    const PreparedImage& prepared,
    int width,
    int height) {
    (void)width;
    (void)height;

    CameraIntrinsics adjusted;
    adjusted.fx = calibration.k_left(0, 0) * prepared.scale;
    adjusted.fy = calibration.k_left(1, 1) * prepared.scale;
    adjusted.cx = calibration.k_left(0, 2) * prepared.scale + static_cast<double>(prepared.pad_left);
    adjusted.cy = calibration.k_left(1, 2) * prepared.scale + static_cast<double>(prepared.pad_top);
    return adjusted;
}

// 将视差图转换为深度图。深度 = fx * baseline / disparity。
// 如果视差无效或深度超出指定范围，则将深度值置零。
cv::Mat disparity_to_depth(
    const cv::Mat& disparity,
    float fx,
    float baseline,
    double zmin,
    double zfar) {
    cv::Mat depth = cv::Mat::zeros(disparity.size(), CV_32FC1);
    const float numerator = fx * baseline;

    for (int y = 0; y < disparity.rows; ++y) {
        const float* disp_row = disparity.ptr<float>(y);
        float* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float disp_value = disp_row[x];
            if (!std::isfinite(disp_value) || disp_value <= 0.0f) {
                depth_row[x] = 0.0f;
                continue;
            }

            const float depth_value = numerator / disp_value;
            if (!std::isfinite(depth_value) || depth_value <= static_cast<float>(zmin) || depth_value > static_cast<float>(zfar)) {
                depth_row[x] = 0.0f;
                continue;
            }

            depth_row[x] = depth_value;
        }
    }

    return depth;
}

// 将单通道浮点图像包装为 ROS Image 消息并发布。
// 这里统一使用 TYPE_32FC1 编码，适合深度图和置信度图等浮点数据。
void publish_float_image(
    const std_msgs::msg::Header& header,
    const cv::Mat& image,
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher) {
    cv_bridge::CvImage cv_image;
    cv_image.header = header;
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = image;
    publisher->publish(*cv_image.toImageMsg());
}

}  // namespace

class DisparityToDepthNode : public rclcpp::Node {
   public:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image>;

    DisparityToDepthNode() : Node("disparity_to_depth_node") {
        // 声明并读取深度范围参数
        this->declare_parameter("zfar", 10.0);
        zfar_ = this->get_parameter("zfar").as_double();

        this->declare_parameter("zmin", 0.1);
        zmin_ = this->get_parameter("zmin").as_double();

        this->declare_parameter("caminfo_path", std::string("caminfo.txt"));
        caminfo_path_ = this->get_parameter("caminfo_path").as_string();

        // 读取双目相机标定文件并提取基线长度
        std::string calibration_error;
        if (!LoadStereoCalibrationFromTxt(caminfo_path_, &calibration_, &calibration_error)) {
            throw std::runtime_error(calibration_error);
        }
        baseline_ = calibration_.baseline;

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

        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 10);
        filtered_depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("filtered_depth_image", 10);
        confidence_pub_ = this->create_publisher<sensor_msgs::msg::Image>("confidence_map", 10);
        weight_pub_ = this->create_publisher<sensor_msgs::msg::Image>("weight_map", 10);

        disparity_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "disparity");
        left_image_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "left_image");
        right_image_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "right_image");

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(20));
        sync_->connectInput(*disparity_sub_, *left_image_sub_, *right_image_sub_);
        sync_->setAgePenalty(0.20);
        sync_->registerCallback(
            std::bind(
                &DisparityToDepthNode::stereo_callback,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3));

        RCLCPP_INFO(
            this->get_logger(),
            "DisparityToDepthNode initialized with caminfo_path=%s, baseline=%.6f m.",
            caminfo_path_.c_str(),
            baseline_);
    }

   private:
    // 确保为当前图像尺寸计算了双目校正映射。
    // 校正映射只需为新的图像尺寸计算一次并缓存。
    bool ensure_rectification_maps(const cv::Size& image_size) {
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
            "Prepared rectification maps for %dx%d stereo images.",
            image_size.width,
            image_size.height);
        return true;
    }

    // 同步回调：处理视差图和左右图像，生成深度、置信度、权重图并发布。
    void stereo_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& disparity_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) {
        if (baseline_ <= 0.0) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Invalid baseline %.6f. Please configure a positive baseline.",
                baseline_);
            return;
        }

        cv::Mat disparity_mat;
        cv::Mat left_image;
        cv::Mat right_image;

        try {
            disparity_mat = cv_bridge::toCvCopy(
                disparity_msg,
                sensor_msgs::image_encodings::TYPE_32FC1)->image;
            left_image = cv_bridge::toCvCopy(left_msg, left_msg->encoding)->image;
            right_image = cv_bridge::toCvCopy(right_msg, right_msg->encoding)->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        if (left_image.size() != right_image.size()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Left/right image sizes do not match: %dx%d vs %dx%d.",
                left_image.cols,
                left_image.rows,
                right_image.cols,
                right_image.rows);
            return;
        }

        if (!ensure_rectification_maps(left_image.size())) {
            return;
        }

        cv::Mat rectified_left_image;
        cv::Mat rectified_right_image;
        cv::remap(
            left_image,
            rectified_left_image,
            rectification_maps_.left_map_x,
            rectification_maps_.left_map_y,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT);
        cv::remap(
            right_image,
            rectified_right_image,
            rectification_maps_.right_map_x,
            rectification_maps_.right_map_y,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT);

        PreparedImage prepared_left;
        PreparedImage prepared_right;
        try {
            prepared_left = prepare_stereo_input(rectified_left_image, disparity_mat.cols, disparity_mat.rows);
            prepared_right = prepare_stereo_input(rectified_right_image, disparity_mat.cols, disparity_mat.rows);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Image preparation failed: %s", e.what());
            return;
        }

        const CameraIntrinsics adjusted_intrinsics = adjust_camera_intrinsics(
            calibration_,
            prepared_left,
            disparity_mat.cols,
            disparity_mat.rows);

        const float fx = static_cast<float>(adjusted_intrinsics.fx);
        const float fy = static_cast<float>(adjusted_intrinsics.fy);
        const float cx = static_cast<float>(adjusted_intrinsics.cx);
        const float cy = static_cast<float>(adjusted_intrinsics.cy);

        if (fx <= 0.0f || fy <= 0.0f) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Invalid focal lengths fx=%.6f fy=%.6f after camera adjustment.",
                fx,
                fy);
            return;
        }

        // 视差转深度并计算置信度及权重图。
        cv::Mat depth_raw = disparity_to_depth(disparity_mat, fx, static_cast<float>(baseline_), zmin_, zfar_);
        cv::Mat confidence_map = depth_confidence::compute_confidence_map(
            prepared_left.image,
            prepared_right.image,
            depth_raw,
            fx,
            fy,
            cx,
            cy,
            static_cast<float>(baseline_),
            previous_depth_,
            confidence_params_);
        cv::Mat filtered_depth = depth_confidence::filter_depth_with_confidence(
            depth_raw,
            confidence_map,
            confidence_params_.conf_threshold);
        cv::Mat weight_map = depth_confidence::confidence_to_weight(
            confidence_map,
            confidence_params_.weight_gamma);

        previous_depth_ = depth_raw.clone();

        publish_float_image(disparity_msg->header, depth_raw, depth_pub_);
        publish_float_image(disparity_msg->header, filtered_depth, filtered_depth_pub_);
        publish_float_image(disparity_msg->header, confidence_map, confidence_pub_);
        publish_float_image(disparity_msg->header, weight_map, weight_pub_);
    }

    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> disparity_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_image_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> right_image_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr filtered_depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr confidence_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr weight_pub_;

    std::string caminfo_path_;
    StereoCalibration calibration_;
    StereoRectificationMaps rectification_maps_;
    bool rectification_ready_ = false;

    double baseline_ = 0.0;
    double zfar_ = 10.0;
    double zmin_ = 0.1;

    depth_confidence::ConfidenceParameters confidence_params_;
    cv::Mat previous_depth_;
};

// 捕获 SIGINT 信号，安全关闭 ROS 节点。
void signal_handler(int sig) {
    RCLCPP_WARN(rclcpp::get_logger("DisparityToDepthNode"), "catch sig %d", sig);
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DisparityToDepthNode>());
    rclcpp::shutdown();
    return 0;
}
