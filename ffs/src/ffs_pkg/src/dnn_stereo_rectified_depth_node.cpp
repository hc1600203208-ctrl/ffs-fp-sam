// -----------------------------------------------------------------------------
// 头文件包含区：声明本文件依赖的 ROS、OpenCV、标准库和自定义估计器接口。
// -----------------------------------------------------------------------------
#include "dnn_stereo_rectified_depth_node.h"  // 包含节点类声明，便于访问成员函数和成员变量

#include <algorithm>  // 提供 std::min/std::max/std::sort 等通用算法
#include <chrono>     // 提供时间类型与时间计时工具
#include <csignal>    // 提供信号处理函数，如 SIGINT
#include <cmath>      // 提供数学函数，如 std::fabs、std::round、std::isfinite
#include <iomanip>    // 提供 std::setw、std::setfill 等格式化输出工具
#include <limits>     // 提供数值范围，如 uint16_t 的最大值
#include <sstream>    // 提供字符串流，便于构造输出文件名
#include <vector>     // 提供 std::vector 容器

#include "estimator/fast_foundation_stereo_estimator.h"  // 包含双目深度估计器实现

using namespace std::chrono_literals;  // 便于写入 1s、200ms 等 chrono 字面量
namespace fs = std::filesystem;

namespace {

// -----------------------------------------------------------------------------
// 准备后的图像描述结构：存储缩放、补边、裁剪恢复所需信息。
// -----------------------------------------------------------------------------
struct PreparedImage {
    cv::Mat image;          // 经过缩放和补边后的目标图像
    double scale = 1.0;     // 输入图像相对目标分辨率的缩放比例
    int pad_left = 0;       // 左侧补边宽度
    int pad_top = 0;        // 上侧补边高度
    int resized_width = 0;  // 缩放后图像宽度
    int resized_height = 0; // 缩放后图像高度
};

struct CameraIntrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

// -----------------------------------------------------------------------------
// 检查两幅图像时间戳是否在允许误差范围内。
// -----------------------------------------------------------------------------
bool CheckTimestamp(
    const builtin_interfaces::msg::Time& time1,
    const builtin_interfaces::msg::Time& time2,
    double* cur_timestamp) {
    // 把 ROS Time 转换为秒数，便于做差值比较。
    const double timestamp1 =
        static_cast<double>(time1.sec) + static_cast<double>(time1.nanosec) / 1e9;
    const double timestamp2 =
        static_cast<double>(time2.sec) + static_cast<double>(time2.nanosec) / 1e9;

    // 将当前时间戳保存给调用者，便于后续使用。
    *cur_timestamp = timestamp1;

    // 计算时间差，判断是否超过允许阈值 0.07 秒。
    const double delta = timestamp1 - timestamp2;
    if (std::fabs(delta) > 0.07) {
        // 若时间差过大，记录错误日志并拒绝处理这对图像。
        RCLCPP_ERROR(
            rclcpp::get_logger("checkTimestamp"),
            "timestamp1: %f, timestamp2: %f, delta: %f",
            timestamp1,
            timestamp2,
            delta);
        return false;
    }

    // 时间差正常，继续处理。
    return true;
}

// -----------------------------------------------------------------------------
// 发布单张图像消息到指定话题。
// -----------------------------------------------------------------------------
void PublishDisparity(
    const std_msgs::msg::Header& header,
    const std::string& encoding,
    const cv::Mat& image_mat,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher) {
    // 创建 cv_bridge 图像消息容器，用于把 OpenCV Mat 转换成 ROS Image。
    cv_bridge::CvImage out_msg;
    out_msg.header = header;      // 复制原始消息头，便于时间戳和 frame_id 对齐
    out_msg.encoding = encoding;  // 指明图像编码格式，如 32FC1 或 BGR8
    out_msg.image = image_mat;    // 绑定待发布图像数据

    // 把 CvImage 转换为 ROS 消息并发布。
    publisher->publish(*out_msg.toImageMsg());
}

// -----------------------------------------------------------------------------
// 将输入图像缩放到目标尺寸，并在必要时进行边缘补齐，便于送入网络。
// 返回值中保存了缩放比例及补边信息，便于后续恢复原始分辨率。
// -----------------------------------------------------------------------------
PreparedImage PrepareStereoInput(const cv::Mat& input, int target_width, int target_height) {
    // 创建返回对象，并默认初始化为无效状态。
    PreparedImage result;

    // 若输入图像为空，直接返回空结果，调用方可判空处理。
    if (input.empty()) {
        return result;
    }

    // 创建一个可用于缩放的彩色图像副本。
    cv::Mat color;
    if (input.channels() == 1) {
        // 单通道图像（如灰度）转换为三通道，便于统一处理。
        cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
    } else if (input.channels() == 3) {
        // 三通道图像直接复制，避免额外转换开销。
        color = input.clone();
    } else {
        // 仅支持单通道或者三通道输入，否则无法正确处理。
        throw std::runtime_error("Unsupported image channel count.");
    }

    // 计算统一缩放比例，保证图像能同时缩放到目标宽高范围内。
    result.scale = std::min(
        static_cast<double>(target_width) / static_cast<double>(color.cols),
        static_cast<double>(target_height) / static_cast<double>(color.rows));

    // 根据缩放比例计算最终缩放后的尺寸，并保证至少为 1。
    const int resized_width = std::max(1, static_cast<int>(std::round(color.cols * result.scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(color.rows * result.scale)));
    result.resized_width = resized_width;
    result.resized_height = resized_height;

    // 将原图按目标缩放比例进行重采样。
    cv::Mat resized;
    cv::resize(color, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    // 计算目标尺寸与缩放后尺寸之间的差值，用于左右/上下补边。
    const int pad_width = target_width - resized_width;
    const int pad_height = target_height - resized_height;
    result.pad_left = pad_width / 2;                 // 左侧补边量
    const int pad_right = pad_width - result.pad_left;  // 右侧补边量
    result.pad_top = pad_height / 2;                 // 上侧补边量
    const int pad_bottom = pad_height - result.pad_top; // 下侧补边量

    // 通过复制边缘像素的方式把图像补齐到网络输入尺寸。
    cv::copyMakeBorder(
        resized,
        result.image,
        result.pad_top,
        pad_bottom,
        result.pad_left,
        pad_right,
        cv::BORDER_REPLICATE);

    // 返回包含缩放和补边信息的结果对象。
    return result;
}

// -----------------------------------------------------------------------------
// 将原始视差图转换为可视化图像，便于在 RViz 或图像查看器中观察结果。
// 这里使用 2% 到 98% 分位数来动态拉伸对比度，避免少量异常值影响显示。
// -----------------------------------------------------------------------------
cv::Mat VisualizeDisparity(const cv::Mat& disparity) {
    // 收集所有有效且大于 0 的视差值，用于后续统计量计算。
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

    // 创建一个单通道灰度图，用于后续生成颜色映射可视化。
    cv::Mat disp_vis = cv::Mat::zeros(disparity.size(), CV_8UC1);

    // 如果没有有效视差数据，直接返回一个默认的伪彩色图。
    if (valid_values.empty()) {
        cv::Mat color_vis;
        cv::applyColorMap(disp_vis, color_vis, cv::COLORMAP_JET);
        return color_vis;
    }

    // 对有效值排序，便于计算下界与上界。
    std::sort(valid_values.begin(), valid_values.end());
    const size_t lo_idx = static_cast<size_t>(0.02 * static_cast<double>(valid_values.size() - 1));
    const size_t hi_idx = static_cast<size_t>(0.98 * static_cast<double>(valid_values.size() - 1));
    const float lo = valid_values[lo_idx];
    const float hi = valid_values[hi_idx];

    // 如果上下界差值过小，则直接跳过归一化，保持黑图。
    if (hi - lo >= 1e-6f) {
        for (int y = 0; y < disparity.rows; ++y) {
            const float* src_row = disparity.ptr<float>(y);
            uint8_t* dst_row = disp_vis.ptr<uint8_t>(y);
            for (int x = 0; x < disparity.cols; ++x) {
                const float value = src_row[x];
                if (!std::isfinite(value) || value <= 0.0f) {
                    dst_row[x] = 0;
                    continue;
                }

                // 把视差值限制到 [lo, hi]，再映射到 [0,255]。
                const float clamped = std::min(std::max(value, lo), hi);
                const float normalized = (clamped - lo) / (hi - lo);
                dst_row[x] = static_cast<uint8_t>(std::round(normalized * 255.0f));
            }
        }
    }

    // 使用伪彩色映射把灰度图转换为彩色图像，便于观看。
    cv::Mat color_vis;
    cv::applyColorMap(disp_vis, color_vis, cv::COLORMAP_JET);
    return color_vis;
}

// -----------------------------------------------------------------------------
// 计算校正后图像对的基线长度（单位：米）。
// 这里从 rectification_maps.p2 中读取投影矩阵的相关值。
// -----------------------------------------------------------------------------
double ComputeRectifiedBaselineMeters(const StereoRectificationMaps& rectification_maps) {
    // 如果没有有效的 P2 矩阵，则无法计算基线。
    if (rectification_maps.p2.empty()) {
        return 0.0;
    }

    // 取 P2 的 fx 值和平移项，用于估算基线长度。
    const double fx = rectification_maps.p2.at<double>(0, 0);
    if (std::abs(fx) <= 1e-9) {
        return 0.0;
    }

    // 基线 = |Tx| / fx，其中 Tx 来自投影矩阵的第 0 行第 3 列。
    return std::abs(rectification_maps.p2.at<double>(0, 3) / fx);
}

// -----------------------------------------------------------------------------
// 计算模型输入图像对应的相机内参（考虑缩放和补边）。
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// 将视差图转换为深度图，单位为米。
// 公式：depth = fx * baseline / disparity。
// -----------------------------------------------------------------------------
cv::Mat DisparityToDepthMeters(
    const cv::Mat& disparity,
    float fx,
    float baseline,
    float min_depth_meters,
    float max_depth_meters) {
    // 创建一个与视差图同尺寸的深度图，默认初始化为 0。
    cv::Mat depth = cv::Mat::zeros(disparity.size(), CV_32FC1);
    const float numerator = fx * baseline;

    for (int y = 0; y < disparity.rows; ++y) {
        const float* disparity_row = disparity.ptr<float>(y);
        float* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float disparity_value = disparity_row[x];

            // 只有有效且大于 0 的视差值才可用于深度计算。
            if (!std::isfinite(disparity_value) || disparity_value <= 0.0f) {
                depth_row[x] = 0.0f;
                continue;
            }

            // 依据双目几何公式计算深度。
            const float depth_value = numerator / disparity_value;

            // 对结果进行范围过滤，剔除无效值和超出设定范围的点。
            if (!std::isfinite(depth_value) ||
                depth_value < min_depth_meters ||
                depth_value > max_depth_meters) {
                depth_row[x] = 0.0f;
                continue;
            }

            // 保留有效深度值。
            depth_row[x] = depth_value;
        }
    }

    return depth;
}

// -----------------------------------------------------------------------------
// 将模型输入分辨率下的深度图恢复到校正后原图分辨率。
// -----------------------------------------------------------------------------
cv::Mat RestoreToRectifiedResolution(
    const cv::Mat& image,
    const PreparedImage& prepared,
    const cv::Size& rectified_size,
    int interpolation) {
    // 输入图像为空或目标尺寸非法时，直接返回空结果。
    if (image.empty() || rectified_size.width <= 0 || rectified_size.height <= 0) {
        return {};
    }

    // 根据补边信息计算有效区域的 ROI，避免恢复时把补边像素混入结果。
    const cv::Rect valid_roi(
        prepared.pad_left,
        prepared.pad_top,
        std::min(prepared.resized_width, image.cols - prepared.pad_left),
        std::min(prepared.resized_height, image.rows - prepared.pad_top));

    // 若 ROI 非法，则说明没有可恢复的有效区域。
    if (valid_roi.width <= 0 || valid_roi.height <= 0) {
        return {};
    }

    // 只保留有效区域，再按原始校正图像大小进行放大恢复。
    const cv::Mat cropped = image(valid_roi);
    cv::Mat restored;
    cv::resize(cropped, restored, rectified_size, 0, 0, interpolation);
    return restored;
}

// -----------------------------------------------------------------------------
// 将米制深度图转换为 16 位 PNG 常用的整数深度格式。
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// 将输入图像转换为便于保存的 BGR 图像。
// -----------------------------------------------------------------------------
cv::Mat ConvertToSaveableBgr(const cv::Mat& input) {
    if (input.empty()) {
        return {};
    }

    if (input.channels() == 3) {
        return input.clone();
    }
    if (input.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (input.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }

    throw std::runtime_error("Unsupported image channel count for saving.");
}

// -----------------------------------------------------------------------------
// 生成每帧保存文件名的公共 stem，结合时间戳和计数器保证可追踪性与唯一性。
// -----------------------------------------------------------------------------
std::string BuildFrameStem(const std_msgs::msg::Header& header, std::uint64_t frame_index) {
    std::ostringstream oss;
    oss << "frame_" << std::setw(6) << std::setfill('0') << frame_index
        << "_" << header.stamp.sec
        << "_" << std::setw(9) << std::setfill('0') << header.stamp.nanosec;
    return oss.str();
}

}  // namespace

// -----------------------------------------------------------------------------
// 节点构造函数：声明参数、初始化 QoS 配置，并启动配置与激活流程。
// -----------------------------------------------------------------------------
DnnStereoRectifiedDepthNode::DnnStereoRectifiedDepthNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("dnn_stereo_rectified_depth_node", options),
      image_qos_profile_(rclcpp::QoS(10)),
      cam_info_qos_profile_(rclcpp::QoS(10)) {
    // 声明图像话题的可靠性策略参数，默认采用 best effort。
    this->declare_parameter(
        "image_reliability",
        static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));
    // 声明相机信息话题的可靠性策略参数。
    this->declare_parameter(
        "cam_info_reliability",
        static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));

    // 声明模型输入尺寸参数。
    this->declare_parameter("model_input_height", 448);
    this->declare_parameter("model_input_width", 640);
    // 声明原始输入图像尺寸参数。
    this->declare_parameter("input_image_height", 448);
    this->declare_parameter("input_image_width", 640);
    // 声明深度范围参数。
    this->declare_parameter("min_depth_meters", 0.1);
    this->declare_parameter("max_depth_meters", 100.0);
    this->declare_parameter("publish_filtered_depth", false);
    this->declare_parameter("publish_disparity", true);
    this->declare_parameter("publish_disparity_vis", true);
    this->declare_parameter("publish_depth_image", true);
    this->declare_parameter("publish_depth_image_raw", true);
    this->declare_parameter("publish_depth_image_filtered", true);
    this->declare_parameter("publish_confidence_map", true);
    this->declare_parameter("publish_weight_map", true);
    this->declare_parameter("save_results", false);
    this->declare_parameter("save_output_dir", std::string("/tmp/fast_foundation_stereo_capture"));
    this->declare_parameter("save_depth_scale", 1000.0);
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
    // 声明相机标定文件路径参数。
    this->declare_parameter("caminfo_path", std::string("caminfo.txt"));

    // 声明模型文件路径、模型类型以及触发模式参数。
    this->declare_parameter("engine_file_path", std::vector<std::string>{""});
    this->declare_parameter("model_type", "");
    this->declare_parameter("trigger_on_demands", true);

    // 构造函数中直接调用配置与激活逻辑，确保节点一创建就可工作。
    onConfigure();
    onActivate();
}

// -----------------------------------------------------------------------------
// 析构函数：先关闭订阅/同步器，再清理资源。
// -----------------------------------------------------------------------------
DnnStereoRectifiedDepthNode::~DnnStereoRectifiedDepthNode() {
    onDeactivate();
    onShutdown();
}

// -----------------------------------------------------------------------------
// 配置节点：读取参数、加载标定文件、初始化发布器和估计器。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::onConfigure() {
    RCLCPP_INFO(this->get_logger(), "Configuring rectified stereo depth node...");

    // 从参数服务器读取各类配置值，并保存到成员变量中。
    image_reliability_ = this->get_parameter("image_reliability").as_int();
    cam_info_reliability_ = this->get_parameter("cam_info_reliability").as_int();
    model_input_height_ = this->get_parameter("model_input_height").as_int();
    model_input_width_ = this->get_parameter("model_input_width").as_int();
    input_image_height_ = this->get_parameter("input_image_height").as_int();
    input_image_width_ = this->get_parameter("input_image_width").as_int();
    min_depth_meters_ = this->get_parameter("min_depth_meters").as_double();
    max_depth_meters_ = this->get_parameter("max_depth_meters").as_double();
    publish_filtered_depth_ = this->get_parameter("publish_filtered_depth").as_bool();
    publish_disparity_ = this->get_parameter("publish_disparity").as_bool();
    publish_disparity_vis_ = this->get_parameter("publish_disparity_vis").as_bool();
    publish_depth_image_ = this->get_parameter("publish_depth_image").as_bool();
    publish_depth_image_raw_ = this->get_parameter("publish_depth_image_raw").as_bool();
    publish_depth_image_filtered_ = this->get_parameter("publish_depth_image_filtered").as_bool();
    publish_confidence_map_ = this->get_parameter("publish_confidence_map").as_bool();
    publish_weight_map_ = this->get_parameter("publish_weight_map").as_bool();
    save_results_ = this->get_parameter("save_results").as_bool();
    save_output_dir_ = fs::path(this->get_parameter("save_output_dir").as_string());
    save_depth_scale_ = this->get_parameter("save_depth_scale").as_double();
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
    caminfo_path_ = this->get_parameter("caminfo_path").as_string();

    // 读取模型文件路径和模型类型。
    const std::vector<std::string> engine_file_path =
        this->get_parameter("engine_file_path").as_string_array();
    const std::string model_type = this->get_parameter("model_type").as_string();
    trigger_on_demands_ = this->get_parameter("trigger_on_demands").as_bool();

    // 根据参数设置图像和相机信息 QoS 策略。
    image_qos_profile_.reliability(
        static_cast<rclcpp::ReliabilityPolicy>(image_reliability_));
    image_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    image_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);

    cam_info_qos_profile_.reliability(
        static_cast<rclcpp::ReliabilityPolicy>(cam_info_reliability_));
    cam_info_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    cam_info_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);

    // 加载双目标定数据；失败时直接抛异常，终止节点初始化。
    std::string calibration_error;
    if (!LoadStereoCalibrationFromTxt(caminfo_path_, &calibration_, &calibration_error)) {
        throw std::runtime_error(calibration_error);
    }

    // 创建三个输出话题：视差图、可视化视差图和深度图。
    disparity_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("disparity", 1);
    disparity_image_vis_pub_ = this->create_publisher<sensor_msgs::msg::Image>("disparity_vis", 1);
    depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 1);
    depth_image_raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image_raw", 1);
    depth_image_filtered_pub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image_filtered", 1);
    confidence_map_pub_ = this->create_publisher<sensor_msgs::msg::Image>("confidence_map", 1);
    weight_map_pub_ = this->create_publisher<sensor_msgs::msg::Image>("weight_map", 1);

    if (save_results_) {
        save_rgb_dir_ = save_output_dir_ / "rgb";
        save_raw_depth_dir_ = save_output_dir_ / "depth_raw";
        save_filtered_depth_dir_ = save_output_dir_ / "depth_filtered";

        fs::create_directories(save_rgb_dir_);
        fs::create_directories(save_raw_depth_dir_);
        fs::create_directories(save_filtered_depth_dir_);

        RCLCPP_INFO(
            this->get_logger(),
            "Saving RGB/depth frames to %s (rgb: %s, raw depth: %s, filtered depth: %s).",
            save_output_dir_.c_str(),
            save_rgb_dir_.c_str(),
            save_raw_depth_dir_.c_str(),
            save_filtered_depth_dir_.c_str());
    }

    // 打印模型类型，便于运行时确认配置是否正确。
    RCLCPP_INFO(this->get_logger(), "model_type: %s.", model_type.c_str());
    const RectifiedStereoModel stereo_model = StringToRectifiedStereoModel(model_type);

    // 根据模型类型创建对应的估计器实例。
    switch (stereo_model) {
        case RectifiedStereoModel::FAST_FOUNDATION_STEREO:
            // 该类型需要两个 engine 文件：特征模型与后处理模型。
            if (engine_file_path.size() != 2) {
                throw std::runtime_error("FAST_FOUNDATION_STEREO expects exactly 2 engine paths.");
            }
            RCLCPP_INFO(this->get_logger(), "feature_model_file: %s.", engine_file_path[0].c_str());
            RCLCPP_INFO(this->get_logger(), "post_model_file: %s.", engine_file_path[1].c_str());

            // 构造深度估计器对象。
            estimator_ = std::make_unique<FastFoundationStereoEstimator>(
                engine_file_path[0],
                engine_file_path[1],
                model_input_height_,
                model_input_width_);
            break;
        default:
            // 不支持的模型类型直接报错。
            throw std::runtime_error("Unsupported stereo model type: " + model_type);
    }

    // 输出加载结果，方便确认基线信息是否正确。
    RCLCPP_INFO(
        this->get_logger(),
        "Loaded stereo calibration from %s with baseline %.6f m.",
        caminfo_path_.c_str(),
        calibration_.baseline);
    RCLCPP_INFO(this->get_logger(), "Configured.");
}

// -----------------------------------------------------------------------------
// 激活节点：建立图像订阅器、同步器以及触发服务。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::onActivate() {
    RCLCPP_INFO(this->get_logger(), "Activating...");

    // 创建左右图像订阅器，并使用配置好的 QoS。
    left_ir_image_sub_ =
        std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this,
            "left_ir_image",
            image_qos_profile_.get_rmw_qos_profile());
    right_ir_image_sub_ =
        std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this,
            "right_ir_image",
            image_qos_profile_.get_rmw_qos_profile());

    // 如果配置为按需触发，则在初始化时先取消订阅，避免一直接收数据。
    if (trigger_on_demands_) {
        left_ir_image_sub_->unsubscribe();
        right_ir_image_sub_->unsubscribe();
    }

    // 创建时间同步器，用于对齐左右图像帧。
    sync_ =
        std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
            ApproximateSyncPolicy(30));
    sync_->connectInput(*left_ir_image_sub_, *right_ir_image_sub_);
    sync_->setAgePenalty(0.20);
    sync_->registerCallback(
        std::bind(
            &DnnStereoRectifiedDepthNode::onStereoSyncCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2));

    // 创建服务，用于在外部调用时开启订阅处理流程。
    trigger_service_ = this->create_service<std_srvs::srv::Empty>(
        "trigger_dnn_stereo",
        std::bind(
            &DnnStereoRectifiedDepthNode::onTriggerCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3));

    RCLCPP_INFO(this->get_logger(), "Activated");
}

// -----------------------------------------------------------------------------
// 触发服务回调：收到服务调用后启用图像订阅。
// -----------------------------------------------------------------------------
bool DnnStereoRectifiedDepthNode::onTriggerCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<std_srvs::srv::Empty::Request> req,
    const std::shared_ptr<std_srvs::srv::Empty::Response> res) {
    (void)request_header;
    (void)req;
    (void)res;
    // 调用启用逻辑，开始接收并处理左右图像。
    onEnableCallback();
    return true;
}

// -----------------------------------------------------------------------------
// 启用回调：重新订阅左右图像，并启动一个定时器用于 3 秒后自动停用。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::onEnableCallback() {
    if (enabled_) {
        RCLCPP_WARN(this->get_logger(), "This service has already been enabled.");
        return;
    }

    // 标记节点进入工作状态，并恢复订阅。
    enabled_ = true;
    left_ir_image_sub_->subscribe();
    right_ir_image_sub_->subscribe();

    // 设置 3 秒后的定时器，用于自动关闭处理逻辑。
    const auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(3000.0));
    trigger_timer_ = rclcpp::create_timer(
        this,
        this->get_clock(),
        period_ms,
        std::bind(&DnnStereoRectifiedDepthNode::onDisableCallback, this));
}

// -----------------------------------------------------------------------------
// 停用回调：取消订阅、停止定时器，并停止处理流程。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::onDisableCallback() {
    if (!enabled_) {
        RCLCPP_WARN(this->get_logger(), "This service has already been disabled.");
        return;
    }

    // 关闭工作状态并取消定时器。
    enabled_ = false;
    trigger_timer_->cancel();

    // 暂时取消左右图像订阅，防止后续无效处理。
    left_ir_image_sub_->unsubscribe();
    right_ir_image_sub_->unsubscribe();
}

// -----------------------------------------------------------------------------
// 反激活节点：释放订阅器与同步器资源。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::onDeactivate() {
    RCLCPP_INFO(this->get_logger(), "Deactivating...");

    // 释放资源，避免退出时出现悬空引用。
    left_ir_image_sub_.reset();
    right_ir_image_sub_.reset();
    sync_.reset();

    RCLCPP_INFO(this->get_logger(), "Deactivated");
}

// -----------------------------------------------------------------------------
// 关闭节点：仅记录日志，实际清理由析构函数完成。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::onShutdown() {
    RCLCPP_INFO(this->get_logger(), "Shutting down...");
}

// -----------------------------------------------------------------------------
// 保存当前帧的左目 RGB 图像和对应深度图到指定目录。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::saveFrameResults(
    const std_msgs::msg::Header& header,
    const cv::Mat& left_rgb_image,
    const cv::Mat& raw_depth_aligned,
    const cv::Mat& filtered_depth_aligned) {
    if (!save_results_ || left_rgb_image.empty() ||
        raw_depth_aligned.empty() || filtered_depth_aligned.empty()) {
        return;
    }

    const std::string frame_stem = BuildFrameStem(header, saved_frame_count_++);
    const fs::path rgb_path = save_rgb_dir_ / (frame_stem + ".png");
    const fs::path raw_depth_path = save_raw_depth_dir_ / (frame_stem + ".png");
    const fs::path filtered_depth_path = save_filtered_depth_dir_ / (frame_stem + ".png");

    cv::Mat rgb_to_save;
    try {
        rgb_to_save = ConvertToSaveableBgr(left_rgb_image);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to convert RGB image for saving: %s", e.what());
        return;
    }

    const cv::Mat raw_depth_u16 =
        ConvertDepthMetersToUint16Millimeters(raw_depth_aligned, save_depth_scale_);
    const cv::Mat filtered_depth_u16 =
        ConvertDepthMetersToUint16Millimeters(filtered_depth_aligned, save_depth_scale_);

    if (!cv::imwrite(rgb_path.string(), rgb_to_save)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to save RGB image: %s", rgb_path.c_str());
        return;
    }
    if (!cv::imwrite(raw_depth_path.string(), raw_depth_u16)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to save raw depth image: %s", raw_depth_path.c_str());
        return;
    }
    if (!cv::imwrite(filtered_depth_path.string(), filtered_depth_u16)) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Failed to save filtered depth image: %s",
            filtered_depth_path.c_str());
        return;
    }
}

// -----------------------------------------------------------------------------
// 同步回调：一旦左右图像帧对齐，就交给 processOnce 处理。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::onStereoSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg) {
    processOnce(left_ir_msg, right_ir_msg);
}

// -----------------------------------------------------------------------------
// 确保校正映射已准备好；若输入尺寸变化，则重新计算。
// -----------------------------------------------------------------------------
bool DnnStereoRectifiedDepthNode::ensureRectificationMaps(const cv::Size& image_size) {
    // 如果已经缓存过且尺寸未变，则直接复用缓存结果。
    if (rectification_ready_ && rectification_maps_.image_size == image_size) {
        return true;
    }

    std::string rectification_error;
    // 通过标定数据和当前图像尺寸计算校正映射。
    if (!ComputeStereoRectificationMaps(
            calibration_,
            image_size,
            &rectification_maps_,
            &rectification_error)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to compute rectification maps: %s", rectification_error.c_str());
        return false;
    }

    // 标记校正映射已经准备好。
    rectification_ready_ = true;
    RCLCPP_INFO(
        this->get_logger(),
        "Rectification maps prepared for %dx%d input images.",
        image_size.width,
        image_size.height);
    return true;
}

// -----------------------------------------------------------------------------
// 预处理函数：校正、缩放、补边，确保左右图像满足网络输入格式。
// -----------------------------------------------------------------------------
bool DnnStereoRectifiedDepthNode::preprocess(cv::Mat& left_image, cv::Mat& right_image) {
    // 如果左右图像分辨率不一致，无法进行双目处理。
    if (left_image.size() != right_image.size()) {
        RCLCPP_INFO(this->get_logger(), "left_image.size() != right_image.size()");
        return false;
    }

    // 确保校正映射已经根据当前分辨率准备好。
    if (!ensureRectificationMaps(left_image.size())) {
        return false;
    }

    // 对左右图像执行校正（remap）。
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

    // 如果校正后的尺寸与配置的期望尺寸不一致，则记录一次警告。
    if (left_rectified.rows != input_image_height_ || left_rectified.cols != input_image_width_) {
        RCLCPP_WARN_ONCE(
            this->get_logger(),
            "Rectified input image size %dx%d does not match configured input_image_%dx%d, resizing to model size.",
            left_rectified.cols,
            left_rectified.rows,
            input_image_width_,
            input_image_height_);
    }

    // 将校正后的图像送入 PrepareStereoInput，获得网络输入尺寸。
    const PreparedImage prepared_left = PrepareStereoInput(left_rectified, model_input_width_, model_input_height_);
    const PreparedImage prepared_right = PrepareStereoInput(right_rectified, model_input_width_, model_input_height_);

    // 把预处理后的图像回写给调用方。
    left_image = prepared_left.image;
    right_image = prepared_right.image;

    // 若任一图像为空，说明预处理失败。
    if (left_image.empty() || right_image.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to prepare rectified stereo input.");
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------------
// 处理一对同步后的左右图像：转换消息、校正、推理、发布结果。
// -----------------------------------------------------------------------------
void DnnStereoRectifiedDepthNode::processOnce(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg) {
    // 处理空指针，避免崩溃。
    if (left_ir_msg == nullptr || right_ir_msg == nullptr) {
        return;
    }

    // 检查左右图像时间戳是否同步。
    double cur_timestamp = 0.0;
    if (!CheckTimestamp(left_ir_msg->header.stamp, right_ir_msg->header.stamp, &cur_timestamp)) {
        return;
    }

    // 将 ROS 图像消息转换为 OpenCV Mat，方便后续处理。
    cv::Mat left_ir_image;
    cv::Mat right_ir_image;
    try {
        const cv_bridge::CvImagePtr left_ir_cv_ptr =
            cv_bridge::toCvCopy(left_ir_msg, left_ir_msg->encoding);
        left_ir_image = left_ir_cv_ptr->image;

        const cv_bridge::CvImagePtr right_ir_cv_ptr =
            cv_bridge::toCvCopy(right_ir_msg, right_ir_msg->encoding);
        right_ir_image = right_ir_cv_ptr->image;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Exception in image callback: %s", e.what());
        return;
    }

    const cv::Mat left_image_to_save = left_ir_image.clone();

    // 记录整体处理开始时间，统计耗时。
    const auto start = std::chrono::high_resolution_clock::now();

    // 如果左右图片分辨率不同，直接放弃处理。
    if (left_ir_image.size() != right_ir_image.size()) {
        RCLCPP_ERROR(get_logger(), "Stereo image sizes do not match before preprocessing.");
        return;
    }

    // 确保校正映射与当前图像尺寸匹配。
    if (!ensureRectificationMaps(left_ir_image.size())) {
        RCLCPP_ERROR(get_logger(), "Failed to prepare rectification maps.");
        return;
    }

    // 对原始左右图像执行校正。
    cv::Mat left_rectified;
    cv::Mat right_rectified;
    cv::remap(
        left_ir_image,
        left_rectified,
        rectification_maps_.left_map_x,
        rectification_maps_.left_map_y,
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT);
    cv::remap(
        right_ir_image,
        right_rectified,
        rectification_maps_.right_map_x,
        rectification_maps_.right_map_y,
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT);

    // 将校正后图像缩放和补边为网络输入尺寸。
    const PreparedImage prepared_left = PrepareStereoInput(left_rectified, model_input_width_, model_input_height_);
    const PreparedImage prepared_right = PrepareStereoInput(right_rectified, model_input_width_, model_input_height_);

    // 用预处理后的图像替换原始图像，供推理使用。
    left_ir_image = prepared_left.image;
    right_ir_image = prepared_right.image;

    // 若预处理结果为空，说明处理失败。
    if (left_ir_image.empty() || right_ir_image.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed in preprocess.");
        return;
    }

    // 记录推理前时间，用于测量推理耗时。
    const auto start1 = std::chrono::high_resolution_clock::now();

    // 调用估计器进行左右图像推理，得到视差图。
    cv::Mat disparity;
    estimator_->inference(left_ir_image, right_ir_image, disparity);

    // 记录推理结束及整体结束时间。
    const auto end1 = std::chrono::high_resolution_clock::now();
    const auto end = std::chrono::high_resolution_clock::now();
    const auto total_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const auto infer_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();

    // 输出耗时日志，便于性能分析。
    RCLCPP_INFO(
        get_logger(),
        "Total time cost: %ld ms, InferenceOnce time cost: %ld ms.",
        total_duration,
        infer_duration);

    // 发布原始视差结果。
    if (publish_disparity_) {
        PublishDisparity(
            left_ir_msg->header,
            sensor_msgs::image_encodings::TYPE_32FC1,
            disparity,
            disparity_image_pub_);
    }

    // 计算基线与相机内参，用于把视差转换为深度并执行置信度估计。
    const double baseline_meters = ComputeRectifiedBaselineMeters(rectification_maps_);
    const CameraIntrinsics intrinsics = ComputePreparedIntrinsics(rectification_maps_, prepared_left);

    // 只有在内参有效时，才会继续生成深度图。
    if (baseline_meters > 0.0 && intrinsics.fx > 0.0 && intrinsics.fy > 0.0) {
        // 将视差转换为校正后模型分辨率下的深度图。
        const cv::Mat depth_rectified_model = DisparityToDepthMeters(
            disparity,
            static_cast<float>(intrinsics.fx),
            static_cast<float>(baseline_meters),
            static_cast<float>(min_depth_meters_),
            static_cast<float>(max_depth_meters_));

        const cv::Mat confidence_map = depth_confidence::compute_confidence_map(
            prepared_left.image,
            prepared_right.image,
            depth_rectified_model,
            static_cast<float>(intrinsics.fx),
            static_cast<float>(intrinsics.fy),
            static_cast<float>(intrinsics.cx),
            static_cast<float>(intrinsics.cy),
            static_cast<float>(baseline_meters),
            previous_depth_,
            confidence_params_);
        const cv::Mat filtered_depth_rectified_model = depth_confidence::filter_depth_with_confidence(
            depth_rectified_model,
            confidence_map,
            confidence_params_.conf_threshold);
        const cv::Mat weight_map = depth_confidence::confidence_to_weight(
            confidence_map,
            confidence_params_.weight_gamma);
        previous_depth_ = depth_rectified_model.clone();

        // 将深度图恢复到校正后原始分辨率。
        const cv::Mat raw_depth_rectified_full = RestoreToRectifiedResolution(
            depth_rectified_model,
            prepared_left,
            left_rectified.size(),
            cv::INTER_NEAREST);
        const cv::Mat filtered_depth_rectified_full = RestoreToRectifiedResolution(
            filtered_depth_rectified_model,
            prepared_left,
            left_rectified.size(),
            cv::INTER_NEAREST);

        // 将深度图对齐回原始左图坐标系。
        const cv::Mat raw_depth_aligned = AlignRectifiedDepthToOriginalLeft(
            raw_depth_rectified_full,
            rectification_maps_,
            cv::INTER_NEAREST);
        const cv::Mat filtered_depth_aligned = AlignRectifiedDepthToOriginalLeft(
            filtered_depth_rectified_full,
            rectification_maps_,
            cv::INTER_NEAREST);

        if (confidence_map.empty() || weight_map.empty()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *this->get_clock(),
                2000,
                "Skipping confidence publication because confidence estimation failed.");
            return;
        }

        if (publish_confidence_map_) {
            PublishDisparity(
                left_ir_msg->header,
                sensor_msgs::image_encodings::TYPE_32FC1,
                confidence_map,
                confidence_map_pub_);
        }
        if (publish_weight_map_) {
            PublishDisparity(
                left_ir_msg->header,
                sensor_msgs::image_encodings::TYPE_32FC1,
                weight_map,
                weight_map_pub_);
        }

        // 如果对齐成功，就发布深度结果。
        if (!raw_depth_aligned.empty() && !filtered_depth_aligned.empty()) {
            if (publish_depth_image_raw_) {
                PublishDisparity(
                    left_ir_msg->header,
                    sensor_msgs::image_encodings::TYPE_32FC1,
                    raw_depth_aligned,
                    depth_image_raw_pub_);
            }
            if (publish_depth_image_filtered_) {
                PublishDisparity(
                    left_ir_msg->header,
                    sensor_msgs::image_encodings::TYPE_32FC1,
                    filtered_depth_aligned,
                    depth_image_filtered_pub_);
            }
            if (publish_depth_image_) {
                PublishDisparity(
                    left_ir_msg->header,
                    sensor_msgs::image_encodings::TYPE_32FC1,
                    publish_filtered_depth_ ? filtered_depth_aligned : raw_depth_aligned,
                    depth_image_pub_);
            }
            saveFrameResults(
                left_ir_msg->header,
                left_image_to_save,
                raw_depth_aligned,
                filtered_depth_aligned);
        }
    } else {
        // 如果内参无效，跳过发布深度图，并给出提示。
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *this->get_clock(),
            2000,
            "Skipping depth publication because prepared rectified intrinsics are invalid.");
    }

    // 生成并发布视差可视化图。
    const cv::Mat disp_vis = VisualizeDisparity(disparity);
    if (publish_disparity_vis_) {
        PublishDisparity(
            left_ir_msg->header,
            sensor_msgs::image_encodings::BGR8,
            disp_vis,
            disparity_image_vis_pub_);
    }
}

// -----------------------------------------------------------------------------
// 信号处理函数：当收到 Ctrl+C 时，关闭 ROS 运行时。
// -----------------------------------------------------------------------------
void signal_handler(int sig) {
    RCLCPP_WARN(rclcpp::get_logger("DnnStereoRectifiedDepthNode"), "catch sig %d", sig);
    rclcpp::shutdown();
}

// -----------------------------------------------------------------------------
// 主函数：初始化 ROS、注册信号处理器，并启动节点循环。
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 初始化 ROS 2 客户端库。
    rclcpp::init(argc, argv);
    // 注册 Ctrl+C 信号处理函数，保证程序能正常退出。
    std::signal(SIGINT, signal_handler);

    try {
        // 创建并运行节点对象，直到 ROS 关闭为止。
        rclcpp::spin(std::make_shared<DnnStereoRectifiedDepthNode>());
        // 节点退出后关闭 ROS。
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& e) {
        // 捕获并打印异常信息，便于定位问题。
        RCLCPP_FATAL(
            rclcpp::get_logger("DnnStereoRectifiedDepthNode"),
            "Unhandled exception: %s",
            e.what());
    } catch (...) {
        // 捕获未知异常。
        RCLCPP_FATAL(
            rclcpp::get_logger("DnnStereoRectifiedDepthNode"),
            "Unhandled non-standard exception.");
    }

    // 发生异常时也要确保 ROS 被正常关闭。
    rclcpp::shutdown();
    return 1;
}
