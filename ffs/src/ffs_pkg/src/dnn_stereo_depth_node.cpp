
#include "dnn_stereo_depth_node.h"

#include <chrono>
#include <algorithm>
#include <vector>

#include "estimator/fast_foundation_stereo_estimator.h"

using namespace std::chrono_literals; 

// 当前节点实现了一个基于深度学习的双目视差估计器。
// 该节点订阅左右红外图像，执行双目网络推理，并发布视差图和可视化结果。
namespace {

// 检查左右图像的时间戳是否足够接近，避免异步帧对造成错误。
// 如果两个时间戳差值超过 70ms，则认为它们不同步并返回 false。
bool CheckTimestamp(const builtin_interfaces::msg::Time& time1, const builtin_interfaces::msg::Time& time2, double* cur_timestamp) {
    double timestamp1 = static_cast<double>(time1.sec) + static_cast<double>(time1.nanosec) / 1e9;
    double timestamp2 = static_cast<double>(time2.sec) + static_cast<double>(time2.nanosec) / 1e9;

    *cur_timestamp = timestamp1;

    double delta = timestamp1 - timestamp2;

    // 70ms = 0.07s
    if (std::fabs(delta) > 0.07) {
        RCLCPP_ERROR(rclcpp::get_logger("checkTimestamp"), "timestamp1: %f, timestamp2: %f, delta: %f", timestamp1, timestamp2, delta);
        return false;
    }

    return true;
}

// 将 OpenCV 图像转换为 ROS Image 消息并发布到指定话题。
// header 保持与输入图像一致，以便下游节点可以使用同一时空信息。
void PublishDisparity(const std_msgs::msg::Header& header, const std::string& encoding, const cv::Mat& image_mat, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher) {
    // 1. Create a CvImage object
    cv_bridge::CvImage out_msg;

    // 2. Populate the header (timestamp and frame_id are crucial)
    out_msg.header = header;

    // 3. Specify the image encoding (must match your cv::Mat type)
    // Common encodings: "bgr8" (for CV_8UC3), "mono8" (for CV_8UC1), "TYPE_32FC1" (for CV_32F)
    //out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1; 
    out_msg.encoding = encoding;

    // 4. Assign the cv::Mat to the CvImage object
    out_msg.image = image_mat;

    // 5. Convert to a sensor_msgs::msg::Image::SharedPtr and publish
    sensor_msgs::msg::Image::SharedPtr msg = out_msg.toImageMsg();
    publisher->publish(*msg);
}

// 将输入图像转换为模型输入尺寸：
// 1. 转为 BGR 三通道；
// 2. 等比例缩放到目标尺寸；
// 3. 在上下左右填充以保持目标宽高。
cv::Mat PrepareStereoInput(const cv::Mat& input, int target_width, int target_height) {
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

// 将浮点视差图映射为可视化伪彩色图。
// 这里使用 2%-98% 的分位数来去除异常值影响，并将有效视差归一化为 0-255。
cv::Mat VisualizeDisparity(const cv::Mat& disparity) {
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
        cv::Mat color_vis;
        cv::applyColorMap(disp_vis, color_vis, cv::COLORMAP_JET);
        return color_vis;
    }

    std::sort(valid_values.begin(), valid_values.end());
    const size_t lo_idx = static_cast<size_t>(0.02 * static_cast<double>(valid_values.size() - 1));
    const size_t hi_idx = static_cast<size_t>(0.98 * static_cast<double>(valid_values.size() - 1));
    const float lo = valid_values[lo_idx];
    const float hi = valid_values[hi_idx];

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

                const float clamped = std::min(std::max(value, lo), hi);
                const float normalized = (clamped - lo) / (hi - lo);
                dst_row[x] = static_cast<uint8_t>(std::round(normalized * 255.0f));
            }
        }
    }

    cv::Mat color_vis;
    cv::applyColorMap(disp_vis, color_vis, cv::COLORMAP_JET);
    return color_vis;
}

}  // namespace

// 节点构造函数：声明参数并初始化生命周期配置。
// 这里直接调用 onConfigure 和 onActivate，使节点在创建时即可进入工作状态。
DnnStereoDepthNode::DnnStereoDepthNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("dnn_stereo_depth_node", options),
      image_qos_profile_(rclcpp::QoS(10)), cam_info_qos_profile_(rclcpp::QoS(10)) {
    // Declare parameters
    this->declare_parameter(
        "image_reliability",
        static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));
    this->declare_parameter(
        "cam_info_reliability",
        static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));

    this->declare_parameter("model_input_height", 448);
    this->declare_parameter("model_input_width", 640);
    this->declare_parameter("input_image_height", 448);
    this->declare_parameter("input_image_width", 640);

    this->declare_parameter("engine_file_path", std::vector<std::string>{""});
    this->declare_parameter("model_type", "");

    this->declare_parameter("trigger_on_demands", true);

    onConfigure();
    onActivate();
}

DnnStereoDepthNode::~DnnStereoDepthNode() {
    onDeactivate();
    onShutdown();
}

void DnnStereoDepthNode::onConfigure() {
    RCLCPP_INFO(this->get_logger(), "Configuring...");

    // Get parameters
    image_reliability_ =
        this->get_parameter("image_reliability").as_int();
    cam_info_reliability_ =
        this->get_parameter("cam_info_reliability").as_int();

    model_input_height_ =
        this->get_parameter("model_input_height").as_int();
    model_input_width_ =
        this->get_parameter("model_input_width").as_int(); 
    input_image_height_ =
        this->get_parameter("input_image_height").as_int();
    input_image_width_ =
        this->get_parameter("input_image_width").as_int(); 

    std::vector<std::string> engine_file_path =
        this->get_parameter("engine_file_path").as_string_array();  

    std::string model_type =
        this->get_parameter("model_type").as_string();

    trigger_on_demands_ =
        this->get_parameter("trigger_on_demands").as_bool();


    // Configure QoS profiles
    image_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(image_reliability_));
    image_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    image_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);

    cam_info_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(cam_info_reliability_));
    cam_info_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    cam_info_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);


    // Create publishers for视差图和可视化视差图
    disparity_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "disparity", 1);
    disparity_image_vis_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "disparity_vis", 1);


    // Create StereoEstimator
    RCLCPP_INFO(this->get_logger(), "model_type: %s.", model_type.c_str());
    StereoModel stereo_model = StringToStereoModel(model_type);

    switch(stereo_model) {
        case StereoModel::FAST_FOUNDATION_STEREO:
            assert(engine_file_path.size() == 2);
            RCLCPP_INFO(this->get_logger(), "feature_model_file: %s.", engine_file_path[0].c_str());
            RCLCPP_INFO(this->get_logger(), "post_model_file: %s.", engine_file_path[1].c_str());

            estimator_ = std::make_unique<FastFoundationStereoEstimator>(engine_file_path[0], engine_file_path[1], model_input_height_, model_input_width_);
            break;
        default:
            RCLCPP_ERROR(this->get_logger(), "no model fitted, model_type: %s.", model_type.c_str());
            exit(-1);
            break;
    }

    RCLCPP_INFO(this->get_logger(), "Configured.");
    return;
}

void DnnStereoDepthNode::onActivate() {
    RCLCPP_INFO(this->get_logger(), "Activating...");

    // Create subscribers
    left_ir_image_sub_ =
        std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "left_ir_image", image_qos_profile_.get_rmw_qos_profile());
    

    right_ir_image_sub_ =
        std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "right_ir_image", image_qos_profile_.get_rmw_qos_profile());

    // 如果需要按需触发，则默认先不订阅图像，等待服务触发后再订阅。
    if (trigger_on_demands_) {
        left_ir_image_sub_->unsubscribe();
        right_ir_image_sub_->unsubscribe();
    }

    // Create synchronizer
    sync_ =
        std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
            ApproximateSyncPolicy(30));
    sync_->connectInput(*left_ir_image_sub_, *right_ir_image_sub_);

    sync_->setAgePenalty(0.20); // 50 ms.
    sync_->registerCallback(
        std::bind(&DnnStereoDepthNode::onStereoSyncCallback, this, std::placeholders::_1, std::placeholders::_2));

    // Create trigger service for按需启用图像订阅
    trigger_service_ = this->create_service<std_srvs::srv::Empty>(
        "trigger_dnn_stereo", std::bind(&DnnStereoDepthNode::onTriggerCallback, this,
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3));

    RCLCPP_INFO(this->get_logger(), "Activated");
    return;
}

bool DnnStereoDepthNode::onTriggerCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                      const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                      const std::shared_ptr<std_srvs::srv::Empty::Response> res) {
    onEnableCallback();

    return true;
}

void DnnStereoDepthNode::onEnableCallback() {
    if (enabled_) {
        RCLCPP_WARN(this->get_logger(), "This service has already been enabled.");
        return;
    }

    enabled_ = true;

    left_ir_image_sub_->subscribe();
    right_ir_image_sub_->subscribe();

    auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(3000.0));
    trigger_timer_ = rclcpp::create_timer(
        this, this->get_clock(), period_ms,
        std::bind(&DnnStereoDepthNode::onDisableCallback, this));
}

void DnnStereoDepthNode::onDisableCallback() {
    if (!enabled_) {
        RCLCPP_WARN(this->get_logger(), "This service has already been disabled.");
        return;
    }

    enabled_ = false;

    trigger_timer_->cancel();

    left_ir_image_sub_->unsubscribe();
    right_ir_image_sub_->unsubscribe();
}

void DnnStereoDepthNode::onDeactivate() {
    RCLCPP_INFO(this->get_logger(), "Deactivating...");

    // Reset synchronizer and subscribers
    left_ir_image_sub_.reset();
    right_ir_image_sub_.reset();
    
    sync_.reset();

    RCLCPP_INFO(this->get_logger(), "Deactivated");
    return;
}

void DnnStereoDepthNode::onShutdown() {
    // Reset all resources
    // 此处可以释放模型、计时器、订阅器等资源。
    RCLCPP_INFO(this->get_logger(), "Shutting down...");
    return;
}

void DnnStereoDepthNode::onStereoSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg) {

    processOnce(left_ir_msg, right_ir_msg);

    return;
}

bool DnnStereoDepthNode::preprocess(
    cv::Mat& left_image, cv::Mat& right_image) {

    // 确保左右图像尺寸一致，否则不能直接作为双目输入。
    if (left_image.size() != right_image.size()) {
        RCLCPP_INFO(this->get_logger(), "left_image.size() != right_image.size()");
       return false;
    }

    // 如果输入图像尺寸与配置不同，则打印一次警告，并继续进行 resize 处理。
    if (left_image.rows != input_image_height_ || left_image.cols != input_image_width_) {
        RCLCPP_WARN_ONCE(
            this->get_logger(),
            "Input image size %dx%d does not match configured input_image_%dx%d, resizing to model size.",
            left_image.cols, left_image.rows, input_image_width_, input_image_height_);
    }

    left_image = PrepareStereoInput(left_image, model_input_width_, model_input_height_);
    right_image = PrepareStereoInput(right_image, model_input_width_, model_input_height_);

    if (left_image.empty() || right_image.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to prepare stereo input.");
        return false;
    }

    return true;
}

void DnnStereoDepthNode::processOnce(const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg, const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg) {

    if (left_ir_msg == nullptr || right_ir_msg == nullptr) {
        return;
    }

    // 检查左右图像的时间戳是否同步。
    double cur_timestamp;
    if (!CheckTimestamp(left_ir_msg->header.stamp, right_ir_msg->header.stamp, &cur_timestamp)) {
        return;
    }

    cv::Mat left_ir_image, right_ir_image;
    try {
        // Convert image
        cv_bridge::CvImagePtr left_ir_cv_ptr = cv_bridge::toCvCopy(left_ir_msg, left_ir_msg->encoding);
        left_ir_image = left_ir_cv_ptr->image;

        cv_bridge::CvImagePtr right_ir_cv_ptr = cv_bridge::toCvCopy(right_ir_msg, right_ir_msg->encoding);
        right_ir_image = right_ir_cv_ptr->image;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Exception in image callback: %s", e.what());
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (!preprocess(left_ir_image, right_ir_image)) {
        RCLCPP_ERROR(get_logger(), "Failed in preprocess.");
        return;
    }

    auto start1 = std::chrono::high_resolution_clock::now();

    // 执行深度学习推理，生成视差图。
    cv::Mat disparity;
    estimator_->inference(left_ir_image, right_ir_image, disparity);

    auto end1 = std::chrono::high_resolution_clock::now();


    auto end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();

    RCLCPP_INFO(get_logger(), "Total time cost: %ld ms, InferenceOnce time cost: %ld ms.", total_duration, infer_duration);

    // Publish raw disparity and可视化视差图
    PublishDisparity(left_ir_msg->header, sensor_msgs::image_encodings::TYPE_32FC1, disparity, disparity_image_pub_);
    cv::Mat disp_vis = VisualizeDisparity(disparity);
    PublishDisparity(left_ir_msg->header, sensor_msgs::image_encodings::BGR8, disp_vis, disparity_image_vis_pub_);

    return;
}

void signal_handler(int sig) {
    // 捕获 SIGINT 并安全退出 ROS 运行循环。
    RCLCPP_WARN(rclcpp::get_logger("DnnStereoDepthNode"), "catch sig %d", sig);
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    std::signal(SIGINT, signal_handler);

    try {
        // 启动节点并进入 ROS 事件循环。
        rclcpp::spin(std::make_shared<DnnStereoDepthNode>());
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("DnnStereoDepthNode"), "Unhandled exception: %s", e.what());
    } catch (...) {
        RCLCPP_FATAL(rclcpp::get_logger("DnnStereoDepthNode"), "Unhandled non-standard exception.");
    }

    rclcpp::shutdown();
    return 1;
}
