#ifndef DNN_STEREO_RECTIFIED_DEPTH_NODE_H
#define DNN_STEREO_RECTIFIED_DEPTH_NODE_H

#include <cstdint>
#include <filesystem>
#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <std_srvs/srv/empty.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include "depth_confidence.hpp"
#include "estimator/stereo_estimator.h"
#include "stereo_calibration_utils.hpp"

enum class RectifiedStereoModel {
    FAST_FOUNDATION_STEREO,
};

inline RectifiedStereoModel StringToRectifiedStereoModel(const std::string& model) {
    static const std::unordered_map<std::string, RectifiedStereoModel> model_map = {
        {"FAST_FOUNDATION_STEREO", RectifiedStereoModel::FAST_FOUNDATION_STEREO},
    };

    const auto it = model_map.find(model);
    if (it != model_map.end()) {
        return it->second;
    }

    throw std::invalid_argument("Unknown RectifiedStereoModel string: " + model);
}

class DnnStereoRectifiedDepthNode : public rclcpp::Node {
   public:
    DnnStereoRectifiedDepthNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    ~DnnStereoRectifiedDepthNode();

   private:
    void onConfigure();
    void onActivate();
    void onDeactivate();
    void onShutdown();

    void onStereoSyncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg);

    void onDisableCallback();
    void onEnableCallback();

    bool onTriggerCallback(
        const std::shared_ptr<rmw_request_id_t> request_header,
        const std::shared_ptr<std_srvs::srv::Empty::Request> req,
        const std::shared_ptr<std_srvs::srv::Empty::Response> res);

    void processOnce(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg);

    void saveFrameResults(
        const std_msgs::msg::Header& header,
        const cv::Mat& left_rgb_image,
        const cv::Mat& raw_depth_aligned,
        const cv::Mat& filtered_depth_aligned);

    bool preprocess(cv::Mat& left_ir_image, cv::Mat& right_ir_image);
    bool ensureRectificationMaps(const cv::Size& image_size);

    int image_reliability_ = 1;
    int cam_info_reliability_ = 1;

    rclcpp::QoS image_qos_profile_;
    rclcpp::QoS cam_info_qos_profile_;

    int model_input_height_ = 448;
    int model_input_width_ = 640;
    int input_image_height_ = 448;
    int input_image_width_ = 640;
    double min_depth_meters_ = 0.1;
    double max_depth_meters_ = 100.0;
    bool save_results_ = false;
    double save_depth_scale_ = 1000.0;
    bool publish_filtered_depth_ = false;
    bool publish_disparity_ = true;
    bool publish_disparity_vis_ = true;
    bool publish_depth_image_ = true;
    bool publish_depth_image_raw_ = true;
    bool publish_depth_image_filtered_ = true;
    bool publish_confidence_map_ = true;
    bool publish_weight_map_ = true;

    std::string caminfo_path_;
    std::filesystem::path save_output_dir_;
    std::filesystem::path save_rgb_dir_;
    std::filesystem::path save_raw_depth_dir_;
    std::filesystem::path save_filtered_depth_dir_;
    StereoCalibration calibration_;
    StereoRectificationMaps rectification_maps_;
    bool rectification_ready_ = false;
    std::uint64_t saved_frame_count_ = 0;

    std::unique_ptr<StereoEstimator> estimator_;
    std::atomic<bool> enabled_ = false;
    bool trigger_on_demands_ = true;
    depth_confidence::ConfidenceParameters confidence_params_;
    cv::Mat previous_depth_;

    using ApproximateSyncPolicy =
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image,
            sensor_msgs::msg::Image>;

    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_ir_image_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> right_ir_image_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> sync_;

    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> disparity_image_vis_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> disparity_image_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> depth_image_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> depth_image_raw_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> depth_image_filtered_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> confidence_map_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> weight_map_pub_;

    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr trigger_service_;
    rclcpp::TimerBase::SharedPtr trigger_timer_;
};

#endif  // DNN_STEREO_RECTIFIED_DEPTH_NODE_H
