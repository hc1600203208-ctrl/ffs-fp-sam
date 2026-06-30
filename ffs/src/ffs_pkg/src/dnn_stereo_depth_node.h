#ifndef DNN_STEREO_DEPTH_NODE_H
#define DNN_STEREO_DEPTH_NODE_H

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <std_srvs/srv/empty.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

//#include "estimator/rt_igev_stereo_estimator.h"
// #include "estimator/rt_monster_estimator.h"
//#include "estimator/defom_stereo_estimator.h"

#include "estimator/stereo_estimator.h"

enum class StereoModel {
    FAST_FOUNDATION_STEREO,
};

inline StereoModel StringToStereoModel(const std::string& model) {
    // Static map initialized once
    static const std::unordered_map<std::string, StereoModel> modelMap = {
        { "FAST_FOUNDATION_STEREO", StereoModel::FAST_FOUNDATION_STEREO },
    };

    auto it = modelMap.find(model);
    if (it != modelMap.end()) {
        return it->second;
    } else {
        // Handle unknown string, e.g., throw an exception or return a default value
        throw std::invalid_argument("Unknown StereoModel string: " + model);
    }
}

class DnnStereoDepthNode : public rclcpp::Node {
public:
    DnnStereoDepthNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    ~DnnStereoDepthNode();

private:
    void onConfigure();

    void onActivate();

    void onDeactivate();

    void onShutdown();

    // Parameters
    int image_reliability_;
    int cam_info_reliability_;

    // ROS2 components
    rclcpp::QoS image_qos_profile_;
    rclcpp::QoS cam_info_qos_profile_;

    double baseline_ = 0.095; // meter

    int model_input_height_ = 448; // pixel
    int model_input_width_ = 640;  // pixel

    int input_image_height_ = 448; // pixel
    int input_image_width_ = 640;  // pixel

    std::unique_ptr<StereoEstimator> estimator_;
    //std::unique_ptr<RTIGEVStereoEstimator> estimator_;
    // std::unique_ptr<RTMonsterEstimator> estimator_;

    std::atomic<bool> enabled_ = false;

    bool trigger_on_demands_ = true;

    // Typedef for synchronizer policy
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>
        ApproximateSyncPolicy;

    // Subscribers
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>
        left_ir_image_sub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>
        right_ir_image_sub_;


    // Synchronizer
    std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> sync_;

    // Publisher
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>>
        disparity_image_vis_pub_;

    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>>
        disparity_image_pub_;

    // Service
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr trigger_service_;

    // Timer
    rclcpp::TimerBase::SharedPtr trigger_timer_;

    // Callback
    void onStereoSyncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg);

    void onDisableCallback();

    void onEnableCallback();

    bool onTriggerCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                      const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                      const std::shared_ptr<std_srvs::srv::Empty::Response> res);


    // Processing methods
    void processOnce(const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg, const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg);

    bool preprocess(
        cv::Mat& left_ir_image, cv::Mat& right_ir_image);

};

#endif // DNN_STEREO_DEPTH_NODE_H
