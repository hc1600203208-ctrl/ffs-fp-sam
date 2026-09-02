#include "grounded_sam/pipeline.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

using Image = sensor_msgs::msg::Image;

cv::Mat RosImageToRgb(const Image::ConstSharedPtr &message)
{
  const auto image = cv_bridge::toCvCopy(message, message->encoding)->image;
  const std::string &encoding = message->encoding;

  if (encoding == sensor_msgs::image_encodings::RGB8 || encoding == "8UC3")
  {
    return image.clone();
  }

  cv::Mat rgb;
  if (encoding == sensor_msgs::image_encodings::BGR8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  }
  else if (encoding == sensor_msgs::image_encodings::RGBA8)
  {
    cv::cvtColor(image, rgb, cv::COLOR_RGBA2RGB);
  }
  else if (encoding == sensor_msgs::image_encodings::BGRA8 || encoding == "8UC4")
  {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  }
  else if (encoding == sensor_msgs::image_encodings::MONO8 || encoding == "8UC1")
  {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  }
  else
  {
    throw std::runtime_error("Unsupported image encoding: " + encoding);
  }
  return rgb;
}

std::filesystem::path ResolveMaskPath(const std::string &output_dir, std::string output_name)
{
  if (output_name.empty())
  {
    throw std::invalid_argument("output_name must not be empty");
  }
  if (std::filesystem::path(output_name).extension().empty())
  {
    output_name += ".png";
  }
  return std::filesystem::path(output_dir) / output_name;
}

void SaveMaskAtomically(const std::filesystem::path &path, const cv::Mat &mask)
{
  std::filesystem::create_directories(path.parent_path());
  const auto temporary_path = path.parent_path() / (path.filename().string() + ".tmp.png");
  if (!cv::imwrite(temporary_path.string(), mask))
  {
    throw std::runtime_error("Failed to write first mask to " + temporary_path.string());
  }

  std::error_code error;
  std::filesystem::rename(temporary_path, path, error);
  if (error)
  {
    std::filesystem::remove(temporary_path);
    throw std::runtime_error("Failed to publish first mask at " + path.string() + ": " +
                             error.message());
  }
}

class GroundedSamFirstMaskNode final : public rclcpp::Node
{
public:
  GroundedSamFirstMaskNode()
  : Node("grounded_sam_first_mask_node")
  {
    declare_parameter<std::string>("image_topic", "/left/image_raw");
    declare_parameter<std::string>("mask_topic", "/fisrt_mask");
    declare_parameter<bool>("publish_mask_topic", true);
    declare_parameter<std::string>("output_dir",
                                   "/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_outputs/first_mask");
    declare_parameter<std::string>("output_name", "first_mask.png");
    declare_parameter<std::string>(
        "dino_engine",
        "/home/hc/weizi/ffs+fp+sam/sam/engines/grounding_dino_fixed_prompt.engine");
    declare_parameter<std::string>("dino_onnx", "");
    declare_parameter<std::string>(
        "sam_encoder_engine",
        "/home/hc/weizi/ffs+fp+sam/sam/engines/sam_image_encoder.engine");
    declare_parameter<std::string>(
        "sam_decoder_engine",
        "/home/hc/weizi/ffs+fp+sam/sam/engines/sam_mask_decoder.engine");
    declare_parameter<int>("input_width", 640);
    declare_parameter<int>("input_height", 480);
    declare_parameter<double>("box_threshold", 0.3);
    declare_parameter<int>("max_detections", 16);

    image_topic_ = get_parameter("image_topic").as_string();
    mask_topic_ = get_parameter("mask_topic").as_string();
    publish_mask_topic_ = get_parameter("publish_mask_topic").as_bool();
    output_dir_ = get_parameter("output_dir").as_string();
    output_name_ = get_parameter("output_name").as_string();
    input_width_ = get_parameter("input_width").as_int();
    input_height_ = get_parameter("input_height").as_int();
    if (input_width_ <= 0 || input_height_ <= 0)
    {
      throw std::invalid_argument("input_width and input_height must be positive");
    }

    grounded_sam::PipelineOptions options;
    options.dino_engine = get_parameter("dino_engine").as_string();
    options.dino_onnx = get_parameter("dino_onnx").as_string();
    options.sam_encoder_engine = get_parameter("sam_encoder_engine").as_string();
    options.sam_decoder_engine = get_parameter("sam_decoder_engine").as_string();
    options.box_threshold = static_cast<float>(get_parameter("box_threshold").as_double());
    options.max_detections = get_parameter("max_detections").as_int();
    pipeline_ = std::make_unique<grounded_sam::GroundedSamPipeline>(options);

    const auto image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    if (publish_mask_topic_)
    {
      const auto mask_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
      mask_publisher_ = create_publisher<Image>(mask_topic_, mask_qos);
    }

    image_subscription_ = create_subscription<Image>(
        image_topic_, image_qos,
        std::bind(&GroundedSamFirstMaskNode::ImageCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "GroundedDINO + SAM ready; waiting for first frame on %s",
                image_topic_.c_str());
  }

private:
  void ImageCallback(const Image::ConstSharedPtr message)
  {
    if (processed_.load())
    {
      return;
    }
    processed_.store(true);

    try
    {
      const cv::Mat rgb = RosImageToRgb(message);
      if (rgb.cols != input_width_ || rgb.rows != input_height_)
      {
        throw std::runtime_error(
            "Expected a " + std::to_string(input_width_) + "x" +
            std::to_string(input_height_) + " image, received " +
            std::to_string(rgb.cols) + "x" + std::to_string(rgb.rows));
      }
      const cv::Mat mask = pipeline_->run(rgb);
      const auto path = ResolveMaskPath(output_dir_, output_name_);
      SaveMaskAtomically(path, mask);

      if (mask_publisher_)
      {
        auto mask_message = cv_bridge::CvImage(message->header,
                                               sensor_msgs::image_encodings::MONO8,
                                               mask)
                                .toImageMsg();
        mask_publisher_->publish(*mask_message);
      }

      RCLCPP_INFO(get_logger(),
                  "Saved first mask to %s (%d positive pixels); node is now idle",
                  path.string().c_str(), cv::countNonZero(mask));
      image_subscription_.reset();
    }
    catch (const std::exception &error)
    {
      processed_.store(false);
      RCLCPP_ERROR(get_logger(), "Failed to process first frame: %s", error.what());
    }
  }

  std::string image_topic_;
  std::string mask_topic_;
  bool publish_mask_topic_{true};
  std::string output_dir_;
  std::string output_name_;
  int input_width_{640};
  int input_height_{480};
  std::unique_ptr<grounded_sam::GroundedSamPipeline> pipeline_;
  std::atomic<bool> processed_{false};
  rclcpp::Subscription<Image>::SharedPtr image_subscription_;
  rclcpp::Publisher<Image>::SharedPtr mask_publisher_;
};

}  // namespace

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<GroundedSamFirstMaskNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::fprintf(stderr, "grounded_sam_first_mask_node: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
