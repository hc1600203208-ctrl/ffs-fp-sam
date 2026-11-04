#include "grounded_sam/pipeline.hpp"

// 首帧分割入口：ROS 图像 -> 每目标 GroundedDINO 检测框 -> SAM 掩码 -> PNG。
// tracker 读取 PNG 文件；mask topic 只供其他订阅者使用，不参与本 launch 的注册。

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

using Image = sensor_msgs::msg::Image;

std::string SanitizeName(const std::string &name)
{
  // 仅用于 ROS topic 后缀；文件名由 mask_output_names 独立决定。
  std::string result;
  result.reserve(name.size());
  for (const unsigned char character : name)
  {
    if (std::isalnum(character))
    {
      result.push_back(static_cast<char>(std::tolower(character)));
    }
    else if (character == '_' || character == '-')
    {
      result.push_back(static_cast<char>(character));
    }
    else
    {
      result.push_back('_');
    }
  }
  return result.empty() ? "object" : result;
}

cv::Mat RosImageToRgb(const Image::ConstSharedPtr &message)
{
  // cv_bridge 按消息 encoding 解码；后续 DINO/SAM 均要求 RGB 顺序。
  const cv::Mat image = cv_bridge::toCvCopy(message, message->encoding)->image;
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
  // 与 tracker 的 tracked_mask_image_names 必须指向同一组文件。
  if (output_name.empty())
  {
    throw std::invalid_argument("mask_output_names must not contain an empty value");
  }
  if (std::filesystem::path(output_name).extension().empty())
  {
    output_name += ".png";
  }
  return std::filesystem::path(output_dir) / output_name;
}

void SaveMaskAtomically(const std::filesystem::path &path, const cv::Mat &mask)
{
  // 先写临时 PNG 再 rename，使 tracker 不会读取到半写入的文件。
  std::filesystem::create_directories(path.parent_path());
  const auto temporary_path = path.parent_path() / (path.filename().string() + ".tmp.png");
  if (!cv::imwrite(temporary_path.string(), mask))
  {
    throw std::runtime_error("Failed to write mask to " + temporary_path.string());
  }

  std::error_code error;
  std::filesystem::rename(temporary_path, path, error);
  if (error)
  {
    std::filesystem::remove(temporary_path);
    throw std::runtime_error("Failed to publish mask at " + path.string() + ": " + error.message());
  }
}

struct ObjectMaskPipeline
{
  // YAML 中 object_names、DINO 模型路径、mask_output_names 按相同下标配对。
  std::string name;
  std::filesystem::path output_path;
  std::unique_ptr<grounded_sam::GroundedSamPipeline> pipeline;
  rclcpp::Publisher<Image>::SharedPtr publisher;
};

class GroundedSamMultiFirstMaskNode final : public rclcpp::Node
{
public:
  GroundedSamMultiFirstMaskNode()
  : Node("grounded_sam_multi_first_mask_node")
  {
    // ROS 参数由 launch 的 YAML 和显式覆盖值注入；每个目标独占 DINO+SAM pipeline。
    declare_parameter<std::string>("image_topic", "/left/image_raw");
    declare_parameter<bool>("publish_mask_topics", true);
    declare_parameter<std::string>("mask_topic_prefix", "/first_masks");
    declare_parameter<std::string>(
        "output_dir", "/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_outputs/first_mask");
    declare_parameter<std::vector<std::string>>("object_names", std::vector<std::string>{});
    declare_parameter<std::vector<std::string>>("dino_engines", std::vector<std::string>{});
    declare_parameter<std::vector<std::string>>("dino_onnxs", std::vector<std::string>{});
    declare_parameter<std::vector<std::string>>("mask_output_names", std::vector<std::string>{});
    declare_parameter<std::string>(
        "sam_encoder_engine", "/home/hc/weizi/ffs+fp+sam/sam/engines/sam_image_encoder.engine");
    declare_parameter<std::string>(
        "sam_decoder_engine", "/home/hc/weizi/ffs+fp+sam/sam/engines/sam_mask_decoder.engine");
    declare_parameter<int>("input_width", 640);
    declare_parameter<int>("input_height", 480);
    declare_parameter<double>("box_threshold", 0.3);
    declare_parameter<int>("max_detections", 16);

    image_topic_ = get_parameter("image_topic").as_string();
    output_dir_ = get_parameter("output_dir").as_string();
    input_width_ = get_parameter("input_width").as_int();
    input_height_ = get_parameter("input_height").as_int();
    if (input_width_ <= 0 || input_height_ <= 0)
    {
      throw std::invalid_argument("input_width and input_height must be positive");
    }

    const auto object_names = get_parameter("object_names").as_string_array();
    const auto dino_engines = get_parameter("dino_engines").as_string_array();
    const auto dino_onnxs = get_parameter("dino_onnxs").as_string_array();
    const auto mask_output_names = get_parameter("mask_output_names").as_string_array();
    const bool use_dino_onnxs = !dino_onnxs.empty();
    if (object_names.empty())
    {
      throw std::invalid_argument("object_names must contain at least one object");
    }
    if (object_names.size() != mask_output_names.size())
    {
      throw std::invalid_argument(
          "object_names and mask_output_names must have the same length");
    }
    if (use_dino_onnxs)
    {
      if (object_names.size() != dino_onnxs.size())
      {
        throw std::invalid_argument("dino_onnxs must be empty or have the same length as object_names");
      }
      if (!dino_engines.empty() && object_names.size() != dino_engines.size())
      {
        throw std::invalid_argument("dino_engines must be empty or have the same length as object_names");
      }
    }
    else if (object_names.size() != dino_engines.size())
    {
      throw std::invalid_argument("dino_engines must have the same length as object_names");
    }

    const bool publish_mask_topics = get_parameter("publish_mask_topics").as_bool();
    const std::string mask_topic_prefix = get_parameter("mask_topic_prefix").as_string();
    const std::string sam_encoder_engine = get_parameter("sam_encoder_engine").as_string();
    const std::string sam_decoder_engine = get_parameter("sam_decoder_engine").as_string();
    const float box_threshold = static_cast<float>(get_parameter("box_threshold").as_double());
    const int max_detections = get_parameter("max_detections").as_int();
    const auto mask_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    // transient_local 使晚加入的订阅者仍可收到最后一张掩码。

    object_pipelines_.reserve(object_names.size());
    const auto initialization_started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < object_names.size(); ++index)
    {
      const bool missing_dino_model =
          use_dino_onnxs ? dino_onnxs[index].empty() : dino_engines[index].empty();
      if (object_names[index].empty() || missing_dino_model)
      {
        throw std::invalid_argument("object_names and DINO model paths must not contain empty values");
      }

      grounded_sam::PipelineOptions options;
      if (!dino_engines.empty())
      {
        options.dino_engine = dino_engines[index];
      }
      if (use_dino_onnxs)
      {
        options.dino_onnx = dino_onnxs[index];
      }
      options.sam_encoder_engine = sam_encoder_engine;
      options.sam_decoder_engine = sam_decoder_engine;
      options.box_threshold = box_threshold;
      options.max_detections = max_detections;

      ObjectMaskPipeline object_pipeline;
      object_pipeline.name = object_names[index];
      object_pipeline.output_path = ResolveMaskPath(output_dir_, mask_output_names[index]);
      object_pipeline.pipeline = std::make_unique<grounded_sam::GroundedSamPipeline>(options);
      if (publish_mask_topics)
      {
        object_pipeline.publisher = create_publisher<Image>(
            mask_topic_prefix + "/" + SanitizeName(object_pipeline.name), mask_qos);
      }
      object_pipelines_.push_back(std::move(object_pipeline));
    }

    const auto image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    image_subscription_ = create_subscription<Image>(
        // 最多保留一帧；处理成功后退订，失败则在下一帧重试。
        image_topic_, image_qos,
        std::bind(&GroundedSamMultiFirstMaskNode::ImageCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "GroundedDINO + SAM ready for %zu objects; waiting for a frame containing all targets on %s",
                object_pipelines_.size(), image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "First-mask model initialization took %.2f s",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - initialization_started).count());
  }

private:
  void ImageCallback(const Image::ConstSharedPtr message)
  {
    // 原子标志防止多个 ROS 回调同时处理图像；目标未齐或处理失败时允许下一帧重试。
    if (processed_.exchange(true))
    {
      return;
    }

    const auto frame_received = std::chrono::steady_clock::now();
    RCLCPP_INFO_ONCE(get_logger(), "First image received; checking detections for %zu objects", object_pipelines_.size());

    try
    {
      const cv::Mat rgb = RosImageToRgb(message);
      if (rgb.cols != input_width_ || rgb.rows != input_height_)
      {
        throw std::runtime_error(
            "Expected a " + std::to_string(input_width_) + "x" + std::to_string(input_height_) +
            " image, received " + std::to_string(rgb.cols) + "x" + std::to_string(rgb.rows));
      }

      std::vector<std::vector<grounded_sam::Detection>> detections;
      detections.reserve(object_pipelines_.size());
      for (auto &object_pipeline : object_pipelines_)
      {
        detections.push_back(object_pipeline.pipeline->detect(rgb));
        if (detections.back().empty())
        {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                               "Waiting for all targets; %s was not detected", object_pipeline.name.c_str());
          processed_.store(false);
          return;
        }
      }

      std::vector<cv::Mat> masks;
      masks.reserve(object_pipelines_.size());
      for (std::size_t index = 0; index < object_pipelines_.size(); ++index)
      {
        masks.push_back(object_pipelines_[index].pipeline->segment(rgb, detections[index]));
        if (masks.back().empty() || cv::countNonZero(masks.back()) == 0)
        {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                               "Waiting for all masks; %s produced an empty mask",
                               object_pipelines_[index].name.c_str());
          processed_.store(false);
          return;
        }
      }

      for (std::size_t index = 0; index < object_pipelines_.size(); ++index)
      {
        auto &object_pipeline = object_pipelines_[index];
        const cv::Mat &mask = masks[index];
        SaveMaskAtomically(object_pipeline.output_path, mask);
        if (object_pipeline.publisher)
        {
          auto mask_message = cv_bridge::CvImage(
                                  message->header, sensor_msgs::image_encodings::MONO8, mask)
                                  .toImageMsg();
          object_pipeline.publisher->publish(*mask_message);
        }
        RCLCPP_INFO(get_logger(), "Saved %s mask to %s (%d positive pixels)",
                    object_pipeline.name.c_str(), object_pipeline.output_path.string().c_str(),
                    cv::countNonZero(mask));
      }

      image_subscription_.reset();
      RCLCPP_INFO(get_logger(), "All target masks are ready; node is now idle");
      RCLCPP_INFO(get_logger(), "Accepted-image mask processing took %.2f s",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - frame_received).count());
    }
    catch (const std::exception &error)
    {
      for (const auto &object_pipeline : object_pipelines_)
      {
        std::error_code cleanup_error;
        std::filesystem::remove(object_pipeline.output_path, cleanup_error);
      }
      processed_.store(false);
      RCLCPP_ERROR(get_logger(), "Failed to process image: %s", error.what());
    }
  }

  std::string image_topic_;
  std::string output_dir_;
  int input_width_{640};
  int input_height_{480};
  std::vector<ObjectMaskPipeline> object_pipelines_;
  std::atomic<bool> processed_{false};
  rclcpp::Subscription<Image>::SharedPtr image_subscription_;
};

}  // namespace

int main(int argc, char **argv)
{
  // rclcpp::spin 持续分发图像回调；生成掩码后节点保持空闲而不退出。
  rclcpp::init(argc, argv);
  try
  {
    rclcpp::spin(std::make_shared<GroundedSamMultiFirstMaskNode>());
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::fprintf(stderr, "grounded_sam_multi_first_mask_node: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
