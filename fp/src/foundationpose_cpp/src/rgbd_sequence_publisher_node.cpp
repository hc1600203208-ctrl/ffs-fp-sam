#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

namespace fs = std::filesystem;

namespace
{

struct FramePaths
{
  std::string id;
  fs::path    rgb_path;
  fs::path    depth_path;
};

struct CameraIntrinsics
{
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
};

std::vector<FramePaths> CollectFramePaths(const fs::path    &sequence_dir,
                                          const std::string &rgb_subdir,
                                          const std::string &depth_subdir)
{
  const fs::path rgb_dir   = sequence_dir / rgb_subdir;
  const fs::path depth_dir = sequence_dir / depth_subdir;

  if (!fs::exists(rgb_dir) || !fs::is_directory(rgb_dir))
  {
    throw std::invalid_argument("RGB directory does not exist: " + rgb_dir.string());
  }

  if (!fs::exists(depth_dir) || !fs::is_directory(depth_dir))
  {
    throw std::invalid_argument("Depth directory does not exist: " + depth_dir.string());
  }

  std::unordered_set<std::string> depth_stems;
  for (const auto &entry : fs::directory_iterator(depth_dir))
  {
    if (entry.is_regular_file())
    {
      depth_stems.insert(entry.path().stem().string());
    }
  }

  std::vector<FramePaths> frames;
  for (const auto &entry : fs::directory_iterator(rgb_dir))
  {
    if (!entry.is_regular_file())
    {
      continue;
    }

    const auto stem = entry.path().stem().string();
    if (depth_stems.find(stem) == depth_stems.end())
    {
      continue;
    }

    const fs::path depth_path = depth_dir / (stem + entry.path().extension().string());
    if (!fs::exists(depth_path))
    {
      continue;
    }

    frames.push_back(FramePaths{stem, entry.path(), depth_path});
  }

  std::sort(frames.begin(), frames.end(), [](const FramePaths &lhs, const FramePaths &rhs) {
    return lhs.id < rhs.id;
  });

  if (frames.empty())
  {
    throw std::runtime_error("No matching RGB/Depth frame pairs were found in " +
                             sequence_dir.string());
  }

  return frames;
}

cv::Mat LoadRgbImage(const fs::path &path)
{
  cv::Mat rgb = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (rgb.empty())
  {
    throw std::runtime_error("Failed to read RGB image: " + path.string());
  }
  return rgb;
}

cv::Mat LoadDepthImage(const fs::path &path)
{
  cv::Mat depth = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
  if (depth.empty())
  {
    throw std::runtime_error("Failed to read depth image: " + path.string());
  }

  if (depth.channels() != 1)
  {
    throw std::runtime_error("Depth image must be single-channel: " + path.string());
  }

  return depth;
}

std::string GetDepthEncoding(const cv::Mat &depth)
{
  switch (depth.type())
  {
  case CV_16UC1:
    return sensor_msgs::image_encodings::TYPE_16UC1;
  case CV_32FC1:
    return sensor_msgs::image_encodings::TYPE_32FC1;
  case CV_8UC1:
    return sensor_msgs::image_encodings::MONO8;
  default:
    throw std::runtime_error("Unsupported depth image type: " + std::to_string(depth.type()));
  }
}

rclcpp::Time MakeTimestamp(rclcpp::Clock &clock,
                           const std::string &frame_id,
                           bool use_filename_timestamp)
{
  if (!use_filename_timestamp)
  {
    return clock.now();
  }

  try
  {
    return rclcpp::Time(std::stoll(frame_id));
  }
  catch (const std::exception &)
  {
    throw std::runtime_error("Failed to parse frame id into timestamp: " + frame_id);
  }
}

CameraIntrinsics LoadCameraIntrinsics(const fs::path &path)
{
  std::ifstream file(path);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open camera intrinsic file: " + path.string());
  }

  double k00, k01, k02;
  double k10, k11, k12;
  double k20, k21, k22;
  file >> k00 >> k01 >> k02 >> k10 >> k11 >> k12 >> k20 >> k21 >> k22;
  if (file.fail())
  {
    throw std::runtime_error("Failed to parse camera intrinsic file: " + path.string());
  }

  CameraIntrinsics intrinsics;
  intrinsics.fx = k00;
  intrinsics.fy = k11;
  intrinsics.cx = k02;
  intrinsics.cy = k12;
  return intrinsics;
}

sensor_msgs::msg::CameraInfo BuildCameraInfo(const std_msgs::msg::Header &header,
                                             const CameraIntrinsics       &intrinsics,
                                             int                           width,
                                             int                           height)
{
  sensor_msgs::msg::CameraInfo info;
  info.header = header;
  info.width  = static_cast<uint32_t>(width);
  info.height = static_cast<uint32_t>(height);
  info.distortion_model = "plumb_bob";
  info.d = {0.0, 0.0, 0.0, 0.0, 0.0};

  info.k = {intrinsics.fx, 0.0,           intrinsics.cx,
            0.0,           intrinsics.fy, intrinsics.cy,
            0.0,           0.0,           1.0};

  info.r = {1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};

  info.p = {intrinsics.fx, 0.0,           intrinsics.cx, 0.0,
            0.0,           intrinsics.fy, intrinsics.cy, 0.0,
            0.0,           0.0,           1.0,           0.0};

  return info;
}

} // namespace

class RgbdSequencePublisherNode : public rclcpp::Node
{
public:
  RgbdSequencePublisherNode() : Node("rgbd_sequence_publisher_node")
  {
    DeclareParameters();
    LoadParameters();
    SetupPublishers();
    SetupTimer();

    RCLCPP_INFO(this->get_logger(),
                "RGBD publisher ready: %zu frames from %s at %.2f FPS",
                frames_.size(),
                sequence_dir_.c_str(),
                fps_);
  }

private:
  void DeclareParameters()
  {
    this->declare_parameter<std::string>(
        "sequence_dir", "/home/nvidia/ros_workspace/fp/src/foundationpose_cpp/test_data/mustard0");
    this->declare_parameter<std::string>("rgb_subdir", "rgb");
    this->declare_parameter<std::string>("depth_subdir", "depth");
    this->declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
    this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    this->declare_parameter<std::string>("rgb_camera_info_topic", "/camera/color/camera_info");
    this->declare_parameter<std::string>("depth_camera_info_topic", "/camera/depth/camera_info");
    this->declare_parameter<std::string>("frame_id", "camera_color_optical_frame");
    this->declare_parameter<std::string>("camera_info_path", "");
    this->declare_parameter<double>("fps", 10.0);
    this->declare_parameter<bool>("loop", true);
    this->declare_parameter<bool>("use_filename_timestamp", false);
    this->declare_parameter<bool>("publish_camera_info", true);
    this->declare_parameter<bool>("use_sensor_data_qos", false);
  }

  void LoadParameters()
  {
    sequence_dir_           = this->get_parameter("sequence_dir").as_string();
    rgb_subdir_             = this->get_parameter("rgb_subdir").as_string();
    depth_subdir_           = this->get_parameter("depth_subdir").as_string();
    rgb_topic_              = this->get_parameter("rgb_topic").as_string();
    depth_topic_            = this->get_parameter("depth_topic").as_string();
    rgb_camera_info_topic_  = this->get_parameter("rgb_camera_info_topic").as_string();
    depth_camera_info_topic_= this->get_parameter("depth_camera_info_topic").as_string();
    frame_id_               = this->get_parameter("frame_id").as_string();
    camera_info_path_       = this->get_parameter("camera_info_path").as_string();
    fps_                    = this->get_parameter("fps").as_double();
    loop_                   = this->get_parameter("loop").as_bool();
    use_filename_timestamp_ = this->get_parameter("use_filename_timestamp").as_bool();
    publish_camera_info_    = this->get_parameter("publish_camera_info").as_bool();
    use_sensor_data_qos_    = this->get_parameter("use_sensor_data_qos").as_bool();

    if (fps_ <= 0.0)
    {
      throw std::invalid_argument("Parameter `fps` must be positive.");
    }

    frames_ = CollectFramePaths(sequence_dir_, rgb_subdir_, depth_subdir_);

    if (publish_camera_info_)
    {
      fs::path intrinsic_path = camera_info_path_.empty() ? fs::path(sequence_dir_) / "cam_K.txt"
                                                          : fs::path(camera_info_path_);
      camera_intrinsics_ = LoadCameraIntrinsics(intrinsic_path);
    }
  }

  void SetupPublishers()
  {
    auto qos = use_sensor_data_qos_ ? rclcpp::SensorDataQoS() : rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    rgb_pub_   = this->create_publisher<sensor_msgs::msg::Image>(rgb_topic_, qos);
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(depth_topic_, qos);

    if (publish_camera_info_)
    {
      rgb_camera_info_pub_ =
          this->create_publisher<sensor_msgs::msg::CameraInfo>(rgb_camera_info_topic_, qos);
      depth_camera_info_pub_ =
          this->create_publisher<sensor_msgs::msg::CameraInfo>(depth_camera_info_topic_, qos);
    }
  }

  void SetupTimer()
  {
    const auto period =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / fps_));
    timer_ = this->create_wall_timer(period, std::bind(&RgbdSequencePublisherNode::PublishNextFrame, this));
  }

  void PublishNextFrame()
  {
    if (next_index_ >= frames_.size())
    {
      if (!loop_)
      {
        RCLCPP_INFO(this->get_logger(), "RGBD sequence completed. Stopping publisher.");
        timer_->cancel();
        return;
      }

      next_index_ = 0;
      RCLCPP_INFO(this->get_logger(), "RGBD sequence reached the end and restarted from frame 0.");
    }

    const auto &frame = frames_.at(next_index_);

    try
    {
      cv::Mat rgb   = LoadRgbImage(frame.rgb_path);
      cv::Mat depth = LoadDepthImage(frame.depth_path);

      if (rgb.rows != depth.rows || rgb.cols != depth.cols)
      {
        throw std::runtime_error("RGB and depth image sizes do not match for frame: " + frame.id);
      }

      const auto stamp = MakeTimestamp(*this->get_clock(), frame.id, use_filename_timestamp_);

      std_msgs::msg::Header header;
      header.stamp    = stamp;
      header.frame_id = frame_id_;

      auto rgb_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, rgb).toImageMsg();
      auto depth_msg = cv_bridge::CvImage(header, GetDepthEncoding(depth), depth).toImageMsg();

      rgb_pub_->publish(*rgb_msg);
      depth_pub_->publish(*depth_msg);

      if (publish_camera_info_)
      {
        auto rgb_info = BuildCameraInfo(header, camera_intrinsics_, rgb.cols, rgb.rows);
        auto depth_info = BuildCameraInfo(header, camera_intrinsics_, depth.cols, depth.rows);
        rgb_camera_info_pub_->publish(rgb_info);
        depth_camera_info_pub_->publish(depth_info);
      }

      RCLCPP_INFO_THROTTLE(this->get_logger(),
                           *this->get_clock(),
                           2000,
                           "Publishing RGBD sequence, current frame id: %s",
                           frame.id.c_str());
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to publish frame %s: %s", frame.id.c_str(), e.what());
      timer_->cancel();
      return;
    }

    ++next_index_;
  }

private:
  std::string sequence_dir_;
  std::string rgb_subdir_;
  std::string depth_subdir_;
  std::string rgb_topic_;
  std::string depth_topic_;
  std::string rgb_camera_info_topic_;
  std::string depth_camera_info_topic_;
  std::string frame_id_;
  std::string camera_info_path_;

  double fps_{10.0};
  bool   loop_{true};
  bool   use_filename_timestamp_{false};
  bool   publish_camera_info_{true};
  bool   use_sensor_data_qos_{false};

  std::vector<FramePaths> frames_;
  CameraIntrinsics        camera_intrinsics_;
  size_t                  next_index_{0};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rgb_camera_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_camera_info_pub_;
  rclcpp::TimerBase::SharedPtr                          timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    auto node = std::make_shared<RgbdSequencePublisherNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("rgbd_sequence_publisher_node"),
                 "Node startup failed: %s",
                 e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
