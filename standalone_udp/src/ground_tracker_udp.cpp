#include "udp_wire.hpp"
#include "tracker_math.hpp"
#include "depth_confidence.hpp"
#include "adaptive_se3_pose_filter.hpp"
#include "detection_6d_foundationpose/foundationpose.hpp"
#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "estimator/fast_foundation_stereo_estimator.h"
#include "grounded_sam/pipeline.hpp"
#include "trt_core/trt_core.h"

#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Geometry>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

namespace {
using namespace standalone_tracker;
template<class T> T value(const YAML::Node& node, const char* key, T fallback) {
  return node[key] ? node[key].as<T>() : fallback;
}
void put_double(std::vector<std::uint8_t>& bytes, double number) {
  std::uint64_t bits; std::memcpy(&bits, &number, sizeof(bits)); udp_wire::put64(bytes, bits);
}
std::shared_ptr<inference_core::BaseInferCore> pose_core(const std::string& path, bool scorer) {
  const std::uint64_t batch = 252;
  auto input = std::unordered_map<std::string, std::vector<std::uint64_t>>{
      {"transf_input", {batch, 160, 160, 6}}, {"render_input", {batch, 160, 160, 6}}};
  auto output = scorer ? std::unordered_map<std::string, std::vector<std::uint64_t>>{{"scores", {batch, 1}}}
      : std::unordered_map<std::string, std::vector<std::uint64_t>>{{"trans", {batch, 3}}, {"rot", {batch, 3}}};
  return inference_core::CreateTrtInferCore(path, input, output, 1);
}
std::string json_escape(const std::string& text) {
  std::string out;
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c < 32) out += '_';
    else out += c;
  }
  return out;
}
}

class GroundTrackerUdp {
  struct Object {
    std::string name, mask_file;
    cv::Scalar color;
    std::shared_ptr<detection_6d::BaseMeshLoader> mesh;
    std::unique_ptr<grounded_sam::GroundedSamPipeline> sam;
    cv::Mat mask;
    adaptive_pose::Filter filter;
    Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
    bool registered{false};
    std::uint64_t last_stamp{0};
  };
 public:
  GroundTrackerUdp(const std::string& yaml_path, std::uint16_t listen_port,
                   std::uint16_t app_port) {
    auto config = YAML::LoadFile(yaml_path);
    auto fp = config["foundationpose_stereo_tracker_multi_node"]["ros__parameters"];
    auto sam = config["grounded_sam_multi_first_mask_node"]["ros__parameters"];
    if (!fp || !sam) throw std::runtime_error("tracker YAML lacks FP or SAM parameters");
    auto engine_paths = fp["stereo_engine_file_path"].as<std::vector<std::string>>();
    auto names = fp["tracked_object_names"].as<std::vector<std::string>>();
    auto meshes = fp["tracked_mesh_paths"].as<std::vector<std::string>>();
    auto dino_engines = sam["dino_engines"].as<std::vector<std::string>>();
    auto mask_names = fp["tracked_mask_image_names"].as<std::vector<std::string>>();
    if (engine_paths.size() != 2 || names.empty() || names.size() != meshes.size() ||
        names.size() != dino_engines.size() || names.size() != mask_names.size())
      throw std::runtime_error("model and object list sizes do not match");
    for (const auto& file : {engine_paths[0], engine_paths[1],
                             fp["refiner_engine_path"].as<std::string>(),
                             fp["scorer_engine_path"].as<std::string>()})
      if (!std::filesystem::is_regular_file(file)) throw std::runtime_error("missing engine: " + file);
    for (const auto& file : dino_engines)
      if (!std::filesystem::is_regular_file(file)) throw std::runtime_error("missing DINO engine: " + file);
    for (const auto& file : {sam["sam_encoder_engine"].as<std::string>(),
                             sam["sam_decoder_engine"].as<std::string>()})
      if (!std::filesystem::is_regular_file(file)) throw std::runtime_error("missing SAM engine: " + file);
    std::string error;
    if (!LoadStereoCalibrationFromTxt(fp["caminfo_path"].as<std::string>(), &calibration_, &error))
      throw std::runtime_error(error);
    intrinsic_ = BuildIntrinsicMatrix(fp["fx"].as<double>(), fp["fy"].as<double>(),
                                      fp["cx"].as<double>(), fp["cy"].as<double>());
    model_width_ = value(fp, "model_input_width", 640);
    model_height_ = value(fp, "model_input_height", 448);
    max_depth_ = value(fp, "max_depth_meters", 100.0);
    min_depth_ = value(fp, "min_depth_meters", 0.1);
    enable_depth_confidence_filter_ = value(fp, "enable_depth_confidence_filter", false);
    confidence_params_.patch_radius = value(fp, "patch_radius", confidence_params_.patch_radius);
    confidence_params_.texture_ksize = value(fp, "texture_ksize", confidence_params_.texture_ksize);
    confidence_params_.sigma_photo = value(fp, "sigma_photo", confidence_params_.sigma_photo);
    confidence_params_.sigma_e = value(fp, "sigma_e", confidence_params_.sigma_e);
    confidence_params_.sigma_t = value(fp, "sigma_t", confidence_params_.sigma_t);
    confidence_params_.alpha_edge = value(fp, "alpha_edge", confidence_params_.alpha_edge);
    confidence_params_.w_photo = value(fp, "w_photo", confidence_params_.w_photo);
    confidence_params_.w_tex = value(fp, "w_tex", confidence_params_.w_tex);
    confidence_params_.w_grad = value(fp, "w_grad", confidence_params_.w_grad);
    confidence_params_.w_tmp = value(fp, "w_tmp", confidence_params_.w_tmp);
    confidence_params_.conf_threshold = value(fp, "conf_threshold", confidence_params_.conf_threshold);
    confidence_params_.weight_gamma = value(fp, "weight_gamma", confidence_params_.weight_gamma);
    register_iterations_ = value(fp, "register_refine_iterations", 5);
    track_iterations_ = value(fp, "track_refine_iterations", 2);
    filter_params_.fps = value(fp, "filter_fps", 20.0);
    filter_params_.q_pos = value(fp, "filter_q_pos", 1e-4);
    filter_params_.q_rot = value(fp, "filter_q_rot", 1e-3);
    filter_params_.q_vel = value(fp, "filter_q_vel", 1e-2);
    filter_params_.q_omega = value(fp, "filter_q_omega", 1e-1);
    filter_params_.sigma_pos_obs = value(fp, "filter_sigma_pos_obs", 0.005);
    filter_params_.sigma_rot_obs = value(fp, "filter_sigma_rot_obs", 0.02);
    filter_params_.sigma_pos_motion = value(fp, "filter_sigma_pos_motion", 0.05);
    filter_params_.sigma_rot_motion = value(fp, "filter_sigma_rot_motion", 0.7);
    filter_params_.min_quality = value(fp, "filter_min_quality", 0.01);
    filter_params_.gate_threshold = value(fp, "filter_gate_threshold", 100.0);
    filter_params_.reject_cov_increase = value(fp, "filter_reject_cov_increase", 1e-4);
    filter_params_.enable_gating = value(fp, "filter_enable_gating", true);
    filter_params_.enable_adaptive_R = value(fp, "filter_enable_adaptive_R", true);
    stereo_ = std::make_unique<FastFoundationStereoEstimator>(
        engine_paths[0], engine_paths[1], model_height_, model_width_);
    const auto refiner = fp["refiner_engine_path"].as<std::string>();
    const auto scorer = fp["scorer_engine_path"].as<std::string>();
    const auto mask_dir = value(fp, "mask_image_directory", std::string("masks"));
    std::filesystem::create_directories(mask_dir);
    std::vector<std::shared_ptr<detection_6d::BaseMeshLoader>> model_meshes;
    for (std::size_t i = 0; i < names.size(); ++i) {
      Object object;
      object.name = names[i]; object.mask_file = (std::filesystem::path(mask_dir) / mask_names[i]).string();
      object.color = cv::Scalar((i * 71 + 50) % 255, (i * 97 + 150) % 255, (i * 53 + 220) % 255);
      object.mesh = detection_6d::CreateAssimpMeshLoader(names[i], meshes[i]);
      model_meshes.push_back(object.mesh);
      grounded_sam::PipelineOptions options;
      options.dino_engine = dino_engines[i];
      options.sam_encoder_engine = sam["sam_encoder_engine"].as<std::string>();
      options.sam_decoder_engine = sam["sam_decoder_engine"].as<std::string>();
      options.box_threshold = value(sam, "box_threshold", 0.3f);
      options.max_detections = value(sam, "max_detections", 1);
      object.sam = std::make_unique<grounded_sam::GroundedSamPipeline>(options);
      object.filter = adaptive_pose::Filter(filter_params_);
      objects_.push_back(std::move(object));
    }
    // FoundationPose accepts multiple named meshes. Share its TensorRT contexts across
    // targets so a multi-object scene does not allocate one 252-pose context per object.
    model_ = detection_6d::CreateFoundationPoseModel(pose_core(refiner, false),
        pose_core(scorer, true), model_meshes, intrinsic_,
        value(fp, "max_input_image_height", 480), value(fp, "max_input_image_width", 640));
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error("UDP socket failed");
    int buffer_bytes = 16 * 1024 * 1024;
    setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes));
    sockaddr_in local{}; local.sin_family = AF_INET; local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    local.sin_port = htons(listen_port);
    if (bind(fd_, reinterpret_cast<sockaddr*>(&local), sizeof(local))) throw std::runtime_error("UDP bind failed");
    app_.sin_family = AF_INET; app_.sin_addr.s_addr = htonl(INADDR_LOOPBACK); app_.sin_port = htons(app_port);
    std::fprintf(stderr, "ground tracker listening on 127.0.0.1:%u\n", listen_port);
  }
  ~GroundTrackerUdp() { if (fd_ >= 0) close(fd_); }

  void run() {
    std::uint8_t packet[udp_wire::header_size + udp_wire::chunk_size];
    while (true) {
      sockaddr_in from{}; socklen_t length = sizeof(from);
      auto n = recvfrom(fd_, packet, sizeof(packet), 0, reinterpret_cast<sockaddr*>(&from), &length);
      if (n <= 0) continue;
      udp_wire::Message message{};
      if (!reassembler_.push(packet, n, from.sin_addr.s_addr, from.sin_port, &message) ||
          message.kind != udp_wire::stereo_png) continue;
      try { process(message); }
      catch (const std::exception& e) { std::fprintf(stderr, "tracker frame %u: %s\n", message.id, e.what()); }
      reassembler_.expire();
    }
  }

 private:
  cv::Mat depth_from_stereo(const cv::Mat& left, const cv::Mat& right) {
    if (!maps_ready_ || maps_.image_size != left.size()) {
      std::string error;
      if (!ComputeStereoRectificationMaps(calibration_, left.size(), &maps_, &error))
        throw std::runtime_error(error);
      maps_ready_ = true;
    }
    cv::Mat l, r;
    cv::remap(left, l, maps_.left_map_x, maps_.left_map_y, cv::INTER_LINEAR);
    cv::remap(right, r, maps_.right_map_x, maps_.right_map_y, cv::INTER_LINEAR);
    auto prepared_l = PrepareStereoInput(l, model_width_, model_height_);
    auto prepared_r = PrepareStereoInput(r, model_width_, model_height_);
    cv::Mat disparity;
    if (!stereo_->inference(prepared_l.image, prepared_r.image, disparity))
      throw std::runtime_error("FFS inference failed");
    auto k = ComputePreparedIntrinsics(maps_, prepared_l);
    auto baseline = ComputeRectifiedBaselineMeters(maps_);
    auto model_depth = DisparityToDepthMeters(disparity, float(k.fx), float(baseline),
                                              float(min_depth_), float(max_depth_));
    cv::Mat depth_for_restore = model_depth;
    if (enable_depth_confidence_filter_) {
      const auto confidence = depth_confidence::compute_confidence_map(
          prepared_l.image, prepared_r.image, model_depth, float(k.fx), float(k.fy),
          float(k.cx), float(k.cy), float(baseline), previous_depth_, confidence_params_);
      if (confidence.empty()) throw std::runtime_error("depth confidence computation failed");
      depth_for_restore = depth_confidence::filter_depth_with_confidence(
          model_depth, confidence, confidence_params_.conf_threshold);
      previous_depth_ = model_depth.clone();
    }
    auto rectified_depth = RestoreToRectifiedResolution(depth_for_restore, prepared_l, left.size());
    return AlignRectifiedDepthToOriginalLeft(rectified_depth, maps_, cv::INTER_NEAREST);
  }

  void process(const udp_wire::Message& message) {
    if (last_frame_id_ && message.id == 0) {
      previous_depth_.release();
      for (auto& object : objects_) {
        object.registered = false;
        object.mask.release();
        object.pose.setIdentity();
        object.last_stamp = 0;
        object.filter = adaptive_pose::Filter(filter_params_);
      }
      std::fprintf(stderr, "renderer restarted; target registration reset\n");
    } else if (last_frame_id_ && message.id <= *last_frame_id_) {
      return;
    }
    last_frame_id_ = message.id;
    auto bytes = cv::Mat(1, message.payload.size(), CV_8UC1, const_cast<std::uint8_t*>(message.payload.data()));
    auto stereo = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (stereo.empty() || stereo.cols % 2 || stereo.rows <= 0) throw std::runtime_error("bad stereo PNG");
    auto width = stereo.cols / 2, height = stereo.rows;
    cv::Mat left = stereo(cv::Rect(0, 0, width, height)).clone();
    cv::Mat right = stereo(cv::Rect(width, 0, width, height)).clone();
    cv::Mat rgb; cv::cvtColor(left, rgb, cv::COLOR_BGR2RGB);
    auto depth = depth_from_stereo(left, right);
    if (depth.size() != left.size()) throw std::runtime_error("depth/image size mismatch");
    for (auto& object : objects_) {
      if (!object.registered) {
        if (object.mask.empty()) {
          object.mask = object.sam->run(rgb);
          if (object.mask.empty() || cv::countNonZero(object.mask) == 0) continue;
          cv::imwrite(object.mask_file, object.mask);
        }
        Eigen::Matrix4f pose;
        if (model_->Register(rgb, depth, object.mask, object.name, pose, register_iterations_)) {
          object.filter.Initialize(pose); object.pose = pose;
          object.registered = true; object.last_stamp = message.stamp_ns;
        }
      } else {
        auto dt = double(message.stamp_ns - object.last_stamp) / 1e9;
        if (message.stamp_ns <= object.last_stamp) dt = 1.0 / filter_params_.fps;
        auto predicted = object.filter.Predict(dt);
        Eigen::Matrix4f observed;
        bool ok = model_->Track(rgb, depth, predicted, object.name, observed, track_iterations_);
        object.pose = ok ? object.filter.Update(observed) : object.filter.CommitPrediction();
        object.last_stamp = message.stamp_ns;
      }
    }
    cv::Mat tracking = left.clone();
    for (auto& object : objects_) if (object.registered) {
      auto box_pose = detection_6d::ConvertPoseMesh2BBox(object.pose, object.mesh);
      Draw3DBoundingBox(intrinsic_, box_pose, object.mesh->GetObjectDimension(),
                        object.color, object.name, tracking);
    }
    cv::Mat scaled, colored;
    depth.convertTo(scaled, CV_8UC1, 255.0 / 4.7, -0.3 * 255.0 / 4.7);
    cv::applyColorMap(scaled, colored, cv::COLORMAP_TURBO);
    cv::Mat combined(height, width * 3, CV_8UC3);
    left.copyTo(combined(cv::Rect(0, 0, width, height)));
    tracking.copyTo(combined(cv::Rect(width, 0, width, height)));
    colored.copyTo(combined(cv::Rect(width * 2, 0, width, height)));
    std::vector<std::uint8_t> result;
    udp_wire::put16(result, objects_.size());
    for (const auto& object : objects_) {
      udp_wire::put16(result, object.name.size());
      result.insert(result.end(), object.name.begin(), object.name.end());
      result.push_back(object.registered ? 1 : 0);
      Eigen::Quaternionf q(object.pose.block<3, 3>(0, 0)); q.normalize();
      for (int axis = 0; axis < 3; ++axis) put_double(result, object.pose(axis, 3));
      for (double component : {double(q.x()), double(q.y()), double(q.z()), double(q.w())}) put_double(result, component);
    }
    // The monitor image is only for display. JPEG keeps each local UDP burst
    // below typical Linux receive-buffer limits; masks remain lossless PNG.
    std::vector<std::uint8_t> combined_image;
    if (!cv::imencode(".jpg", combined, combined_image, {cv::IMWRITE_JPEG_QUALITY, 82}))
      throw std::runtime_error("cannot encode monitor image");
    udp_wire::put32(result, combined_image.size());
    result.insert(result.end(), combined_image.begin(), combined_image.end());
    udp_wire::put16(result, objects_.size());
    for (const auto& object : objects_) {
      std::vector<std::uint8_t> mask_png;
      if (!object.mask.empty()) cv::imencode(".png", object.mask, mask_png);
      udp_wire::put32(result, mask_png.size());
      result.insert(result.end(), mask_png.begin(), mask_png.end());
    }
    const auto result_size = result.size();
    udp_wire::send(fd_, app_, {udp_wire::result, message.id, message.stamp_ns, std::move(result)});
    if (message.id == 0 || message.id % 30 == 0)
      std::fprintf(stderr, "tracked frame %u, result %zu bytes\n", message.id, result_size);
  }

  int fd_{-1}, model_width_{640}, model_height_{448}, register_iterations_{5}, track_iterations_{2};
  double min_depth_{0.1}, max_depth_{100.0};
  sockaddr_in app_{};
  udp_wire::Reassembler reassembler_;
  StereoCalibration calibration_{};
  StereoRectificationMaps maps_{};
  bool maps_ready_{false};
  bool enable_depth_confidence_filter_{false};
  depth_confidence::ConfidenceParameters confidence_params_{};
  cv::Mat previous_depth_;
  Eigen::Matrix3f intrinsic_{Eigen::Matrix3f::Identity()};
  adaptive_pose::Params filter_params_{};
  std::unique_ptr<FastFoundationStereoEstimator> stereo_;
  std::shared_ptr<detection_6d::Base6DofDetectionModel> model_;
  std::vector<Object> objects_;
  std::optional<std::uint32_t> last_frame_id_;
};

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::fprintf(stderr, "usage: %s <tracker_yaml> [listen_port=50100] [app_port=50101]\n", argv[0]);
    return 2;
  }
  try {
    GroundTrackerUdp tracker(argv[1], argc > 2 ? std::stoi(argv[2]) : 50100,
                             argc > 3 ? std::stoi(argv[3]) : 50101);
    tracker.run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ground tracker: %s\n", e.what()); return 1;
  }
}
