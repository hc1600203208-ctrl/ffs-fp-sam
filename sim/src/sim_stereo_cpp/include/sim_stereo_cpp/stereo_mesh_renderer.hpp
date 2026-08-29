#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "cuda_runtime.h"
#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "nvdiffrast/common/cudaraster/CudaRaster.hpp"

#include "sim_stereo_cpp/foundationpose_cuda_api.hpp"

namespace sim_stereo_cpp
{

class StereoMeshRenderer
{
public:
  StereoMeshRenderer(std::shared_ptr<detection_6d::BaseMeshLoader> mesh_loader,
                     int width,
                     int height);

  ~StereoMeshRenderer();

  StereoMeshRenderer(const StereoMeshRenderer &) = delete;
  StereoMeshRenderer &operator=(const StereoMeshRenderer &) = delete;

  cv::Mat RenderBgr(const Eigen::Matrix4f &pose_in_mesh, const Eigen::Matrix3f &K);

private:
  static constexpr int kNumChannels = 3;
  static constexpr int kTexcoordsDim = 2;
  static constexpr int kVertexPoints = 3;
  static constexpr int kTriangleVertices = 3;

  struct DeviceDeleter
  {
    void operator()(void *ptr) const noexcept;
  };

  template <typename T>
  using DevicePtr = std::unique_ptr<T, DeviceDeleter>;

  static Eigen::Matrix4f BuildProjectionMatrix(const Eigen::Matrix3f &K,
                                               int height,
                                               int width);
  static Eigen::Matrix4f ToGlCameraFromCvCamera();
  static Eigen::Matrix4f ToGlPose(const Eigen::Matrix4f &pose_in_cv);

  bool LoadMesh(const std::shared_ptr<detection_6d::BaseMeshLoader> &mesh_loader);
  bool AllocateBuffers();
  cv::Mat RenderSingleBgr(const Eigen::Matrix4f &pose_in_mesh, const Eigen::Matrix3f &K);
  Eigen::Matrix4f ConvertPoseToCenteredMesh(const Eigen::Matrix4f &pose_in_mesh) const;

  int width_ = 0;
  int height_ = 0;

  Eigen::Vector3f mesh_center_ = Eigen::Vector3f::Zero();
  float mesh_diameter_ = 0.0F;

  std::vector<float> vertices_;
  std::vector<float> vertex_normals_;
  std::vector<float> texcoords_;
  std::vector<int32_t> mesh_faces_;
  std::vector<float> texture_map_float_;
  int texture_map_width_ = 0;
  int texture_map_height_ = 0;

  DevicePtr<float> vertices_device_{nullptr};
  DevicePtr<float> vertex_normals_device_{nullptr};
  DevicePtr<float> texcoords_device_{nullptr};
  DevicePtr<int32_t> mesh_faces_device_{nullptr};
  DevicePtr<float> texture_map_device_{nullptr};

  DevicePtr<float> pose_clip_device_{nullptr};
  DevicePtr<float> rast_out_device_{nullptr};
  DevicePtr<float> diffuse_intensity_device_{nullptr};
  DevicePtr<float> diffuse_intensity_map_device_{nullptr};
  DevicePtr<float> texcoords_out_device_{nullptr};
  DevicePtr<float> color_device_{nullptr};

  cudaStream_t cuda_stream_{nullptr};
  std::shared_ptr<CR::CudaRaster> cr_;
};

} // namespace sim_stereo_cpp

