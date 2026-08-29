#include "sim_stereo_cpp/stereo_mesh_renderer.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace sim_stereo_cpp
{

namespace
{

constexpr float kEps = 1e-6F;

Eigen::Matrix4f TranslateMatrix(const Eigen::Vector3f &translation)
{
  Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
  tf.block<3, 1>(0, 3) = translation;
  return tf;
}

template <typename T>
void CopyToDevice(T *dst, const std::vector<T> &src)
{
  if (src.empty())
  {
    return;
  }
  cudaError_t err = cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    throw std::runtime_error(std::string("cudaMemcpy host->device failed: ") +
                             cudaGetErrorString(err));
  }
}

void CheckCuda(cudaError_t err, const char *what)
{
  if (err != cudaSuccess)
  {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

} // namespace

void StereoMeshRenderer::DeviceDeleter::operator()(void *ptr) const noexcept
{
  if (ptr != nullptr)
  {
    cudaFree(ptr);
  }
}

Eigen::Matrix4f StereoMeshRenderer::ToGlCameraFromCvCamera()
{
  Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
  tf(1, 1) = -1.0F;
  tf(2, 2) = -1.0F;
  return tf;
}

Eigen::Matrix4f StereoMeshRenderer::ToGlPose(const Eigen::Matrix4f &pose_in_cv)
{
  return ToGlCameraFromCvCamera() * pose_in_cv;
}

Eigen::Matrix4f StereoMeshRenderer::BuildProjectionMatrix(const Eigen::Matrix3f &K,
                                                          int height,
                                                          int width)
{
  const float znear = 0.1F;
  const float zfar = 100.0F;
  const float depth = zfar - znear;
  const float q = -(zfar + znear) / depth;
  const float qn = -2.0F * (zfar * znear) / depth;

  Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
  proj << 2.0F * K(0, 0) / width, -2.0F * K(0, 1) / width,
      (-2.0F * K(0, 2) + width) / width, 0.0F,
      0.0F, 2.0F * K(1, 1) / height, (2.0F * K(1, 2) - height) / height, 0.0F,
      0.0F, 0.0F, q, qn,
      0.0F, 0.0F, -1.0F, 0.0F;
  return proj;
}

StereoMeshRenderer::StereoMeshRenderer(std::shared_ptr<detection_6d::BaseMeshLoader> mesh_loader,
                                       int width,
                                       int height)
    : width_(width), height_(height)
{
  if (mesh_loader == nullptr)
  {
    throw std::invalid_argument("mesh_loader is null");
  }

  if (!LoadMesh(mesh_loader))
  {
    throw std::runtime_error("Failed to load mesh");
  }

  if (!AllocateBuffers())
  {
    throw std::runtime_error("Failed to allocate CUDA buffers");
  }

  CheckCuda(cudaStreamCreate(&cuda_stream_), "cudaStreamCreate");
  cr_ = std::make_shared<CR::CudaRaster>();
}

StereoMeshRenderer::~StereoMeshRenderer()
{
  if (cuda_stream_ != nullptr)
  {
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }
}

bool StereoMeshRenderer::LoadMesh(const std::shared_ptr<detection_6d::BaseMeshLoader> &mesh_loader)
{
  mesh_center_ = mesh_loader->GetMeshModelCenter();
  mesh_diameter_ = mesh_loader->GetMeshDiameter();

  const auto &mesh_vertices = mesh_loader->GetMeshVertices();
  const auto &mesh_normals = mesh_loader->GetMeshVertexNormals();
  const auto &mesh_texcoords = mesh_loader->GetMeshTextureCoords();
  const auto &mesh_faces = mesh_loader->GetMeshTriangleFaces();
  const cv::Mat &texture_map = mesh_loader->GetTextureMap();

  if (texture_map.empty())
  {
    throw std::runtime_error("Mesh texture map is empty");
  }

  texture_map_height_ = texture_map.rows;
  texture_map_width_ = texture_map.cols;

  vertices_.reserve(mesh_vertices.size() * kVertexPoints);
  vertex_normals_.reserve(mesh_normals.size() * kVertexPoints);
  texcoords_.reserve(mesh_texcoords.size() * kTexcoordsDim);
  mesh_faces_.reserve(mesh_faces.size() * kTriangleVertices);

  for (size_t i = 0; i < mesh_vertices.size(); ++i)
  {
    const Eigen::Vector3f centered_v = mesh_vertices[i] - mesh_center_;
    vertices_.push_back(centered_v.x());
    vertices_.push_back(centered_v.y());
    vertices_.push_back(centered_v.z());

    vertex_normals_.push_back(mesh_normals[i].x());
    vertex_normals_.push_back(mesh_normals[i].y());
    vertex_normals_.push_back(mesh_normals[i].z());

    texcoords_.push_back(mesh_texcoords[i].x());
    texcoords_.push_back(1.0F - mesh_texcoords[i].y());
  }

  for (const auto &face : mesh_faces)
  {
    mesh_faces_.push_back(static_cast<int32_t>(face[0]));
    mesh_faces_.push_back(static_cast<int32_t>(face[1]));
    mesh_faces_.push_back(static_cast<int32_t>(face[2]));
  }

  texture_map_float_.resize(static_cast<size_t>(texture_map.rows) * texture_map.cols * kNumChannels);
  for (int y = 0; y < texture_map.rows; ++y)
  {
    const cv::Vec3b *row = texture_map.ptr<cv::Vec3b>(y);
    for (int x = 0; x < texture_map.cols; ++x)
    {
      const cv::Vec3b &pixel = row[x];
      const size_t idx = (static_cast<size_t>(y) * texture_map.cols + x) * kNumChannels;
      texture_map_float_[idx + 0] = static_cast<float>(pixel[0]) / 255.0F;
      texture_map_float_[idx + 1] = static_cast<float>(pixel[1]) / 255.0F;
      texture_map_float_[idx + 2] = static_cast<float>(pixel[2]) / 255.0F;
    }
  }

  return true;
}

bool StereoMeshRenderer::AllocateBuffers()
{
  const size_t vertices_bytes = vertices_.size() * sizeof(float);
  const size_t normals_bytes = vertex_normals_.size() * sizeof(float);
  const size_t texcoords_bytes = texcoords_.size() * sizeof(float);
  const size_t faces_bytes = mesh_faces_.size() * sizeof(int32_t);
  const size_t texture_bytes = texture_map_float_.size() * sizeof(float);

  float *vertices = nullptr;
  float *normals = nullptr;
  float *texcoords = nullptr;
  int32_t *faces = nullptr;
  float *texture = nullptr;

  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&vertices), vertices_bytes),
            "cudaMalloc vertices");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&normals), normals_bytes),
            "cudaMalloc normals");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&texcoords), texcoords_bytes),
            "cudaMalloc texcoords");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&faces), faces_bytes), "cudaMalloc faces");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&texture), texture_bytes),
            "cudaMalloc texture");

  vertices_device_.reset(vertices);
  vertex_normals_device_.reset(normals);
  texcoords_device_.reset(texcoords);
  mesh_faces_device_.reset(faces);
  texture_map_device_.reset(texture);

  CopyToDevice(vertices_device_.get(), vertices_);
  CopyToDevice(vertex_normals_device_.get(), vertex_normals_);
  CopyToDevice(texcoords_device_.get(), texcoords_);
  CopyToDevice(mesh_faces_device_.get(), mesh_faces_);
  CopyToDevice(texture_map_device_.get(), texture_map_float_);

  float *pose_clip = nullptr;
  float *rast_out = nullptr;
  float *diffuse = nullptr;
  float *diffuse_map = nullptr;
  float *texcoords_out = nullptr;
  float *color = nullptr;

  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&pose_clip),
                       static_cast<size_t>(vertices_.size() / 3) * 4 * sizeof(float)),
            "cudaMalloc pose_clip");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&rast_out),
                       static_cast<size_t>(width_) * height_ * 4 * sizeof(float)),
            "cudaMalloc rast_out");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&diffuse),
                       static_cast<size_t>(vertices_.size() / 3) * sizeof(float)),
            "cudaMalloc diffuse");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&diffuse_map),
                       static_cast<size_t>(width_) * height_ * sizeof(float)),
            "cudaMalloc diffuse_map");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&texcoords_out),
                       static_cast<size_t>(width_) * height_ * 2 * sizeof(float)),
            "cudaMalloc texcoords_out");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&color),
                       static_cast<size_t>(width_) * height_ * 3 * sizeof(float)),
            "cudaMalloc color");

  pose_clip_device_.reset(pose_clip);
  rast_out_device_.reset(rast_out);
  diffuse_intensity_device_.reset(diffuse);
  diffuse_intensity_map_device_.reset(diffuse_map);
  texcoords_out_device_.reset(texcoords_out);
  color_device_.reset(color);

  return true;
}

Eigen::Matrix4f StereoMeshRenderer::ConvertPoseToCenteredMesh(const Eigen::Matrix4f &pose_in_mesh) const
{
  return pose_in_mesh * TranslateMatrix(mesh_center_);
}

cv::Mat StereoMeshRenderer::RenderBgr(const Eigen::Matrix4f &pose_in_mesh,
                                      const Eigen::Matrix3f &K)
{
  return RenderSingleBgr(pose_in_mesh, K);
}

cv::Mat StereoMeshRenderer::RenderSingleBgr(const Eigen::Matrix4f &pose_in_mesh,
                                            const Eigen::Matrix3f &K)
{
  const Eigen::Matrix4f centered_pose = ConvertPoseToCenteredMesh(pose_in_mesh);
  const Eigen::Matrix4f proj = BuildProjectionMatrix(K, height_, width_);
  const Eigen::Matrix4f gl_pose = ToGlPose(centered_pose);
  const Eigen::Matrix4f clip_transform = proj * gl_pose;

  Eigen::Matrix<float, 1, 4, Eigen::RowMajor> bbox2d;
  bbox2d << 0.0F, 0.0F, static_cast<float>(width_), static_cast<float>(height_);

  float *clip_transform_device = nullptr;
  float *bbox_device = nullptr;
  float *pose_device = nullptr;

  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&clip_transform_device), 16 * sizeof(float)),
            "cudaMalloc clip_transform");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&bbox_device), 4 * sizeof(float)),
            "cudaMalloc bbox");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&pose_device), 16 * sizeof(float)),
            "cudaMalloc pose");

  CheckCuda(cudaMemcpy(clip_transform_device,
                       clip_transform.data(),
                       16 * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy clip_transform");
  CheckCuda(cudaMemcpy(bbox_device,
                       bbox2d.data(),
                       4 * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy bbox");
  CheckCuda(cudaMemcpy(pose_device,
                       centered_pose.data(),
                       16 * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy pose");

  foundationpose_render::generate_pose_clip(cuda_stream_,
                                            clip_transform_device,
                                            bbox_device,
                                            1,
                                            vertices_device_.get(),
                                            static_cast<int>(vertices_.size() / 3),
                                            pose_clip_device_.get(),
                                            height_,
                                            width_);
  CheckCuda(cudaFree(bbox_device), "cudaFree bbox");
  CheckCuda(cudaFree(clip_transform_device), "cudaFree clip_transform");

  foundationpose_render::rasterize(cuda_stream_,
                                   cr_.get(),
                                   pose_clip_device_.get(),
                                   mesh_faces_device_.get(),
                                   rast_out_device_.get(),
                                   static_cast<int>(vertices_.size() / 3),
                                   static_cast<int>(mesh_faces_.size() / 3),
                                   height_,
                                   width_,
                                   1);

  foundationpose_render::interpolate(cuda_stream_,
                                     texcoords_device_.get(),
                                     rast_out_device_.get(),
                                     mesh_faces_device_.get(),
                                     texcoords_out_device_.get(),
                                     static_cast<int>(vertices_.size() / 3),
                                     static_cast<int>(mesh_faces_.size() / 3),
                                     2,
                                     2,
                                     height_,
                                     width_,
                                     1);

  foundationpose_render::transform_normals(cuda_stream_,
                                           pose_device,
                                           1,
                                           vertex_normals_device_.get(),
                                           static_cast<int>(vertices_.size() / 3),
                                           diffuse_intensity_device_.get());

  foundationpose_render::interpolate(cuda_stream_,
                                     diffuse_intensity_device_.get(),
                                     rast_out_device_.get(),
                                     mesh_faces_device_.get(),
                                     diffuse_intensity_map_device_.get(),
                                     static_cast<int>(vertices_.size() / 3),
                                     static_cast<int>(mesh_faces_.size() / 3),
                                     3,
                                     1,
                                     height_,
                                     width_,
                                     1);

  foundationpose_render::texture(cuda_stream_,
                                 texture_map_device_.get(),
                                 texcoords_out_device_.get(),
                                 color_device_.get(),
                                 texture_map_height_,
                                 texture_map_width_,
                                 kNumChannels,
                                 1,
                                 height_,
                                 width_,
                                 1);

  foundationpose_render::refine_color(cuda_stream_,
                                      color_device_.get(),
                                      diffuse_intensity_map_device_.get(),
                                      rast_out_device_.get(),
                                      color_device_.get(),
                                      1,
                                      0.8F,
                                      0.5F,
                                      height_,
                                      width_);

  foundationpose_render::clamp(cuda_stream_, color_device_.get(), 0.0F, 1.0F,
                               height_ * width_ * 3);

  CheckCuda(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize render");
  CheckCuda(cudaFree(pose_device), "cudaFree pose");

  std::vector<float> host_rgb(static_cast<size_t>(height_) * width_ * 3);
  CheckCuda(cudaMemcpy(host_rgb.data(),
                       color_device_.get(),
                       host_rgb.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy color device->host");

  cv::Mat rgb_float(height_, width_, CV_32FC3, host_rgb.data());
  cv::Mat flipped_float;
  cv::flip(rgb_float, flipped_float, 0);

  cv::Mat rgb_8u;
  flipped_float.convertTo(rgb_8u, CV_8UC3, 255.0);
  cv::Mat bgr_8u;
  cv::cvtColor(rgb_8u, bgr_8u, cv::COLOR_RGB2BGR);
  return bgr_8u.clone();
}

} // namespace sim_stereo_cpp
