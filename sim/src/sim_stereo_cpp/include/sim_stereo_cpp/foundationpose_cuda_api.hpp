#pragma once

#include <cstdint>

#include "cuda_runtime.h"
#include "nvdiffrast/common/cudaraster/CudaRaster.hpp"

namespace foundationpose_render
{

void clamp(cudaStream_t stream, float *input, float min_value, float max_value, int N);

void threshold_and_downscale_pointcloud(cudaStream_t stream,
                                        float *pointcloud_input,
                                        float *pose_array_input,
                                        int N,
                                        int n_points,
                                        float downscale_factor,
                                        float min_depth,
                                        float max_depth);

void concat(cudaStream_t stream,
            float *input_a,
            float *input_b,
            float *output,
            int N,
            int H,
            int W,
            int C1,
            int C2);

void rasterize(cudaStream_t stream,
               CR::CudaRaster *cr,
               float *pos_ptr,
               int32_t *tri_ptr,
               float *out,
               int pos_count,
               int tri_count,
               int H,
               int W,
               int C);

void interpolate(cudaStream_t stream,
                 float *attr_ptr,
                 float *rast_ptr,
                 int32_t *tri_ptr,
                 float *out,
                 int num_vertices,
                 int num_triangles,
                 int attr_shape_dim,
                 int attr_dim,
                 int H,
                 int W,
                 int C);

void texture(cudaStream_t stream,
             float *tex_ptr,
             float *uv_ptr,
             float *out,
             int tex_height,
             int tex_width,
             int tex_channel,
             int tex_depth,
             int H,
             int W,
             int N);

void transform_points(cudaStream_t stream,
                      const float *transform_matrixs,
                      int transform_num,
                      const float *points_vectors,
                      int points_num,
                      float *transformed_points_vectors);

void generate_pose_clip(cudaStream_t stream,
                        const float *transform_matrixs,
                        const float *bbox2d_matrix,
                        int transform_num,
                        const float *points_vectors,
                        int points_num,
                        float *transformed_points_vectors,
                        int rgb_H,
                        int rgb_W);

void transform_normals(cudaStream_t stream,
                       const float *transform_matrixs,
                       int transform_num,
                       const float *normals_vectors,
                       int normals_num,
                       float *transformed_normal_vectors);

void refine_color(cudaStream_t stream,
                  const float *color,
                  const float *diffuse_intensity_map,
                  const float *rast,
                  float *output,
                  int poses_num,
                  float w_ambient,
                  float w_diffuse,
                  int rgb_H,
                  int rgb_W);

} // namespace foundationpose_render

