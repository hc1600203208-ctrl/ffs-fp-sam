#ifndef GWC_VOLUME_KERNEL_H
#define GWC_VOLUME_KERNEL_H

#include <cstdint>

#include <cuda_fp16.h>

void LaunchGwcVolumeKernel(
    const half* d_refimg_fea,
    const half* d_targetimg_fea,
    half* d_cost_volume,
    int B, int C, int H, int W,
    int D, int G,
    bool normalize,
    bool reverse_shift,
    cudaStream_t stream);


void convertFloatToHalf(
    const float* d_floatData,
    half* d_halfData,
    size_t size,
    cudaStream_t stream);

void LaunchPreprocessKernel(
    const uint8_t* d_src, float* d_dst, 
    int width, int height, cudaStream_t stream);

#endif // GWC_VOLUME_KERNEL_H
