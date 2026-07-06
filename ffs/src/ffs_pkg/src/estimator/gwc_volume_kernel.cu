#include "gwc_volume_kernel.h"

#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>


__global__ void preprocess_rgb_to_planar_kernel(
    const uint8_t* __restrict__ src, 
    float* __restrict__ dst, 
    int width, int height, 
    float scale,
    float meanR, float meanG, float meanB,
    float stdR, float stdG, float stdB) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;
    int area = width * height;

    // OpenCV stores images in BGR interleaved order.
    float b = static_cast<float>(src[src_idx + 0]);
    float g = static_cast<float>(src[src_idx + 1]);
    float r = static_cast<float>(src[src_idx + 2]);

    // Destination indices for Planar RGB
    int dst_idx_r = y * width + x;
    int dst_idx_g = area + dst_idx_r;
    int dst_idx_b = area * 2 + dst_idx_r;

    // Convert, Scale, and Normalize
    // If you don't want normalization, set means to 0 and stds to 1.
    dst[dst_idx_r] = (r * scale - meanR) / stdR;
    dst[dst_idx_g] = (g * scale - meanG) / stdG;
    dst[dst_idx_b] = (b * scale - meanB) / stdB;
}

// Wrapper function to call from C++
void LaunchPreprocessKernel(
    const uint8_t* d_src, float* d_dst, 
    int width, int height, cudaStream_t stream) 
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    float scale = 1.0f / 255.0f;
    float meanR = 0.485f;
    float meanG = 0.456f;
    float meanB = 0.406f;
    float stdR = 0.229f;
    float stdG = 0.224f;
    float stdB = 0.225f;

    const char* mode = std::getenv("FFS_PREPROCESS_MODE");
    if (mode != nullptr) {
        if (std::strcmp(mode, "raw") == 0) {
            scale = 1.0f;
            meanR = meanG = meanB = 0.0f;
            stdR = stdG = stdB = 1.0f;
        } else if (std::strcmp(mode, "zero_to_one") == 0) {
            scale = 1.0f / 255.0f;
            meanR = meanG = meanB = 0.0f;
            stdR = stdG = stdB = 1.0f;
        }
    }

    preprocess_rgb_to_planar_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, width, height, scale, meanR, meanG, meanB, stdR, stdG, stdB);
}


// CUDA kernel for float to half conversion
__global__ void floatToHalfKernel(const float* input, half* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

void convertFloatToHalf(
    const float* d_floatData,
    half* d_halfData,
    size_t size,
    cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    floatToHalfKernel<<<gridSize, blockSize, 0, stream>>>(d_floatData, d_halfData, size);
}

// Simplified kernel for batch=1
__global__ void gwc_volume_kernel_simple(
    const half* refimg_fea,    // [C, H, W]
    const half* targetimg_fea, // [C, H, W]
    half* cost_volume,         // [G, D, H, W]
    int C, int H, int W,
    int D, int G,
    bool normalize,
    bool reverse_shift)
{
    int C_g = C / G;
    
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z % D;
    int g = blockIdx.z / D;
    
    if (g >= G || d >= D || h >= H || w >= W)
        return;
    
    float dot_product = 0.0f;
    int tgt_w = reverse_shift ? w + d : w - d;
    
    if (tgt_w >= 0 && tgt_w < W) {
        // Compute dot product
        for (int cg = 0; cg < C_g; ++cg) {
            int c = g * C_g + cg;
            
            int ref_idx = ((c * H) + h) * W + w;
            int tgt_idx = ((c * H) + h) * W + tgt_w;
            
            dot_product += __half2float(refimg_fea[ref_idx]) * __half2float(targetimg_fea[tgt_idx]);
        }

        // Group-wise correlation is defined as the mean correlation inside each group.
        dot_product /= static_cast<float>(C_g);

        // Normalize
        if (normalize) {
            float ref_norm = 0.0f;
            float tgt_norm = 0.0f;
            
            for (int cg = 0; cg < C_g; ++cg) {
                int c = g * C_g + cg;
                int ref_idx = ((c * H) + h) * W + w;
                int tgt_idx = ((c * H) + h) * W + tgt_w;
                
                const float ref_val = __half2float(refimg_fea[ref_idx]);
                const float tgt_val = __half2float(targetimg_fea[tgt_idx]);
                ref_norm += ref_val * ref_val;
                tgt_norm += tgt_val * tgt_val;
            }
            
            ref_norm = sqrtf(ref_norm);
            tgt_norm = sqrtf(tgt_norm);
            const float epsilon = 1e-4f;
            
            if (ref_norm > epsilon && tgt_norm > epsilon) {
                dot_product = dot_product / (ref_norm * tgt_norm);
            } else {
                dot_product = 0.0f;
            }
        }
    }
    
    int out_idx = (((g * D) + d) * H + h) * W + w;
    cost_volume[out_idx] = __float2half(dot_product);
}

// Launch function for your specific use case
void LaunchGwcVolumeKernel(
    const half* d_refimg_fea,
    const half* d_targetimg_fea,
    half* d_cost_volume,
    int B, int C, int H, int W,
    int D, int G,
    bool normalize,
    bool reverse_shift,
    cudaStream_t stream)
{
    // Assert batch size is 1
    if (B != 1) {
        //printf("Warning: Batch size should be 1, got %d\n", B);
        return;
    }
    
    // Set up grid and block dimensions
    dim3 block(16, 16, 1);  // 16x16 = 256 threads per block
    
    dim3 grid(
        (W + block.x - 1) / block.x,   // W dimension
        (H + block.y - 1) / block.y,   // H dimension
        D * G                           // Disparity * Groups dimension
    );
    
    // Launch kernel
    gwc_volume_kernel_simple<<<grid, block, 0, stream>>>(
        d_refimg_fea, d_targetimg_fea, d_cost_volume,
        C, H, W, D, G, normalize, reverse_shift
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        //printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
}
