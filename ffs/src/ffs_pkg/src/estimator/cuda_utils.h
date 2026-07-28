#ifndef TRTX_CUDA_UTILS_H_
#define TRTX_CUDA_UTILS_H_

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#ifndef CHECK_CUDA
#define CHECK_CUDA(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            throw std::runtime_error(std::string("CUDA error ") +\
                std::to_string(static_cast<int>(error_code)) + " (" +\
                cudaGetErrorName(error_code) + "): " + cudaGetErrorString(error_code) +\
                " at " + __FILE__ + ":" + std::to_string(__LINE__));\
        }\
    }
#endif  // CHECK_CUDA

#endif  // TRTX_CUDA_UTILS_H_
