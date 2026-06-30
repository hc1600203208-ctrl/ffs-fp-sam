#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class FoundationStereoEstimator {
public:
    struct Config {
        std::string enginePath;
        // The model input size (must match the TRT engine build size)
        int inputWidth = 736;
        int inputHeight = 320; 
    };

    FoundationStereoEstimator(const Config& config);
    ~FoundationStereoEstimator();

    bool init();
    
    /**
     * @brief Run inference on stereo pair
     * @param leftImg Input left image (Grayscale or RGB)
     * @param rightImg Input right image (Grayscale or RGB)
     * @param outputDisparity Output disparity map (CV_32FC1)
     * @return true if successful
     */
    bool inference(const cv::Mat& leftImg, const cv::Mat& rightImg, cv::Mat& outputDisparity);

private:
    // Safer preprocessing using OpenCV DNN blob
    void preprocess(const cv::Mat& img, float* gpu_buffer, cudaStream_t stream);
    std::vector<char> loadEngineFile(const std::string& filename);

    Config mConfig;
    
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    cudaStream_t mStream = nullptr;

    void* mDeviceLeft = nullptr;
    void* mDeviceRight = nullptr;
    void* mDeviceOutput = nullptr;

    std::string mInputNameLeft;
    std::string mInputNameRight;
    std::string mOutputName;

    size_t mInputByteSize;
    size_t mOutputByteSize;
};