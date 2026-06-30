#include "foundation_stereo_estimator.h"
#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp> // Required for blobFromImage

// --- Logger ---
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cout << "[TRT Error] " << msg << std::endl;
        }
    }
} gLogger;

#define CHECK_CUDA(status) \
    do { \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at line " << __LINE__ << ": " \
                      << cudaGetErrorString(status) << std::endl; \
        } \
    } while (0)

FoundationStereoEstimator::FoundationStereoEstimator(const Config& config) 
    : mConfig(config) {
}

FoundationStereoEstimator::~FoundationStereoEstimator() {
    if (mDeviceLeft) cudaFree(mDeviceLeft);
    if (mDeviceRight) cudaFree(mDeviceRight);
    if (mDeviceOutput) cudaFree(mDeviceOutput);
    if (mStream) cudaStreamDestroy(mStream);
}

std::vector<char> FoundationStereoEstimator::loadEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.good()) return {};
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

bool FoundationStereoEstimator::init() {
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!mRuntime) return false;

    auto engineData = loadEngineFile(mConfig.enginePath);
    if (engineData.empty()) {
        std::cerr << "Error: Engine file not found: " << mConfig.enginePath << std::endl;
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!mEngine) return false;

    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) return false;

    // --- IO Name Detection ---
    // Heuristic: Input with "left" is Left, "right" is Right.
    // Fallback: Index 0 is Left, Index 1 is Right.
    int nIO = mEngine->getNbIOTensors();
    std::vector<std::string> inputs;
    for (int i = 0; i < nIO; ++i) {
        const char* name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputs.push_back(name);
        } else {
            mOutputName = name;
        }
    }

    if (inputs.size() < 2) {
        std::cerr << "Error: Engine must have at least 2 inputs." << std::endl;
        return false;
    }

    // Attempt to identify by name
    for (const auto& name : inputs) {
        std::string lower = name; 
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("left") != std::string::npos) mInputNameLeft = name;
        else if (lower.find("right") != std::string::npos) mInputNameRight = name;
    }

    // Fallback if names are generic (e.g. "input_0", "input_1")
    if (mInputNameLeft.empty()) mInputNameLeft = inputs[0];
    if (mInputNameRight.empty()) mInputNameRight = inputs[1];

    std::cout << "Bindings -> Left: " << mInputNameLeft 
              << ", Right: " << mInputNameRight 
              << ", Output: " << mOutputName << std::endl;

    // Allocate Memory
    // Model expects 3 channels (RGB)
    size_t numPixels = mConfig.inputWidth * mConfig.inputHeight;
    mInputByteSize = numPixels * 3 * sizeof(float);
    mOutputByteSize = numPixels * 1 * sizeof(float); 

    CHECK_CUDA(cudaMalloc(&mDeviceLeft, mInputByteSize));
    CHECK_CUDA(cudaMalloc(&mDeviceRight, mInputByteSize));
    CHECK_CUDA(cudaMalloc(&mDeviceOutput, mOutputByteSize));
    CHECK_CUDA(cudaStreamCreate(&mStream));

    // Bind addresses (TRT 10.x style)
    if (!mContext->setTensorAddress(mInputNameLeft.c_str(), mDeviceLeft)) return false;
    if (!mContext->setTensorAddress(mInputNameRight.c_str(), mDeviceRight)) return false;
    if (!mContext->setTensorAddress(mOutputName.c_str(), mDeviceOutput)) return false;

    return true;
}

void FoundationStereoEstimator::preprocess(const cv::Mat& img, float* gpu_buffer, cudaStream_t stream) {
    // 1. Handle Grayscale -> BGR (so blobFromImage can handle it uniformly)
    cv::Mat inputC3;
    if (img.channels() == 1) {
        cv::cvtColor(img, inputC3, cv::COLOR_GRAY2RGB);
    } else {
        inputC3 = img;
    }

    // 2. Use blobFromImage for Safe Preprocessing
    // FoundationStereo / DepthAnything Config:
    // - Size: (inputWidth, inputHeight)
    // - Mean: [0.485, 0.456, 0.406] * 255
    // - Std:  [0.229, 0.224, 0.225] * 255
    // Formula: (x - mean) / std
    // blobFromImage does (x - mean) * scale.
    // So scale should be 1/std.
    
    // Mean in BGR order (0.406, 0.456, 0.485) * 255 => (103.53, 116.28, 123.675)
    // Std in BGR order (0.225, 0.224, 0.229) * 255 => (57.375, 57.12, 58.395)
    
    // However, blobFromImage only supports a SINGLE scalar scale factor for all channels.
    // Since std is slightly different per channel, we cannot use a single scale factor perfectly.
    // We must do it manually or assume approx std.
    // BETTER APPROACH: Normalize to 0-1 float first, then manual sub/div.

    cv::Mat resized;
    cv::resize(inputC3, resized, cv::Size(mConfig.inputWidth, mConfig.inputHeight));

    // Convert to Float 0..1
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    // Create Blob (NCHW)
    // We can use the safe split method to ensure NCHW layout
    float mean[] = {0.485f, 0.456f, 0.406f}; // RGB
    float std[]  = {0.229f, 0.224f, 0.225f}; // RGB


    std::vector<float> host_buffer(3 * mConfig.inputWidth * mConfig.inputHeight);
    
    
    int height = mConfig.inputHeight;
    int width = mConfig.inputWidth;
    //int area = height * width;

    /*
    // Safe Iterator Loop (Handles Padding and Layout correctly)
    // Iterate over rows to respect stride
    for (int h = 0; h < height; ++h) {
        const float* row_ptr = float_img.ptr<float>(h);
        for (int w = 0; w < width; ++w) {
            // OpenCV is BGR
            float b = row_ptr[w * 3 + 0];
            float g = row_ptr[w * 3 + 1];
            float r = row_ptr[w * 3 + 2];

            // R Channel (Plane 0)
            host_buffer[0 * area + h * width + w] = (r - mean[0]) / std[0];
            // G Channel (Plane 1)
            host_buffer[1 * area + h * width + w] = (g - mean[1]) / std[1];
            // B Channel (Plane 2)
            host_buffer[2 * area + h * width + w] = (b - mean[2]) / std[2];
        }
    }
    */


    // Convert HWC (OpenCV BGR) to CHW (Tensor RGB)
    int offset_g = height * width * 1;
    int offset_b = height * width * 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // OpenCV uses BGR
            cv::Vec3b pixel = float_img.at<cv::Vec3b>(y, x);
            
            // Normalize to [0, 1]
            //float r = pixel[2] / 255.0f;
            //float g = pixel[1] / 255.0f;
            //float b = pixel[0] / 255.0f;

            // Normalize with Mean/Std and assign to planar buffer
            //dstBuffer[y * m_inputW + x]            = (r - mean[0]) / std[0]; // R plane
            //dstBuffer[offset_g + y * m_inputW + x] = (g - mean[1]) / std[1]; // G plane
            //dstBuffer[offset_b + y * m_inputW + x] = (b - mean[2]) / std[2]; // B plane

            host_buffer[y * width + x] = pixel[0];
            host_buffer[offset_g + y * width + x] = pixel[1];
            host_buffer[offset_b + y * width + x] = pixel[2];
        }
    }

    CHECK_CUDA(cudaMemcpyAsync(gpu_buffer, host_buffer.data(), mInputByteSize, cudaMemcpyHostToDevice, stream));
}

bool FoundationStereoEstimator::inference(const cv::Mat& leftImg, const cv::Mat& rightImg, cv::Mat& outputDisparity) {
    if (leftImg.empty() || rightImg.empty()) return false;

    // Preprocess
    preprocess(leftImg, (float*) mDeviceLeft, mStream);
    preprocess(rightImg, (float*) mDeviceRight, mStream);

    // Enqueue
    if (!mContext->enqueueV3(mStream)) {
        return false;
    }

    // Download
    std::vector<float> host_output(mConfig.inputWidth * mConfig.inputHeight);
    CHECK_CUDA(cudaMemcpyAsync(host_output.data(), mDeviceOutput, mOutputByteSize, cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);

    // Validation: Check for NaNs (Common issue with FP16 TRT models)
    int nan_count = 0;
    for (float v : host_output) {
        if (std::isnan(v)) nan_count++;
    }
    if (nan_count > 0) {
        std::cerr << "Warning: Output contains " << nan_count << " NaNs. Inference might be invalid." << std::endl;
        return false;
    }

    // Convert to Mat
    cv::Mat raw_disp(mConfig.inputHeight, mConfig.inputWidth, CV_32FC1, host_output.data());

    // Resize back to original
    int origW = leftImg.cols;
    int origH = leftImg.rows;
    cv::resize(raw_disp, outputDisparity, cv::Size(origW, origH), 0, 0, cv::INTER_LINEAR);

    // Scale Disparity Values
    float scale = (float)origW / (float)mConfig.inputWidth;
    outputDisparity = outputDisparity * scale;

    return true;
}