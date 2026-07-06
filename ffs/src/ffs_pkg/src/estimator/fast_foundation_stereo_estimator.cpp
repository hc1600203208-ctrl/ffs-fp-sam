#include "fast_foundation_stereo_estimator.h"

#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

#include <rclcpp/rclcpp.hpp>

#include "gwc_volume_kernel.h"

namespace {

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[FastFoundationStereoEstimator] " << msg << std::endl;
        }
    }
} gLogger;

std::vector<char> _loadEngineData(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) throw std::runtime_error("Error reading engine file: " + enginePath);
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

void checkTrtCall(bool ok, const std::string& what) {
    if (!ok) {
        throw std::runtime_error("TensorRT call failed: " + what);
    }
}

bool HasDynamicDim(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

bool HasIOTensor(const nvinfer1::ICudaEngine* engine, const char* tensor_name) {
    if (engine == nullptr || tensor_name == nullptr) {
        return false;
    }

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* current_name = engine->getIOTensorName(i);
        if (current_name != nullptr && std::string(current_name) == tensor_name) {
            return true;
        }
    }
    return false;
}

size_t NumElements(const nvinfer1::Dims& dims) {
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Encountered unresolved or invalid TensorRT dimension.");
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

std::string DimsToString(const nvinfer1::Dims& dims) {
    std::string out = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(dims.d[i]);
    }
    out += "]";
    return out;
}

nvinfer1::Dims4 ExpectNchw4(const nvinfer1::Dims& dims, const std::string& name) {
    if (dims.nbDims != 4) {
        throw std::runtime_error("Expected 4D NCHW tensor for " + name + ", got " + DimsToString(dims));
    }
    return nvinfer1::Dims4(dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
}

void ValidateStaticOrResolvedShape(const std::string& name, const nvinfer1::Dims& engine_dims, const nvinfer1::Dims& actual_dims) {
    if (engine_dims.nbDims != actual_dims.nbDims) {
        throw std::runtime_error(
            "Rank mismatch for " + name + ": engine=" + DimsToString(engine_dims) +
            " actual=" + DimsToString(actual_dims));
    }

    for (int i = 0; i < engine_dims.nbDims; ++i) {
        if (engine_dims.d[i] > 0 && engine_dims.d[i] != actual_dims.d[i]) {
            throw std::runtime_error(
                "Static shape mismatch for " + name + ": engine=" + DimsToString(engine_dims) +
                " actual=" + DimsToString(actual_dims));
        }
    }
}

}  // namespace

FastFoundationStereoEstimator::FastFoundationStereoEstimator(const std::string& featureEnginePath, const std::string& postEnginePath, int height, int width) 
    : m_inputH(height),
      m_inputW(width),
      m_featureH4(height / 4),
      m_featureW4(width / 4),
      m_featureH8(height / 8),
      m_featureW8(width / 8),
      m_featureH16(height / 16),
      m_featureW16(width / 16),
      m_featureH32(height / 32),
      m_featureW32(width / 32),
      m_stemH2(height / 2),
      m_stemW2(width / 2),
      m_maxDispQuarter(0),
      m_featuresLeft04Size(0),
      m_featuresLeft08Size(0),
      m_featuresLeft16Size(0),
      m_featuresLeft32Size(0),
      m_featuresRight04Size(0),
      m_stem2xSize(0),
      m_gwcVolumeSize(0) {

    if (m_inputH % 32 != 0 || m_inputW % 32 != 0) {
        throw std::invalid_argument("FastFoundationStereoEstimator expects height/width divisible by 32.");
    }
    
    m_padder = std::make_unique<InputPadder>(m_inputH, m_inputW);

    // 1. Resize Host Buffers (Only what we actually need on CPU)
    m_hDisp.resize(1 * m_inputH * m_inputW);

    CHECK_CUDA(cudaStreamCreate(&m_stream));

    // 2. Load Engine and resolve tensor shapes
    loadEngine(featureEnginePath, postEnginePath);
    allocateBuffers();
    printIOTensors();
}

FastFoundationStereoEstimator::~FastFoundationStereoEstimator() {
    // Note: All DeviceBuffer unique_ptrs automatically call cudaFree!

    if (m_stream) cudaStreamDestroy(m_stream);

    if (m_feature_context) delete m_feature_context;
    if (m_feature_engine) delete m_feature_engine;
    if (m_post_context) delete m_post_context;
    if (m_post_engine) delete m_post_engine;
    if (m_runtime) delete m_runtime;
}

void FastFoundationStereoEstimator::loadEngine(const std::string& featureEnginePath, const std::string& postEnginePath) {

    m_runtime = nvinfer1::createInferRuntime(gLogger);
    if (m_runtime == nullptr) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    auto setupContext = [&](const std::string& path, nvinfer1::ICudaEngine*& engine, nvinfer1::IExecutionContext*& context) {
        auto buf = _loadEngineData(path);
        engine = m_runtime->deserializeCudaEngine(buf.data(), buf.size());
        if (engine == nullptr) {
            throw std::runtime_error(
                "Failed to deserialize TensorRT engine: " + path +
                ". The engine may be built for a different TensorRT version or GPU architecture.");
        }
        context = engine->createExecutionContext();
        if (context == nullptr) {
            throw std::runtime_error("Failed to create TensorRT execution context for engine: " + path);
        }
    };

    // feature runner.
    setupContext(featureEnginePath, m_feature_engine, m_feature_context);
    
    nvinfer1::Dims4 inputDims(1, 3, m_inputH, m_inputW);
    checkTrtCall(m_feature_context->setInputShape("left", inputDims), "setInputShape(feature:left)");
    checkTrtCall(m_feature_context->setInputShape("right", inputDims), "setInputShape(feature:right)");

    const auto featureLeft04Dims = ExpectNchw4(m_feature_context->getTensorShape("features_left_04"), "features_left_04");
    const auto featureLeft08Dims = ExpectNchw4(m_feature_context->getTensorShape("features_left_08"), "features_left_08");
    const auto featureLeft16Dims = ExpectNchw4(m_feature_context->getTensorShape("features_left_16"), "features_left_16");
    const auto featureLeft32Dims = ExpectNchw4(m_feature_context->getTensorShape("features_left_32"), "features_left_32");
    const auto featureRight04Dims = ExpectNchw4(m_feature_context->getTensorShape("features_right_04"), "features_right_04");
    const auto stem2xDims = ExpectNchw4(m_feature_context->getTensorShape("stem_2x"), "stem_2x");

    m_featureLeft04Channels = featureLeft04Dims.d[1];
    m_featureLeft08Channels = featureLeft08Dims.d[1];
    m_featureLeft16Channels = featureLeft16Dims.d[1];
    m_featureLeft32Channels = featureLeft32Dims.d[1];
    m_featureRight04Channels = featureRight04Dims.d[1];
    m_stem2xChannels = stem2xDims.d[1];

    m_featureH4 = featureLeft04Dims.d[2];
    m_featureW4 = featureLeft04Dims.d[3];
    m_featureH8 = featureLeft08Dims.d[2];
    m_featureW8 = featureLeft08Dims.d[3];
    m_featureH16 = featureLeft16Dims.d[2];
    m_featureW16 = featureLeft16Dims.d[3];
    m_featureH32 = featureLeft32Dims.d[2];
    m_featureW32 = featureLeft32Dims.d[3];
    m_stemH2 = stem2xDims.d[2];
    m_stemW2 = stem2xDims.d[3];

    m_featuresLeft04Size = NumElements(featureLeft04Dims);
    m_featuresLeft08Size = NumElements(featureLeft08Dims);
    m_featuresLeft16Size = NumElements(featureLeft16Dims);
    m_featuresLeft32Size = NumElements(featureLeft32Dims);
    m_featuresRight04Size = NumElements(featureRight04Dims);
    m_stem2xSize = NumElements(stem2xDims);

    // post runner.
    setupContext(postEnginePath, m_post_engine, m_post_context);

    auto maybeSetShape = [&](const char* tensor_name, const nvinfer1::Dims& dims) {
        if (!HasIOTensor(m_post_engine, tensor_name)) {
            RCLCPP_WARN(
                rclcpp::get_logger("FastFoundationStereoEstimator"),
                "Post engine does not expose tensor '%s'; skipping shape/address binding for this optional input.",
                tensor_name);
            return false;
        }

        const auto engine_dims = m_post_engine->getTensorShape(tensor_name);
        ValidateStaticOrResolvedShape(tensor_name, engine_dims, dims);
        if (HasDynamicDim(engine_dims)) {
            checkTrtCall(m_post_context->setInputShape(tensor_name, dims), std::string("setInputShape(post:") + tensor_name + ")");
        }
        return true;
    };

    maybeSetShape("features_left_04", featureLeft04Dims);
    maybeSetShape("features_left_08", featureLeft08Dims);
    maybeSetShape("features_left_16", featureLeft16Dims);
    maybeSetShape("features_left_32", featureLeft32Dims);
    maybeSetShape("features_right_04", featureRight04Dims);
    maybeSetShape("stem_2x", stem2xDims);

    auto gwcVolumeDims = m_post_engine->getTensorShape("gwc_volume");
    if (gwcVolumeDims.nbDims != 5) {
        throw std::runtime_error("Expected 5D tensor for gwc_volume, got " + DimsToString(gwcVolumeDims));
    }
    nvinfer1::Dims resolvedGwcVolumeDims = gwcVolumeDims;
    resolvedGwcVolumeDims.d[0] = resolvedGwcVolumeDims.d[0] > 0 ? resolvedGwcVolumeDims.d[0] : 1;
    resolvedGwcVolumeDims.d[1] = resolvedGwcVolumeDims.d[1] > 0 ? resolvedGwcVolumeDims.d[1] : 8;
    resolvedGwcVolumeDims.d[2] = resolvedGwcVolumeDims.d[2] > 0 ? resolvedGwcVolumeDims.d[2] : 48;
    resolvedGwcVolumeDims.d[3] = resolvedGwcVolumeDims.d[3] > 0 ? resolvedGwcVolumeDims.d[3] : m_featureH4;
    resolvedGwcVolumeDims.d[4] = resolvedGwcVolumeDims.d[4] > 0 ? resolvedGwcVolumeDims.d[4] : m_featureW4;
    ValidateStaticOrResolvedShape("gwc_volume", gwcVolumeDims, resolvedGwcVolumeDims);
    if (!maybeSetShape("gwc_volume", resolvedGwcVolumeDims)) {
        throw std::runtime_error("Post engine does not expose required tensor 'gwc_volume'.");
    }

    m_gwcGroups = resolvedGwcVolumeDims.d[1];
    m_maxDispQuarter = resolvedGwcVolumeDims.d[2];
    m_gwcVolumeSize = NumElements(resolvedGwcVolumeDims);

    const auto dispDims = m_post_context->getTensorShape("disp");
    if (HasDynamicDim(dispDims)) {
        throw std::runtime_error("Output tensor disp still has unresolved dimensions: " + DimsToString(dispDims));
    }

    RCLCPP_INFO(
        rclcpp::get_logger("FastFoundationStereoEstimator"),
        "Resolved feature/post shapes: left04=%s left08=%s left16=%s left32=%s right04=%s stem_2x=%s gwc=%s disp=%s",
        DimsToString(featureLeft04Dims).c_str(),
        DimsToString(featureLeft08Dims).c_str(),
        DimsToString(featureLeft16Dims).c_str(),
        DimsToString(featureLeft32Dims).c_str(),
        DimsToString(featureRight04Dims).c_str(),
        DimsToString(stem2xDims).c_str(),
        DimsToString(resolvedGwcVolumeDims).c_str(),
        DimsToString(dispDims).c_str());

}

void FastFoundationStereoEstimator::allocateBuffers() {
    const size_t inputBytes = static_cast<size_t>(3) * m_inputH * m_inputW * sizeof(float);
    m_dLeft  = std::make_unique<DeviceBuffer>(inputBytes);
    m_dRight = std::make_unique<DeviceBuffer>(inputBytes);

    const auto dispDims = m_post_context->getTensorShape("disp");
    const size_t dispElements = NumElements(dispDims);
    m_dDisp  = std::make_unique<DeviceBuffer>(dispElements * sizeof(float));

    m_dRawRGBLeft = std::make_unique<DeviceBuffer>(static_cast<size_t>(m_inputH) * m_inputW * 3 * sizeof(uint8_t));
    m_dRawRGBRight = std::make_unique<DeviceBuffer>(static_cast<size_t>(m_inputH) * m_inputW * 3 * sizeof(uint8_t));

    CHECK_CUDA(cudaMemset(m_dDisp->ptr, 0, m_dDisp->size));

    m_dFeaturesLeft04  = std::make_unique<DeviceBuffer>(m_featuresLeft04Size * sizeof(float));
    m_dFeaturesLeft08  = std::make_unique<DeviceBuffer>(m_featuresLeft08Size * sizeof(float));
    m_dFeaturesLeft16  = std::make_unique<DeviceBuffer>(m_featuresLeft16Size * sizeof(float));
    m_dFeaturesLeft32  = std::make_unique<DeviceBuffer>(m_featuresLeft32Size * sizeof(float));
    m_dFeaturesRight04 = std::make_unique<DeviceBuffer>(m_featuresRight04Size * sizeof(float));
    m_dStem2x          = std::make_unique<DeviceBuffer>(m_stem2xSize * sizeof(float));
    m_dGwcVolume       = std::make_unique<DeviceBuffer>(m_gwcVolumeSize * sizeof(half));

    m_dFeaturesLeft04Half  = std::make_unique<DeviceBuffer>(m_featuresLeft04Size * sizeof(half));
    m_dFeaturesRight04Half = std::make_unique<DeviceBuffer>(m_featuresRight04Size * sizeof(half));

    checkTrtCall(m_feature_context->setTensorAddress("left", m_dLeft->ptr), "setTensorAddress(feature:left)");
    checkTrtCall(m_feature_context->setTensorAddress("right", m_dRight->ptr), "setTensorAddress(feature:right)");
    checkTrtCall(m_feature_context->setTensorAddress("features_left_04", m_dFeaturesLeft04->ptr), "setTensorAddress(feature:features_left_04)");
    checkTrtCall(m_feature_context->setTensorAddress("features_left_08", m_dFeaturesLeft08->ptr), "setTensorAddress(feature:features_left_08)");
    checkTrtCall(m_feature_context->setTensorAddress("features_left_16", m_dFeaturesLeft16->ptr), "setTensorAddress(feature:features_left_16)");
    checkTrtCall(m_feature_context->setTensorAddress("features_left_32", m_dFeaturesLeft32->ptr), "setTensorAddress(feature:features_left_32)");
    checkTrtCall(m_feature_context->setTensorAddress("features_right_04", m_dFeaturesRight04->ptr), "setTensorAddress(feature:features_right_04)");
    checkTrtCall(m_feature_context->setTensorAddress("stem_2x", m_dStem2x->ptr), "setTensorAddress(feature:stem_2x)");

    checkTrtCall(m_post_context->setTensorAddress("features_left_04", m_dFeaturesLeft04->ptr), "setTensorAddress(post:features_left_04)");
    checkTrtCall(m_post_context->setTensorAddress("features_left_08", m_dFeaturesLeft08->ptr), "setTensorAddress(post:features_left_08)");
    if (HasIOTensor(m_post_engine, "features_left_16")) {
        checkTrtCall(m_post_context->setTensorAddress("features_left_16", m_dFeaturesLeft16->ptr), "setTensorAddress(post:features_left_16)");
    }
    checkTrtCall(m_post_context->setTensorAddress("features_left_32", m_dFeaturesLeft32->ptr), "setTensorAddress(post:features_left_32)");
    checkTrtCall(m_post_context->setTensorAddress("features_right_04", m_dFeaturesRight04->ptr), "setTensorAddress(post:features_right_04)");
    checkTrtCall(m_post_context->setTensorAddress("stem_2x", m_dStem2x->ptr), "setTensorAddress(post:stem_2x)");
    checkTrtCall(m_post_context->setTensorAddress("gwc_volume", m_dGwcVolume->ptr), "setTensorAddress(post:gwc_volume)");
    checkTrtCall(m_post_context->setTensorAddress("disp", m_dDisp->ptr), "setTensorAddress(post:disp)");
}

/*
void FastFoundationStereoEstimator::preprocessImage(const cv::Mat& src, std::vector<float>& dstBuffer) {
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(src, bgr_channels); 

    int area = m_inputH * m_inputW;
    cv::Mat r_plane(m_inputH, m_inputW, CV_32FC1, dstBuffer.data());             
    cv::Mat g_plane(m_inputH, m_inputW, CV_32FC1, dstBuffer.data() + area);        
    cv::Mat b_plane(m_inputH, m_inputW, CV_32FC1, dstBuffer.data() + area * 2);    

    bgr_channels[2].convertTo(r_plane, CV_32FC1, 1.0f, 0.0f);
    bgr_channels[1].convertTo(g_plane, CV_32FC1, 1.0f, 0.0f);
    bgr_channels[0].convertTo(b_plane, CV_32FC1, 1.0f, 0.0f);
}
*/

void FastFoundationStereoEstimator::printIOTensors() {
    auto printEngineTensors = [](nvinfer1::ICudaEngine* engine, const char* name) {
        std::cout << "\n=== " << name << " Engine ===" << std::endl;
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* tensorName = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(tensorName);
            auto dims = engine->getTensorShape(tensorName);
            
            std::cout << "Tensor: " << tensorName 
                      << " (I/O: " << (mode == nvinfer1::TensorIOMode::kINPUT ? "IN" : "OUT") << ")"
                      << " dims: ";
            for (int j = 0; j < dims.nbDims; ++j) std::cout << dims.d[j] << " ";
            std::cout << std::endl;
        }
    };
    printEngineTensors(m_feature_engine, "Feature");
    printEngineTensors(m_post_engine, "Post");
}

bool FastFoundationStereoEstimator::inference(const cv::Mat& leftRaw, const cv::Mat& rightRaw, cv::Mat& outputDisp) {    
    if (leftRaw.rows != m_inputH || leftRaw.cols != m_inputW) {
        std::cerr << "Check left_image shape failed! shape: " << leftRaw.rows << "x" << leftRaw.cols << std::endl;
        return false;
    }

    if (rightRaw.rows != m_inputH || rightRaw.cols != m_inputW) {
        std::cerr << "Check right_image shape failed! shape: " << rightRaw.rows << "x" << rightRaw.cols << std::endl;
        return false;
    }

    cv::Mat leftImg = m_padder->pad(leftRaw);
    cv::Mat rightImg = m_padder->pad(rightRaw);

    bool ret = infer(leftImg, rightImg, outputDisp);

    outputDisp = m_padder->unpad(outputDisp);

    return ret;
}

bool FastFoundationStereoEstimator::infer(const cv::Mat& leftImg, const cv::Mat& rightImg, cv::Mat& outputDisp) {
    if (leftImg.empty() || rightImg.empty()) {
        std::cerr << "Input images are empty!" << std::endl;
        return false;
    }

    if (leftImg.rows != m_inputH || leftImg.cols != m_inputW) {
        std::cerr << "Input size mismatch! Expected " << m_inputW << "x" << m_inputH << std::endl;
        return false;
    }

    // 1. Preprocess (CPU -> Host Buffer)
    CHECK_CUDA(cudaMemcpyAsync(m_dRawRGBLeft->ptr, leftImg.data, m_inputH * m_inputW * 3, cudaMemcpyHostToDevice, m_stream));
    CHECK_CUDA(cudaMemcpyAsync(m_dRawRGBRight->ptr, rightImg.data, m_inputH * m_inputW * 3, cudaMemcpyHostToDevice, m_stream));

    // 2. Run CUDA Preprocessing Kernels (Converts BGR HWC -> RGB CHW Float)
    LaunchPreprocessKernel(m_dRawRGBLeft->as<uint8_t>(), m_dLeft->as<float>(), 
                           m_inputW, m_inputH, m_stream);
    LaunchPreprocessKernel(m_dRawRGBRight->as<uint8_t>(), m_dRight->as<float>(), 
                           m_inputW, m_inputH, m_stream);

    // 3. Feature Extraction
    if (!m_feature_context->enqueueV3(m_stream)) {
        std::cerr << "TensorRT execution failed!" << std::endl;
        return false;
    }

    // 4. Build GWC Volume via CUDA
    convertFloatToHalf(
        m_dFeaturesLeft04->as<float>(),
        m_dFeaturesLeft04Half->as<half>(),
        m_featuresLeft04Size,
        m_stream);
    convertFloatToHalf(
        m_dFeaturesRight04->as<float>(),
        m_dFeaturesRight04Half->as<half>(),
        m_featuresRight04Size,
        m_stream);

    int B = 1;
    int C = m_featureLeft04Channels;
    int H = m_featureH4;
    int W = m_featureW4;

    const char* gwc_normalize_env = std::getenv("FFS_GWC_NORMALIZE");
    const char* gwc_direction_env = std::getenv("FFS_GWC_DIRECTION");
    const bool gwc_normalize =
        gwc_normalize_env != nullptr && std::strcmp(gwc_normalize_env, "1") == 0;
    const bool reverse_gwc_shift =
        gwc_direction_env != nullptr && std::strcmp(gwc_direction_env, "right") == 0;

    LaunchGwcVolumeKernel(
        m_dFeaturesLeft04Half->as<half>(), 
        m_dFeaturesRight04Half->as<half>(), 
        m_dGwcVolume->as<half>(), 
        B, C, H, W, m_maxDispQuarter, m_gwcGroups, gwc_normalize, reverse_gwc_shift, m_stream
    );

    // 5. Post Processing Inference
    if (!m_post_context->enqueueV3(m_stream)) {
        std::cerr << "TensorRT execution failed!" << std::endl;
        return false;
    }

    // 6. Copy results to CPU
    CHECK_CUDA(cudaMemcpyAsync(m_hDisp.data(), m_dDisp->ptr, m_dDisp->size, cudaMemcpyDeviceToHost, m_stream));
    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    
    cv::Mat(m_inputH, m_inputW, CV_32FC1, m_hDisp.data()).copyTo(outputDisp);

    return true;
}
