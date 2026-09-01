#include "grounded_sam/trt_utils.hpp"

#include "grounded_sam/trt_logger.hpp"

#include <cuda_fp16.h>
#include <fstream>
#include <numeric>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace grounded_sam {

namespace {

std::vector<char> readBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open engine file: " + path);
  }
  file.seekg(0, std::ios::end);
  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<std::size_t>(size));
  file.read(data.data(), size);
  return data;
}

}  // namespace

std::size_t volume(const nvinfer1::Dims& dims) {
  std::size_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("Cannot compute volume for dynamic dims: " + dimsToString(dims));
    }
    result *= static_cast<std::size_t>(dims.d[i]);
  }
  return result;
}

std::size_t dataTypeSize(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT64:
      return 8;
    default:
      throw std::runtime_error("Unsupported TensorRT data type");
  }
}

std::string dimsToString(const nvinfer1::Dims& dims) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i) oss << "x";
    oss << dims.d[i];
  }
  oss << "]";
  return oss.str();
}

void checkCuda(cudaError_t code, const char* expr, const char* file, int line) {
  if (code != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
        << cudaGetErrorString(code);
    throw std::runtime_error(oss.str());
  }
}

void copyFloatToTensor(TensorBuffer& buffer, const float* src, std::size_t count) {
  const std::size_t expected = volume(buffer.dims);
  if (count != expected) {
    throw std::runtime_error("Float tensor element count mismatch for " + buffer.name);
  }
  if (buffer.dtype == nvinfer1::DataType::kFLOAT) {
    std::memcpy(buffer.host.data(), src, expected * sizeof(float));
    return;
  }
  if (buffer.dtype == nvinfer1::DataType::kHALF) {
    auto* dst = reinterpret_cast<__half*>(buffer.host.data());
    for (std::size_t i = 0; i < expected; ++i) {
      dst[i] = __float2half(src[i]);
    }
    return;
  }
  if (buffer.dtype == nvinfer1::DataType::kINT32) {
    auto* dst = reinterpret_cast<std::int32_t*>(buffer.host.data());
    for (std::size_t i = 0; i < expected; ++i) {
      dst[i] = static_cast<std::int32_t>(src[i]);
    }
    return;
  }
  throw std::runtime_error("Unsupported input tensor dtype for " + buffer.name);
}

void copyBoolToTensor(TensorBuffer& buffer, const std::uint8_t* src, std::size_t count) {
  const std::size_t expected = volume(buffer.dims);
  if (count != expected) {
    throw std::runtime_error("Bool tensor element count mismatch for " + buffer.name);
  }
  if (buffer.dtype == nvinfer1::DataType::kBOOL || buffer.dtype == nvinfer1::DataType::kUINT8) {
    std::memcpy(buffer.host.data(), src, expected);
    return;
  }
  if (buffer.dtype == nvinfer1::DataType::kFLOAT) {
    auto* dst = reinterpret_cast<float*>(buffer.host.data());
    for (std::size_t i = 0; i < expected; ++i) {
      dst[i] = src[i] ? 1.0f : 0.0f;
    }
    return;
  }
  if (buffer.dtype == nvinfer1::DataType::kHALF) {
    auto* dst = reinterpret_cast<__half*>(buffer.host.data());
    for (std::size_t i = 0; i < expected; ++i) {
      dst[i] = __float2half(src[i] ? 1.0f : 0.0f);
    }
    return;
  }
  throw std::runtime_error("Unsupported bool tensor dtype for " + buffer.name);
}

std::vector<float> tensorToFloatVector(const TensorBuffer& buffer) {
  const std::size_t expected = volume(buffer.dims);
  std::vector<float> out(expected);
  if (buffer.dtype == nvinfer1::DataType::kFLOAT) {
    std::memcpy(out.data(), buffer.host.data(), expected * sizeof(float));
    return out;
  }
  if (buffer.dtype == nvinfer1::DataType::kHALF) {
    const auto* src = reinterpret_cast<const __half*>(buffer.host.data());
    for (std::size_t i = 0; i < expected; ++i) {
      out[i] = __half2float(src[i]);
    }
    return out;
  }
  if (buffer.dtype == nvinfer1::DataType::kBOOL || buffer.dtype == nvinfer1::DataType::kUINT8) {
    const auto* src = reinterpret_cast<const std::uint8_t*>(buffer.host.data());
    for (std::size_t i = 0; i < expected; ++i) {
      out[i] = src[i] ? 1.0f : 0.0f;
    }
    return out;
  }
  if (buffer.dtype == nvinfer1::DataType::kINT32) {
    const auto* src = reinterpret_cast<const std::int32_t*>(buffer.host.data());
    for (std::size_t i = 0; i < expected; ++i) {
      out[i] = static_cast<float>(src[i]);
    }
    return out;
  }
  throw std::runtime_error("Unsupported output tensor dtype for " + buffer.name);
}

TrtEngine::TrtEngine(const std::string& engine_path) {
  const auto data = readBinaryFile(engine_path);
  runtime_.reset(nvinfer1::createInferRuntime(trtLogger()));
  if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT engine: " + engine_path);

  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("Failed to create TensorRT execution context");

  GS_CHECK_CUDA(cudaStreamCreate(&stream_));

  const int nb_tensors = engine_->getNbIOTensors();
  for (int i = 0; i < nb_tensors; ++i) {
    const char* name = engine_->getIOTensorName(i);
    TensorBuffer buffer;
    buffer.name = name;
    buffer.mode = engine_->getTensorIOMode(name);
    buffer.dtype = engine_->getTensorDataType(name);
    buffer.dims = engine_->getTensorShape(name);
    tensors_.emplace(buffer.name, std::move(buffer));
  }
  refreshBuffers();
}

TrtEngine::~TrtEngine() {
  for (auto& item : tensors_) {
    if (item.second.device) {
      cudaFree(item.second.device);
      item.second.device = nullptr;
    }
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void TrtEngine::setInputShape(const std::string& name, const nvinfer1::Dims& dims) {
  if (!context_->setInputShape(name.c_str(), dims)) {
    throw std::runtime_error("Failed to set input shape for " + name + " to " + dimsToString(dims));
  }
  refreshBuffers();
}

void TrtEngine::refreshBuffers() {
  for (auto& item : tensors_) {
    auto& buffer = item.second;
    buffer.dims = context_->getTensorShape(buffer.name.c_str());
    if (buffer.dims.nbDims <= 0) {
      buffer.dims = engine_->getTensorShape(buffer.name.c_str());
    }
    bool has_dynamic = false;
    for (int i = 0; i < buffer.dims.nbDims; ++i) {
      has_dynamic = has_dynamic || buffer.dims.d[i] < 0;
    }
    if (!has_dynamic) {
      allocateTensor(buffer);
      if (!context_->setTensorAddress(buffer.name.c_str(), buffer.device)) {
        throw std::runtime_error("Failed to bind TensorRT tensor: " + buffer.name);
      }
    }
  }
}

void TrtEngine::allocateTensor(TensorBuffer& buffer) {
  const std::size_t bytes = volume(buffer.dims) * dataTypeSize(buffer.dtype);
  if (bytes == buffer.bytes && buffer.device) return;
  if (buffer.device) {
    GS_CHECK_CUDA(cudaFree(buffer.device));
    buffer.device = nullptr;
  }
  buffer.bytes = bytes;
  buffer.host.resize(bytes);
  GS_CHECK_CUDA(cudaMalloc(&buffer.device, bytes));
}

void TrtEngine::infer() {
  for (auto& item : tensors_) {
    auto& buffer = item.second;
    if (buffer.mode == nvinfer1::TensorIOMode::kINPUT) {
      GS_CHECK_CUDA(cudaMemcpyAsync(buffer.device, buffer.host.data(), buffer.bytes,
                                    cudaMemcpyHostToDevice, stream_));
    }
  }

  if (!context_->enqueueV3(stream_)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  for (auto& item : tensors_) {
    auto& buffer = item.second;
    if (buffer.mode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (buffer.host.empty()) continue;
      GS_CHECK_CUDA(cudaMemcpyAsync(buffer.host.data(), buffer.device, buffer.bytes,
                                    cudaMemcpyDeviceToHost, stream_));
    }
  }
  GS_CHECK_CUDA(cudaStreamSynchronize(stream_));
}

TensorBuffer& TrtEngine::tensor(const std::string& name) {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("Unknown tensor: " + name);
  return it->second;
}

const TensorBuffer& TrtEngine::tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("Unknown tensor: " + name);
  return it->second;
}

bool TrtEngine::hasTensor(const std::string& name) const {
  return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> TrtEngine::inputNames() const {
  std::vector<std::string> names;
  for (const auto& item : tensors_) {
    if (item.second.mode == nvinfer1::TensorIOMode::kINPUT) names.push_back(item.first);
  }
  return names;
}

std::vector<std::string> TrtEngine::outputNames() const {
  std::vector<std::string> names;
  for (const auto& item : tensors_) {
    if (item.second.mode == nvinfer1::TensorIOMode::kOUTPUT) names.push_back(item.first);
  }
  return names;
}

}  // namespace grounded_sam
