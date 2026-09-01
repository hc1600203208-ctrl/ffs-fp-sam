#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace grounded_sam {

struct TensorBuffer {
  std::string name;
  nvinfer1::Dims dims{};
  nvinfer1::DataType dtype{};
  nvinfer1::TensorIOMode mode{};
  std::vector<std::uint8_t> host;
  void* device{nullptr};
  std::size_t bytes{0};
};

class TrtEngine {
 public:
  explicit TrtEngine(const std::string& engine_path);
  ~TrtEngine();

  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  void setInputShape(const std::string& name, const nvinfer1::Dims& dims);
  void infer();

  TensorBuffer& tensor(const std::string& name);
  const TensorBuffer& tensor(const std::string& name) const;
  bool hasTensor(const std::string& name) const;
  std::vector<std::string> inputNames() const;
  std::vector<std::string> outputNames() const;

  template <typename T>
  T* hostData(const std::string& name) {
    return reinterpret_cast<T*>(tensor(name).host.data());
  }

  template <typename T>
  const T* hostData(const std::string& name) const {
    return reinterpret_cast<const T*>(tensor(name).host.data());
  }

 private:
  void refreshBuffers();
  void allocateTensor(TensorBuffer& buffer);

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{nullptr};
  std::map<std::string, TensorBuffer> tensors_;
};

std::size_t volume(const nvinfer1::Dims& dims);
std::size_t dataTypeSize(nvinfer1::DataType type);
std::string dimsToString(const nvinfer1::Dims& dims);
void checkCuda(cudaError_t code, const char* expr, const char* file, int line);
void copyFloatToTensor(TensorBuffer& buffer, const float* src, std::size_t count);
void copyBoolToTensor(TensorBuffer& buffer, const std::uint8_t* src, std::size_t count);
std::vector<float> tensorToFloatVector(const TensorBuffer& buffer);

}  // namespace grounded_sam

#define GS_CHECK_CUDA(expr) ::grounded_sam::checkCuda((expr), #expr, __FILE__, __LINE__)
