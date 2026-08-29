#pragma once

#include <NvInfer.h>

namespace grounded_sam {

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

TrtLogger& trtLogger();

}  // namespace grounded_sam

