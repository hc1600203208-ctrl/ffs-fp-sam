#include "grounded_sam/trt_logger.hpp"

#include <iostream>

namespace grounded_sam {

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TensorRT] " << msg << '\n';
  }
}

TrtLogger& trtLogger() {
  static TrtLogger logger;
  return logger;
}

}  // namespace grounded_sam

