cmake_minimum_required(VERSION 3.16)
project(satellite_udp_standalone LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

add_executable(satellite_renderer_udp src/satellite_renderer_udp.cpp)
target_include_directories(satellite_renderer_udp PRIVATE include)
target_link_libraries(satellite_renderer_udp PRIVATE Eigen3::Eigen PNG::PNG ZLIB::ZLIB Threads::Threads)
target_compile_options(satellite_renderer_udp PRIVATE -O2 -Wall -Wextra)
install(TARGETS satellite_renderer_udp RUNTIME DESTINATION bin)

option(BUILD_GROUND_TRACKER "Build non-ROS CUDA/TensorRT ground tracker" OFF)
if(BUILD_GROUND_TRACKER)
  set(BUILD_ROS2_NODE OFF CACHE BOOL "" FORCE)
  set(BUILD_SIMPLE_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_GROUNDED_SAM_DEMO OFF CACHE BOOL "" FORCE)
  set(ENABLE_TENSORRT ON CACHE BOOL "" FORCE)
  find_package(CUDA REQUIRED)
  find_package(OpenCV REQUIRED)
  find_package(yaml-cpp REQUIRED)
  find_library(NVINFER_LIB nvinfer HINTS /usr/src/tensorrt/lib /usr/src/tensorrt/targets/x86_64-linux-gnu/lib REQUIRED)
  find_library(NVONNXPARSER_LIB nvonnxparser HINTS /usr/src/tensorrt/lib /usr/src/tensorrt/targets/x86_64-linux-gnu/lib REQUIRED)
  add_subdirectory(../fp/src/foundationpose_cpp ${CMAKE_BINARY_DIR}/fp_core)
  add_subdirectory(../sam ${CMAKE_BINARY_DIR}/sam_core)
  set(FFS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ffs/src/ffs_pkg/src)
  cuda_add_library(fast_foundation_stereo_udp SHARED
    src/fast_foundation_stereo_estimator_udp.cpp
    ${FFS_DIR}/estimator/gwc_volume_kernel.cu)
  target_include_directories(fast_foundation_stereo_udp PUBLIC
    ${FFS_DIR} ${FFS_DIR}/estimator ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS} /usr/src/tensorrt/include)
  target_link_libraries(fast_foundation_stereo_udp
    ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${NVINFER_LIB} ${NVONNXPARSER_LIB})
  add_executable(ground_tracker_udp src/ground_tracker_udp.cpp ${FFS_DIR}/depth_confidence.cpp)
  target_include_directories(ground_tracker_udp PRIVATE
    include
    ${FFS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../fp/src/foundationpose_cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../fp/src/foundationpose_cpp/detection_6d_foundationpose/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../fp/src/foundationpose_cpp/easy_deploy_tool/inference_core/trt_core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../sam/include)
  target_link_libraries(ground_tracker_udp PRIVATE
    detection_6d_foundationpose trt_core grounded_sam_core fast_foundation_stereo_udp
    yaml-cpp ${OpenCV_LIBS} Threads::Threads ZLIB::ZLIB)
  install(TARGETS ground_tracker_udp RUNTIME DESTINATION bin)
endif()
