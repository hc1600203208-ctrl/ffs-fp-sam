cmake_minimum_required(VERSION 3.18)
project(grounded_sam_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_GROUNDED_SAM_DEMO "Build the Grounded DINO + SAM demo" ON)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT root directory")

if(NOT TENSORRT_ROOT AND EXISTS "/usr/src/tensorrt")
  set(TENSORRT_ROOT "/usr/src/tensorrt")
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(ament_cmake QUIET)

if(ament_cmake_FOUND)
  set(DEFAULT_BUILD_ROS2_NODE ON)
else()
  set(DEFAULT_BUILD_ROS2_NODE OFF)
endif()

option(BUILD_ROS2_NODE "Build the ROS 2 first-mask node" ${DEFAULT_BUILD_ROS2_NODE})

if(NOT TENSORRT_ROOT)
  message(FATAL_ERROR "Set TENSORRT_ROOT or install TensorRT under /usr/src/tensorrt")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS
    ${TENSORRT_ROOT}/include
    ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/include
  NO_DEFAULT_PATH
)

find_library(TENSORRT_NVINFER nvinfer
  PATHS
    ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
  NO_DEFAULT_PATH
)

find_library(TENSORRT_NVPARSER nvparsers
  PATHS
    ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
  NO_DEFAULT_PATH
)

find_library(TENSORRT_NVONNXPARSER nvonnxparser
  PATHS
    ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
  NO_DEFAULT_PATH
)

find_library(TENSORRT_NVINFER_PLUGIN nvinfer_plugin
  PATHS
    ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
  NO_DEFAULT_PATH
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER OR NOT TENSORRT_NVONNXPARSER OR NOT TENSORRT_NVINFER_PLUGIN)
  message(FATAL_ERROR "Failed to find TensorRT headers/libs under ${TENSORRT_ROOT}")
endif()

add_library(grounded_sam_core
  src/trt_logger.cpp
  src/trt_utils.cpp
  src/image_utils.cpp
  src/grounding_dino_runner.cpp
  src/sam_runner.cpp
  src/pipeline.cpp
)

target_include_directories(grounded_sam_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(grounded_sam_core
  PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVONNXPARSER}
    ${TENSORRT_NVINFER_PLUGIN}
    pthread
)

target_compile_definitions(grounded_sam_core PRIVATE
  TRT_ENABLE_FALLBACK_WARNING=1
)

if(BUILD_GROUNDED_SAM_DEMO)
  add_executable(grounded_sam_demo src/main.cpp)
  target_link_libraries(grounded_sam_demo PRIVATE grounded_sam_core)
  install(TARGETS grounded_sam_demo RUNTIME DESTINATION bin)
endif()

if(BUILD_ROS2_NODE)
  if(NOT ament_cmake_FOUND)
    find_package(ament_cmake REQUIRED)
  endif()

  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)

  add_executable(grounded_sam_first_mask_node
    src/grounded_sam_first_mask_node.cpp
  )
  add_executable(grounded_sam_multi_first_mask_node
    src/grounded_sam_multi_first_mask_node.cpp
  )
  target_link_libraries(grounded_sam_first_mask_node grounded_sam_core)
  target_link_libraries(grounded_sam_multi_first_mask_node grounded_sam_core)
  ament_target_dependencies(grounded_sam_first_mask_node
    rclcpp
    sensor_msgs
    cv_bridge
  )
  ament_target_dependencies(grounded_sam_multi_first_mask_node
    rclcpp
    sensor_msgs
    cv_bridge
  )

  install(TARGETS grounded_sam_first_mask_node grounded_sam_multi_first_mask_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

if(ament_cmake_FOUND)
  ament_package()
endif()
