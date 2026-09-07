cmake_minimum_required(VERSION 3.18)
project(grounded_sam_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_GROUNDED_SAM_DEMO "Build the Grounded DINO + SAM demo" ON)
option(BUILD_DINO_ONNXRUNTIME "Build the ONNX Runtime backend for GroundingDINO" OFF)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT root directory")
set(ONNXRUNTIME_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../fp/src/foundationpose_cpp/easy_deploy_tool/inference_core/ort_core/onnxruntime-pkg-aarch64"
  CACHE PATH "ONNX Runtime root directory"
)

if(NOT TENSORRT_ROOT)
  foreach(_tensorrt_root
      /usr/src/tensorrt
      /usr/local/tensorrt
      /usr/local/TensorRT
  )
    if(EXISTS "${_tensorrt_root}/include/NvInfer.h")
      set(TENSORRT_ROOT "${_tensorrt_root}" CACHE PATH "TensorRT root directory" FORCE)
      break()
    endif()
  endforeach()
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

set(TENSORRT_SEARCH_PATHS)
if(TENSORRT_ROOT)
  list(APPEND TENSORRT_SEARCH_PATHS
    ${TENSORRT_ROOT}/include
    ${TENSORRT_ROOT}/targets/aarch64-linux-gnu/include
    ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/include
  )
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS
    ${TENSORRT_SEARCH_PATHS}
    /usr/src/tensorrt/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
)

set(TENSORRT_LIBRARY_PATHS)
if(TENSORRT_ROOT)
  list(APPEND TENSORRT_LIBRARY_PATHS
    ${TENSORRT_ROOT}/targets/aarch64-linux-gnu/lib
    ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
  )
endif()

find_library(TENSORRT_NVINFER nvinfer
  PATHS
    ${TENSORRT_LIBRARY_PATHS}
    /usr/src/tensorrt/targets/x86_64-linux-gnu/lib
    /usr/src/tensorrt/targets/aarch64-linux-gnu/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)

find_library(TENSORRT_NVONNXPARSER nvonnxparser
  PATHS
    ${TENSORRT_LIBRARY_PATHS}
    /usr/src/tensorrt/targets/x86_64-linux-gnu/lib
    /usr/src/tensorrt/targets/aarch64-linux-gnu/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER_PLUGIN nvinfer_plugin
  PATHS
    ${TENSORRT_LIBRARY_PATHS}
    /usr/src/tensorrt/targets/x86_64-linux-gnu/lib
    /usr/src/tensorrt/targets/aarch64-linux-gnu/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER OR NOT TENSORRT_NVONNXPARSER OR NOT TENSORRT_NVINFER_PLUGIN)
  message(FATAL_ERROR "Failed to find TensorRT headers/libs. Set TENSORRT_ROOT for a custom TensorRT install.")
endif()

if(BUILD_DINO_ONNXRUNTIME)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS
      ${ONNXRUNTIME_ROOT}/include
      /usr/include
      /usr/local/include
  )
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    PATHS
      ${ONNXRUNTIME_ROOT}/lib
      /usr/lib/aarch64-linux-gnu
      /usr/local/lib
  )
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "Failed to find ONNX Runtime. Set ONNXRUNTIME_ROOT or configure with -DBUILD_DINO_ONNXRUNTIME=OFF.")
  endif()
  get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  get_filename_component(ONNXRUNTIME_LIBRARY_REALPATH "${ONNXRUNTIME_LIBRARY}" REALPATH)
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

if(BUILD_DINO_ONNXRUNTIME)
  target_include_directories(grounded_sam_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_compile_definitions(grounded_sam_core PUBLIC GROUNDED_SAM_WITH_ONNXRUNTIME=1)
endif()

target_link_libraries(grounded_sam_core
  PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVONNXPARSER}
    ${TENSORRT_NVINFER_PLUGIN}
    pthread
)

if(BUILD_DINO_ONNXRUNTIME)
  target_link_libraries(grounded_sam_core PUBLIC ${ONNXRUNTIME_LIBRARY})
endif()

target_compile_definitions(grounded_sam_core PRIVATE
  TRT_ENABLE_FALLBACK_WARNING=1
)

function(configure_grounded_sam_runtime target_name)
  if(BUILD_DINO_ONNXRUNTIME)
    set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
    set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH "$ORIGIN/..")
    set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
  endif()
endfunction()

if(BUILD_GROUNDED_SAM_DEMO)
  add_executable(grounded_sam_demo src/main.cpp)
  target_link_libraries(grounded_sam_demo PRIVATE grounded_sam_core)
  configure_grounded_sam_runtime(grounded_sam_demo)
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
  configure_grounded_sam_runtime(grounded_sam_first_mask_node)
  configure_grounded_sam_runtime(grounded_sam_multi_first_mask_node)
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

  if(BUILD_DINO_ONNXRUNTIME)
    install(FILES "${ONNXRUNTIME_LIBRARY_REALPATH}"
      DESTINATION lib
    )
  endif()
endif()

if(ament_cmake_FOUND)
  ament_package()
endif()
