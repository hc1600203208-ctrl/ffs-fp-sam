cmake_minimum_required(VERSION 3.8)
project(foundationpose_cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(FOUNDATIONPOSE_INSTALL_RPATH
    "$ORIGIN"
    "$ORIGIN/.."
    "/usr/local/lib")
set(CMAKE_INSTALL_RPATH "${FOUNDATIONPOSE_INSTALL_RPATH}")
list(APPEND CMAKE_BUILD_RPATH "/usr/local/lib")

if(DEFINED ENV{AMENT_PREFIX_PATH} AND EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
      "Python3 interpreter used by ROS 2 ament tooling" FORCE)
  set(PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
      "Python interpreter used by legacy ROS 2 ament helpers" FORCE)
  find_package(Python3 QUIET COMPONENTS Interpreter)
endif()

find_package(ament_cmake QUIET)

if(ament_cmake_FOUND)
  set(DEFAULT_BUILD_ROS2_NODE ON)
  set(DEFAULT_BUILD_SIMPLE_TESTS OFF)
else()
  set(DEFAULT_BUILD_ROS2_NODE OFF)
  set(DEFAULT_BUILD_SIMPLE_TESTS ON)
endif()

option(BUILD_ROS2_NODE "Build the ROS 2 tracking node" ${DEFAULT_BUILD_ROS2_NODE})
option(BUILD_SIMPLE_TESTS "Build the standalone demo tests" ${DEFAULT_BUILD_SIMPLE_TESTS})

if(BUILD_ROS2_NODE AND EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
      "Python3 interpreter used by ROS 2 ament tooling" FORCE)
  set(PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
      "Python interpreter used by legacy ROS 2 ament helpers" FORCE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

set(TENSORRT_INCLUDE_DIR /usr/src/tensorrt/include)
set(TENSORRT_LIBRARY_DIR /usr/src/tensorrt/targets/x86_64-linux-gnu/lib)
find_package(CUDA REQUIRED)
find_library(TENSORRT_NVINFER_LIB nvinfer HINTS ${TENSORRT_LIBRARY_DIR} REQUIRED)
find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_LIBRARY_DIR} REQUIRED)
find_library(TENSORRT_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_LIBRARY_DIR} REQUIRED)
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)
add_subdirectory(easy_deploy_tool)
add_subdirectory(detection_6d_foundationpose)

if(BUILD_SIMPLE_TESTS)
  add_subdirectory(simple_tests)
endif()

if(BUILD_ROS2_NODE)
  if(NOT ament_cmake_FOUND)
    find_package(ament_cmake REQUIRED)
  endif()

  if(NOT TARGET trt_core)
    message(FATAL_ERROR "BUILD_ROS2_NODE requires TensorRT support. Reconfigure with -DENABLE_TENSORRT=ON.")
  endif()

  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(message_filters REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(OpenCV REQUIRED)

  set(FAST_FOUNDATION_STEREO_SOURCE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../ffs/src/ffs_pkg"
      CACHE PATH "Path to the fast_foundation_stereo source package")

  if(NOT EXISTS "${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src/estimator/fast_foundation_stereo_estimator.cpp")
    message(FATAL_ERROR
      "FAST_FOUNDATION_STEREO_SOURCE_DIR does not point to ffs_pkg: "
      "${FAST_FOUNDATION_STEREO_SOURCE_DIR}")
  endif()

  cuda_add_library(fast_foundation_stereo_embedded SHARED
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src/estimator/fast_foundation_stereo_estimator.cpp
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src/estimator/gwc_volume_kernel.cu
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src/depth_confidence.cpp
  )

  target_include_directories(fast_foundation_stereo_embedded PUBLIC
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src/estimator
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
  )

  target_link_directories(fast_foundation_stereo_embedded PRIVATE ${TENSORRT_LIBRARY_DIR})
  target_link_libraries(fast_foundation_stereo_embedded
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
  )

  ament_target_dependencies(fast_foundation_stereo_embedded
    rclcpp
  )

  add_executable(foundationpose_tracker_node
    src/foundationpose_ros2_node.cpp
  )

  add_executable(foundationpose_file_mask_tracker_node
    src/foundationpose_file_mask_ros2_node.cpp
  )

  add_executable(foundationpose_stereo_tracker_node
    src/foundationpose_stereo_tracker_node.cpp
    src/adaptive_se3_pose_filter.cpp
    src/se3_utils.cpp
  )

  add_executable(foundationpose_stereo_tracker_fast_node
    src/foundationpose_stereo_tracker_fast_node.cpp
  )

  add_executable(rgbd_sequence_publisher_node
    src/rgbd_sequence_publisher_node.cpp
  )

  add_executable(foundationpose_offline_rgbd_node
    src/foundationpose_offline_rgbd_node.cpp
    src/adaptive_se3_pose_filter.cpp
    src/se3_utils.cpp
  )

  add_executable(foundationpose_offline_rgbd_no_filter_node
    src/foundationpose_offline_rgbd_no_filter_node.cpp
  )

  target_link_libraries(foundationpose_tracker_node
    detection_6d_foundationpose
    trt_core
    ${OpenCV_LIBS}
  )

  target_link_libraries(foundationpose_file_mask_tracker_node
    detection_6d_foundationpose
    trt_core
    ${OpenCV_LIBS}
  )

  target_link_libraries(foundationpose_stereo_tracker_node
    detection_6d_foundationpose
    trt_core
    fast_foundation_stereo_embedded
    ${OpenCV_LIBS}
  )

  target_link_libraries(foundationpose_stereo_tracker_fast_node
    detection_6d_foundationpose
    trt_core
    fast_foundation_stereo_embedded
    ${OpenCV_LIBS}
  )

  target_link_libraries(rgbd_sequence_publisher_node
    ${OpenCV_LIBS}
  )

  target_link_libraries(foundationpose_offline_rgbd_node
    detection_6d_foundationpose
    trt_core
    ${OpenCV_LIBS}
  )

  target_link_libraries(foundationpose_offline_rgbd_no_filter_node
    detection_6d_foundationpose
    trt_core
    ${OpenCV_LIBS}
  )

  target_include_directories(foundationpose_tracker_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_6d_foundationpose/include
    ${CMAKE_CURRENT_SOURCE_DIR}/easy_deploy_tool/inference_core/trt_core/include
    ${OpenCV_INCLUDE_DIRS}
  )

  target_include_directories(foundationpose_file_mask_tracker_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_6d_foundationpose/include
    ${CMAKE_CURRENT_SOURCE_DIR}/easy_deploy_tool/inference_core/trt_core/include
    ${OpenCV_INCLUDE_DIRS}
  )

  target_include_directories(foundationpose_stereo_tracker_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_6d_foundationpose/include
    ${CMAKE_CURRENT_SOURCE_DIR}/easy_deploy_tool/inference_core/trt_core/include
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src/estimator
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
  )

  target_include_directories(foundationpose_stereo_tracker_fast_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_6d_foundationpose/include
    ${CMAKE_CURRENT_SOURCE_DIR}/easy_deploy_tool/inference_core/trt_core/include
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src
    ${FAST_FOUNDATION_STEREO_SOURCE_DIR}/src/estimator
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
  )

  target_include_directories(rgbd_sequence_publisher_node PRIVATE
    ${OpenCV_INCLUDE_DIRS}
  )

  target_include_directories(foundationpose_offline_rgbd_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_6d_foundationpose/include
    ${CMAKE_CURRENT_SOURCE_DIR}/easy_deploy_tool/inference_core/trt_core/include
    ${OpenCV_INCLUDE_DIRS}
  )

  target_include_directories(foundationpose_offline_rgbd_no_filter_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_6d_foundationpose/include
    ${CMAKE_CURRENT_SOURCE_DIR}/easy_deploy_tool/inference_core/trt_core/include
    ${OpenCV_INCLUDE_DIRS}
  )

  ament_target_dependencies(foundationpose_tracker_node
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
    message_filters
    tf2
  )

  ament_target_dependencies(foundationpose_file_mask_tracker_node
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
    message_filters
    tf2
  )

  ament_target_dependencies(foundationpose_stereo_tracker_node
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
    message_filters
    tf2
  )

  ament_target_dependencies(foundationpose_stereo_tracker_fast_node
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
    message_filters
    tf2
  )

  ament_target_dependencies(rgbd_sequence_publisher_node
    rclcpp
    sensor_msgs
    std_msgs
    cv_bridge
  )

  ament_target_dependencies(foundationpose_offline_rgbd_node
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
  )

  ament_target_dependencies(foundationpose_offline_rgbd_no_filter_node
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
  )

  install(TARGETS foundationpose_tracker_node
                  foundationpose_file_mask_tracker_node
                  foundationpose_stereo_tracker_node
                  foundationpose_stereo_tracker_fast_node
                  fast_foundation_stereo_embedded
                  rgbd_sequence_publisher_node
                  foundationpose_offline_rgbd_node
                  foundationpose_offline_rgbd_no_filter_node
    DESTINATION lib/${PROJECT_NAME}
  )

  install(DIRECTORY config launch
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
  )

  ament_package()
endif()
