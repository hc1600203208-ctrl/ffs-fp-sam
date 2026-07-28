cmake_minimum_required(VERSION 3.8)
project(fast_foundation_stereo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -fopenmp")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
set(dependencies
    ament_cmake
    Eigen3
    rclcpp
    rclcpp_components
    rclcpp_lifecycle
    message_filters
    sensor_msgs
    std_msgs
    std_srvs
    tf2
    tf2_msgs
    tf2_ros
    tf2_sensor_msgs
    tf2_geometry_msgs
    nav_msgs
    visualization_msgs
    cv_bridge
    pcl_conversions
    depth_image_proc
)

foreach(dep IN LISTS dependencies)
    find_package(${dep} REQUIRED)
endforeach()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)

set(TENSORRT_INCLUDE_DIR /usr/src/tensorrt/include)
set(TENSORRT_LIBRARY_DIR /usr/src/tensorrt/targets/x86_64-linux-gnu/lib)
find_library(TENSORRT_NVINFER_LIB nvinfer HINTS ${TENSORRT_LIBRARY_DIR} REQUIRED)
find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_LIBRARY_DIR} REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)



# library
cuda_add_library(stereo_estimator SHARED
  src/estimator/fast_foundation_stereo_estimator.cpp
  src/estimator/gwc_volume_kernel.cu
)
ament_target_dependencies(stereo_estimator 
  rclcpp 
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
)

# Add include directories
target_include_directories(stereo_estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE ${TENSORRT_INCLUDE_DIR})

# Add library directories
target_link_directories(stereo_estimator PRIVATE ${TENSORRT_LIBRARY_DIR})
target_link_libraries(stereo_estimator
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_NVINFER_LIB}
  ${TENSORRT_NVONNXPARSER_LIB})



cuda_add_executable(dnn_stereo_depth_node src/dnn_stereo_depth_node.cpp)
ament_target_dependencies(dnn_stereo_depth_node 
  rclcpp 
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
  std_srvs
)
target_link_libraries(dnn_stereo_depth_node stereo_estimator)
set_target_properties(dnn_stereo_depth_node PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN;${TENSORRT_LIBRARY_DIR}")

cuda_add_executable(dnn_stereo_rectified_depth_node
  src/dnn_stereo_rectified_depth_node.cpp
  src/depth_confidence.cpp
)
ament_target_dependencies(dnn_stereo_rectified_depth_node
  rclcpp
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
  std_srvs
)
target_link_libraries(dnn_stereo_rectified_depth_node stereo_estimator)
set_target_properties(dnn_stereo_rectified_depth_node PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN;${TENSORRT_LIBRARY_DIR}")

cuda_add_executable(dnn_stereo_fast_depth_node
  src/dnn_stereo_fast_depth_node.cpp
)
ament_target_dependencies(dnn_stereo_fast_depth_node
  rclcpp
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
)
target_link_libraries(dnn_stereo_fast_depth_node stereo_estimator)
set_target_properties(dnn_stereo_fast_depth_node PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN;${TENSORRT_LIBRARY_DIR}")

cuda_add_executable(offline_stereo_depth_node
  src/offline_stereo_depth_node.cpp
  src/depth_confidence.cpp
)
ament_target_dependencies(offline_stereo_depth_node
  rclcpp
  cv_bridge
  CUDA OpenCV
  sensor_msgs
  std_msgs
)
target_link_libraries(offline_stereo_depth_node stereo_estimator)
set_target_properties(offline_stereo_depth_node PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN;${TENSORRT_LIBRARY_DIR}")

cuda_add_executable(offline_depth_benchmark_node
  src/benchmark/offline_depth_benchmark_node.cpp
  src/depth_confidence.cpp
)
ament_target_dependencies(offline_depth_benchmark_node
  rclcpp
  CUDA OpenCV
  sensor_msgs
  std_msgs
)
target_link_libraries(offline_depth_benchmark_node stereo_estimator)
set_target_properties(offline_depth_benchmark_node PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN;${TENSORRT_LIBRARY_DIR}")

add_executable(disparity2depth_node
  src/disparity_to_pc_node.cpp
  src/depth_confidence.cpp
)
ament_target_dependencies(disparity2depth_node
  rclcpp 
  cv_bridge
  CUDA OpenCV
  message_filters
  sensor_msgs
  std_msgs
)
set_target_properties(disparity2depth_node PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")



# executable
cuda_add_executable(fast_foundation_stereo_test src/test/fast_foundation_stereo_test.cpp)
ament_target_dependencies(fast_foundation_stereo_test 
rclcpp 
cv_bridge
CUDA OpenCV
)
target_link_libraries(fast_foundation_stereo_test stereo_estimator)
set_target_properties(fast_foundation_stereo_test PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")



install(TARGETS
  fast_foundation_stereo_test
  stereo_estimator
  dnn_stereo_depth_node
  dnn_stereo_rectified_depth_node
  dnn_stereo_fast_depth_node
  offline_stereo_depth_node
  offline_depth_benchmark_node
  disparity2depth_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch rviz
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/depth_inspector.py
  scripts/batch_stereo_rectification_vis.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
