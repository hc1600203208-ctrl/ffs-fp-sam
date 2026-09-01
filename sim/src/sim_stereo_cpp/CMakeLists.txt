cmake_minimum_required(VERSION 3.16)
project(sim_stereo_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(FP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../fp")
set(FFS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../ffs")

set(FP_INSTALL_LIB "${FP_ROOT}/install/foundationpose_cpp/lib")
set(FP_INSTALL_INCLUDE "${FP_ROOT}/install/foundationpose_cpp/include")
set(FP_SOURCE_INCLUDE "${FP_ROOT}/src/foundationpose_cpp/detection_6d_foundationpose/include")
set(FP_SOURCE_SRC "${FP_ROOT}/src/foundationpose_cpp/detection_6d_foundationpose/src")
set(FP_DEPLOY_INCLUDE "${FP_ROOT}/src/foundationpose_cpp/easy_deploy_tool/deploy_core/include")
set(FFS_STEREO_INCLUDE "${FFS_ROOT}/src/ffs_pkg/src")

find_library(FOUNDATIONPOSE_DETECTION_LIB
  NAMES detection_6d_foundationpose
  PATHS "${FP_INSTALL_LIB}"
  REQUIRED
)

find_library(FOUNDATIONPOSE_DEPLOY_LIB
  NAMES deploy_core
  PATHS "${FP_INSTALL_LIB}"
  REQUIRED
)

add_executable(stereo_render_cpp_node
  src/motion_profile.cpp
  src/stereo_mesh_renderer.cpp
  src/stereo_render_node.cpp
)

target_include_directories(stereo_render_cpp_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${FP_INSTALL_INCLUDE}
  ${FP_SOURCE_INCLUDE}
  ${FP_SOURCE_SRC}
  ${FP_DEPLOY_INCLUDE}
  ${FFS_STEREO_INCLUDE}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(stereo_render_cpp_node PRIVATE
  ${FP_INSTALL_LIB}
)

target_link_libraries(stereo_render_cpp_node
  ${FOUNDATIONPOSE_DETECTION_LIB}
  ${FOUNDATIONPOSE_DEPLOY_LIB}
  CUDA::cudart
  ${OpenCV_LIBS}
)

ament_target_dependencies(stereo_render_cpp_node
  rclcpp
  sensor_msgs
  geometry_msgs
)

target_compile_options(stereo_render_cpp_node PRIVATE -O3)

set(FP_RUNTIME_LIB_DIR "${FP_INSTALL_LIB}")
set(FP_RUNTIME_CUDA_DIR "/usr/local/cuda/lib64")
set(FP_RUNTIME_CVCUDA_DIR "/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu")
set_target_properties(stereo_render_cpp_node PROPERTIES
  BUILD_RPATH "${FP_RUNTIME_LIB_DIR};${FP_RUNTIME_CUDA_DIR};${FP_RUNTIME_CVCUDA_DIR}"
  INSTALL_RPATH "${FP_RUNTIME_LIB_DIR};${FP_RUNTIME_CUDA_DIR};${FP_RUNTIME_CVCUDA_DIR}"
)

install(TARGETS stereo_render_cpp_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
