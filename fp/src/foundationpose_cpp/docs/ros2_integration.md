# FoundationPose ROS 2 Integration

## Overview

This repository now provides a ROS 2 node named `foundationpose_tracker_node` that:

- subscribes to synchronized RGB, depth, and mask images for the first-frame `Register`,
- switches to synchronized RGB/depth `Track` on subsequent frames,
- publishes the object 6DoF pose as `geometry_msgs/msg/PoseStamped`.

It also provides a ROS 2 node named `rgbd_sequence_publisher_node` that replays a local RGB/depth
image sequence as live topics when you do not have a rosbag yet.

For pure offline processing, the repository also provides `foundationpose_offline_rgbd_node`.
This node reads a local RGB-D sequence directly from disk, runs first-frame `Register` followed
by per-frame `Track`, and writes pose CSV results plus visualization images/video.

The published pose is the object pose in the camera frame, expressed in the original mesh
coordinate frame. This matches the pose matrices saved by the official Python FoundationPose
demo under `debug/ob_in_cam`.

For visualization parity with the official Python demo, place `to_origin.txt` and `extents.txt`
next to the mesh file. These files should be exported from `trimesh.bounds.oriented_bounds(mesh)`;
the helper script `tools/export_trimesh_oriented_bounds.py` writes them using the same convention.
When the files are absent, the C++ loader falls back to its internal PCA oriented box.

## Build

```bash
cd /path/to/ros2_ws
colcon build --packages-select foundationpose_cpp --cmake-args -DENABLE_TENSORRT=ON
```

If you still want the standalone demo tests, add:

```bash
colcon build --packages-select foundationpose_cpp  --cmake-args -DENABLE_TENSORRT=ON -DBUILD_SIMPLE_TESTS=ON
cd /home/hc/weizi/fp/build/foundationpose_cpp/bin
./simple_tests --gtest_filter=foundationpose_test.test
```

## Run

Edit the example parameter file:

`config/foundationpose_ros2_example.yaml`

Then launch the node:

```bash
source install/setup.bash
ros2 run foundationpose_cpp foundationpose_tracker_node \
  --ros-args --params-file /home/hc/weizi/fp/src/foundationpose_cpp/config/foundationpose_ros2_example.yaml
```

To publish the local sample RGBD sequence at 10 FPS:

```bash
source install/setup.bash
ros2 run foundationpose_cpp rgbd_sequence_publisher_node \
  --ros-args --params-file /home/hc/weizi/fp/src/foundationpose_cpp/config/rgbd_sequence_publisher_example.yaml
```

For a complete live loop, make sure another ROS 2 node also publishes the first-frame mask on
`mask_topic`. Start the RGB-D publisher first, then the mask publisher, and finally run
`foundationpose_tracker_node` in a second terminal.

To process the sample `mustard0` RGB-D sequence directly from disk:

```bash
source install/setup.bash
ros2 run foundationpose_cpp foundationpose_offline_rgbd_node \
  --ros-args --params-file /home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_offline_rgbd_example.yaml
```

By default the publisher uses `reliable` QoS and also publishes `CameraInfo`, which makes it
friendlier to RViz2 `Image` and `Camera` displays. If you need sensor-style QoS for another
consumer, set `use_sensor_data_qos: true`.

## Main Parameters

- `refiner_engine_path`: TensorRT engine for the refiner model.
- `scorer_engine_path`: TensorRT engine for the scorer model.
- `mesh_path`: local mesh path used for pose estimation.
- `rgb_topic`: RGB image topic.
- `depth_topic`: depth image topic.
- `mask_topic`: mask image topic used only for first-frame registration.
- `pose_topic`: output `PoseStamped` topic.
- `fx`, `fy`, `cx`, `cy`: camera intrinsics.
- `depth_scale`: scale applied to integer depth images, usually `0.001` for millimeters to meters.
- `register_refine_iterations`: refine iterations used during initialization.
- `track_refine_iterations`: refine iterations used during tracking.
- `sequence_dir`: dataset directory containing `rgb/` and `depth/` folders for replay.
- `fps`: replay rate for `rgbd_sequence_publisher_node`.
- `camera_info_path`: optional `cam_K.txt` path used to publish `CameraInfo`.
- `publish_camera_info`: whether `rgbd_sequence_publisher_node` also publishes `CameraInfo`.
- `use_sensor_data_qos`: switch replay publishing from RViz-friendly `reliable` to `SensorDataQoS`.
- `sequence_dir`, `rgb_subdir`, `depth_subdir`, `masks_subdir`: local offline dataset layout.
- `initial_frame_id`, `initial_mask_path`: registration seed frame and its mask for offline processing.
- `output_pose_path`, `visualization_output_dir`, `video_output_path`: offline result export paths.
