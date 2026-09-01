# C++ stereo renderer

This package renders a textured mesh into a synthetic stereo pair and publishes the images through ROS 2.

It reuses:

- `fp/install/foundationpose_cpp/lib/libdetection_6d_foundationpose.so`
- `fp/install/foundationpose_cpp/include/detection_6d_foundationpose/mesh_loader.hpp`
- the CUDA rasterization helpers shipped with the FoundationPose renderer
- `ffs/src/ffs_pkg/src/stereo_calibration_utils.hpp`

Default mesh:

- `/home/hc/weizi/dataset/jrnew-blue/mesh1/textured_mesh.obj`

Default calibration:

- `/home/hc/weizi/ffs+fp+sam/ffs/640.txt`

The node prompts for:

- initial object pose relative to the left camera
- translation amplitudes in meters
- translation frequencies in Hz
- rotation amplitudes in degrees
- rotation frequencies in Hz

## Build

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
colcon build --packages-select sim_stereo_cpp
```

The workspace uses `sim/colcon_defaults.yaml`, so `build/`, `install/`, and `log/` stay inside `sim/`.

## Run

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
ros2 launch sim_stereo_cpp stereo_render_cpp.launch.py
```

The node keeps the interactive prompts even under `ros2 launch` by reading from `/dev/tty` when available.
