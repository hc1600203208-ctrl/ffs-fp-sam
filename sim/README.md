# Stereo mesh renderer

This package renders a textured mesh as a synthetic stereo pair and publishes the images through ROS2.

Default assets:

- Mesh: `/home/hc/weizi/dataset/jrnew-blue/mesh1/textured_mesh.obj`
- Calibration: `/home/hc/weizi/ffs+fp+sam/ffs/640.txt`

## Run

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
~/anaconda3/bin/conda run -n sam python -m sim_stereo.node
```

The node starts in interactive mode by default and asks for:

- initial object pose relative to the left camera
- translation oscillation amplitude and frequency
- rotation oscillation amplitude and frequency

Rotation inputs are in degrees. Translation inputs are in meters. Frequencies are in Hz.

You can also pass `--ros-args -p interactive:=false` and rely on the YAML defaults or launch file.

## C++ version

A ROS 2 C++ implementation lives in `sim/src/sim_stereo_cpp` and reuses the mesh loader and rendering helpers from `fp`.

Build:

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
colcon build --packages-select sim_stereo_cpp
```

The workspace uses `sim/colcon_defaults.yaml`, so `build/`, `install/`, and `log/` stay inside `sim/`.

Run:

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
ros2 launch sim_stereo_cpp stereo_render_cpp.launch.py
```
