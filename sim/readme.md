cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
colcon build --packages-select sim_stereo_cpp

source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
ros2 launch sim_stereo_cpp stereo_render_cpp.launch.py