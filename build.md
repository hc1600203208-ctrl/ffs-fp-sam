xhost +
 
rm -rf build install log
cd /workspace/code/ffs+fp+sam/ffs
colcon build

cd /workspace/code/ffs+fp+sam/fp
colcon build --packages-select foundationpose_cpp  --cmake-args -DENABLE_TENSORRT=ON -DBUILD_SIMPLE_TESTS=ON

cd /workspace/code/ffs+fp+sam/sim
colcon build --packages-select sim_stereo_cpp
source install/setup.bash
ros2 launch sim_stereo_cpp stereo_render_cpp.launch.py

cd /workspace/code/ffs+fp+sam/sam
./scripts/build_cpp.sh
./scripts/build_ros.sh
./scripts/export_dino.sh

cd /workspace/code/ffs+fp+sam/fp
source install/setup.bash
source /workspace/code/ffs+fp+sam/sam/ros2_install/setup.bash
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_fast.launch.py

cd /workspace/data
ros2 bag play bluegold1 -l


