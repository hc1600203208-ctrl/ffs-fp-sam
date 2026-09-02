xhost +
 
cd /workspace/code/ffs+fp+sam/ffs
colcon build

cd /workspace/code/ffs+fp+sam/fp
colcon build --packages-select foundationpose_cpp  --cmake-args -DENABLE_TENSORRT=ON -DBUILD_SIMPLE_TESTS=ON

cd /workspace/code/ffs+fp+sam/sim
colcon build --packages-select sim_stereo_cpp

cd /workspace/code/ffs+fp+sam/sam
./scripts/build_cpp.sh
./scripts/build_ros.sh
