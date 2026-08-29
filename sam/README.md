# Grounded DINO + SAM TensorRT Deployment

This directory contains a standalone C++/TensorRT deployment for the
`Grounded-Segment-Anything` pipeline, without modifying the original Python
project.

The deployed pipeline matches the practical workflow used in the Python demo:

1. Load an RGB image.
2. Run GroundingDINO with a fixed text prompt to produce candidate boxes.
3. Select boxes above confidence thresholds.
4. Run SAM box-prompt segmentation for the selected boxes.
5. Merge masks into a single binary mask and optionally save it as PNG.

The current implementation is intentionally prompt-specialized for deployment
stability and ease of TensorRT export. If you want a different prompt, rerun
the export steps with a new prompt.

GroundingDINO uses a fixed `800x1066` inference canvas for the `640x480`
camera input. This is the exact single-image shape produced by the Python
`RandomResize([800], max_size=1333)` transform. The export bakes a full-valid
positional encoding into the graph; this avoids the dynamic padding-mask path
that can produce incorrect deformable attention coordinates in TensorRT on
this platform.

## Layout

- `scripts/` - helper scripts to export ONNX models and build TensorRT engines
- `tools/` - optional Python utilities used during export
- `src/` and `include/` - C++ runtime and TensorRT wrappers
- `models/` - place checkpoints and generated ONNX files here
- `engines/` - generated TensorRT engine files
- `package.xml` and `src/grounded_sam_first_mask_node.cpp` - ROS 2 first-mask package

## Recommended workflow

1. Put the original checkpoints into `models/` or point the export scripts to
   the checkpoint paths in `../Grounded-Segment-Anything/`.
2. Export GroundingDINO and SAM ONNX models.
3. Build TensorRT engines from the exported ONNX files.
4. Build the C++ demo.
5. Run the binary on an RGB image folder or a single image.

For one `640x480` RGB image, run:

```bash
GS_DINO_DEBUG=1 ./build/grounded_sam_demo \
  --input /path/to/image.png \
  --output_dir /tmp/grounded_sam_output \
  --dino_engine ./engines/grounding_dino_fixed_prompt.engine \
  --sam_encoder_engine ./engines/sam_image_encoder.engine \
  --sam_decoder_engine ./engines/sam_mask_decoder.engine \
  --box_threshold 0.3 \
  --max_detections 16
```

The output mask is saved as `<input_stem>_mask.png`. The demo accepts PNG,
JPG, JPEG, and BMP files and converts OpenCV's BGR input to RGB before
inference.

## Current status

- C++ runtime builds successfully with TensorRT 10 and OpenCV 4.
- `sam_image_encoder.onnx` and `sam_mask_decoder.onnx` export successfully.
- `sam_image_encoder.engine` and `sam_mask_decoder.engine` build successfully.
- GroundingDINO exports with fixed prompt and fixed positional encoding without
  NaNs in the deformable transformer path.
- `grounding_dino_fixed_prompt.engine` builds with strict FP32 TensorRT
  execution and produces detections from the C++ runner.
- The C++ demo runs the complete DINO plus SAM pipeline and writes a binary
  mask PNG.
- The ROS 2 node subscribes to one first RGB frame, publishes `/fisrt_mask`
  when enabled, and writes the same `first_mask.png` file that the
  FoundationPose tracker polls. Its default input size is `640x480`, matching
  the stereo camera and FoundationPose calibration.

The fixed positional encoding is deliberate: do not build this DINO engine
with a dynamic `masks` input unless the TensorRT mask/position path has been
validated on the target GPU. The export defaults can be changed with
`DINO_HEIGHT` and `DINO_WIDTH`, but they must match the C++ runner and engine
profile.

The helper scripts are written to be explicit about paths and to fail loudly
when a dependency is missing.

## ROS 2 first-mask node

Build the package after sourcing the ROS 2 environment:

```bash
./scripts/build_ros2.sh
source ../fp/install/setup.bash
source ros2_install/setup.bash
```

The integrated FoundationPose launch file starts this node directly and no
longer starts the Python `ros2_first_mask_node.py` process:

```bash
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_fast.launch.py
```

The default TensorRT engine uses the fixed export prompt `blue carton`. A
different prompt requires re-exporting GroundingDINO and rebuilding its
engine. Useful launch overrides include:

```bash
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_fast.launch.py \
  dino_engine:=/path/to/grounding_dino_fixed_prompt.engine \
  first_mask_output_dir:=/path/to/first_mask \
  input_width:=640 input_height:=480 \
  box_threshold:=0.3
```

The FoundationPose tracker still reads
`first_mask_output_dir/first_mask.png`, while `/fisrt_mask` is available for
ROS 2 consumers when `publish_mask_topic` is true.

cd /home/hc/weizi/ffs+fp+sam/sam

/home/hc/anaconda3/envs/sam/bin/python scripts/export_onnx.py grounding_dino \
  --device cpu \
  --text_prompt "red cup" \
  --dino_height 800 \
  --dino_width 1066 \
  --fixed_mask \
  --output models/grounding_dino_red_cup.onnx

  /usr/src/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
  --onnx=/home/hc/weizi/ffs+fp+sam/sam/models/grounding_dino_red_cup.onnx \
  --saveEngine=/home/hc/weizi/ffs+fp+sam/sam/engines/grounding_dino_red_cup.engine \
  --minShapes=images:1x3x800x1066 \
  --optShapes=images:1x3x800x1066 \
  --maxShapes=images:1x3x800x1066 \
  --noTF32 \
  --stronglyTyped

  source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash

ros2 launch foundationpose_cpp foundationpose_stereo_tracker_fast.launch.py \
  dino_engine:=/home/hc/weizi/ffs+fp+sam/sam/engines/grounding_dino_red_cup.engine \
  box_threshold:=0.3

  cd ~/weizi/ffs+fp+sam/sam
./scripts/export_dino.sh