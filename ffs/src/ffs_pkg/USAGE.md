# Usage

## Environment
Requires ROS2 Humble. The node subscribes to stereo image topics, runs inference, and publishes depth + point cloud.

## Weights

Download from [Google Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link) and place under `weights/`.

| Checkpoint | PyTorch (ms) | Python-TRT (ms) | Notes |
|-----------|-------------|---------|-------|
| `23-36-37` | 49.4 | 23.4 | Best accuracy |
| `20-26-39` | 43.6 | 19.4 | Balanced |
| `20-30-48` | 38.4 | 16.6 | Fastest |

> Profiled on RTX 3090 Ti, 480×640, `valid_iters=8`.


| Checkpoint | FPS (Hz) | CPP-TRT (ms) | Notes |
|-----------|-------------|---------|-------|
| `23-36-37` | 6.3 | 156.4 | Best accuracy |
| `20-26-39` | 9.1 | 110.3 | Balanced |
| `20-30-48` | 10.2 | 97.6 | Fastest |

> Profiled on Jetson Orin AGX 64G, 480×640, `valid_iters=4`.

If you export a different engine resolution, make the runtime config match it exactly.
For example, if your TensorRT `feature_runner` and `post_runner` were exported at `448x640`,
set `model_input_height=448` and feed `448x640` images into the node/test binary.



## TensorRT (Tested on 10.3.0/10.7.0)

```bash
# Export ONNX
python scripts/make_onnx.py \
    --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
    --save_path output/23-36-37/ --height 480 --width 640

# Build engine (via trtexec)
/usr/src/tensorrt/bin/trtexec --onnx=/home/hc/model/20-30-48/feature_runner.onnx --saveEngine=/home/hc/model/20-30-48/feature_runner_fp16_5060.engine --fp16  --useCudaGraph
/usr/src/tensorrt/bin/trtexec --onnx=/home/hc/model/20-30-48/post_runner.onnx --saveEngine=/home/hc/model/20-30-48/post_runner_fp16_5060.engine --fp16  --useCudaGraph

```

## Demo

```bash
# Inference Once
./install/fast_foundation_stereo/lib/fast_foundation_stereo/fast_foundation_stereo_test

# ROS Node/Launch
ros2 launch fast_foundation_stereo dnn_stereo_depth.launch.py
ros2 bag play /home/hc/weizi/ffs/assets/ess_rosbag -l

# Or let the launch file start rosbag after a short delay
ros2 launch fast_foundation_stereo dnn_stereo_depth.launch.py \
  play_rosbag:=true \
  rosbag_path:=/home/hc/weizi/ffs/assets/ess_rosbag \
  rosbag_start_delay:=3.0

# Save each aligned depth frame and the corresponding left RGB image
ros2 launch fast_foundation_stereo dnn_stereo_depth.launch.py \
  publish_filtered_depth:=true \
  save_results:=true \
  save_output_dir:=/home/hc/weizi/output/dnn_capture \
  save_depth_scale:=1000.0

```
ros2 launch fast_foundation_stereo offline_stereo_depth.launch.py

The offline launch also writes matching LAF inputs under `dataset_root`:
`laf_rgb/<frame>.png` (rectified model-size left image),
`disparity/<frame>.png` (16-bit disparity in pixels times 256), and
`topk_cost/<frame>.npy` (`float32`, shape `[7, model_height, model_width]`).
Top-K cost is the seven smallest negative mean group-wise correlations from
the FFS quarter-resolution volume, resized to model resolution. Channels are
ordered best to seventh best. These are raw FFS costs, not the MC-CNN costs
used to train the released LAF weights; use the seven-channel cost feature
branch directly and retrain or calibrate LAF for FFS data.
In `models.py`, replace the two cost preprocessing lines (`L2normalize`/`softmax`
and `torch.topk`) with `x = cost` when loading this seven-channel NPY.
Read disparity as `png.astype(np.float32) / 256.0`; LAF additionally divides
pixel disparity by 228 before its disparity branch.

```bash
python3 -c 'import cv2, numpy as np, sys; p=sys.argv[1]; d=cv2.imread(p+"/disparity/0001.png",-1); r=cv2.imread(p+"/laf_rgb/0001.png"); c=np.load(p+"/topk_cost/0001.npy"); assert d.dtype==np.uint16 and c.dtype==np.float32 and c.shape==(7,*d.shape) and r.shape[:2]==d.shape and np.isfinite(c).all() and (c[:-1]<=c[1:]+1e-6).all(); print("disparity px", d.max()/256, "cost", c.shape)' /home/hc/dataset/myobject/blue_fast/laf_export
```

## FFS internal matching confidence

The original post-runner engine exposes only `disp`. The offline launch uses
`post_runner_with_prob_fp16_5060.engine`, whose second output is the model's
softmax probability volume (`prob`, `[1,48,112,160]` at 448x640 input).
`ffs_confidence/<frame>.png` is the maximum probability across disparity bins,
upsampled and aligned with the saved depth (`uint16 / 65535` gives `[0,1]`).
`depth_ffs_filter/<frame>.png` retains raw depth where this score is at least
`model_conf_threshold` (default `0.3`). This score describes the initial
matching distribution before FFS's iterative disparity refinement; it is not
a calibrated probability that the final depth is correct.
The external-filter result in `depth_filter` is independent of this output.

```bash
source /home/hc/weizi/ffs+fp+sam/ffs/install/setup.bash
ros2 launch fast_foundation_stereo offline_stereo_depth.launch.py \
  dataset_root:=/home/hc/dataset/myobject/blue_fast \
  caminfo_path:=/home/hc/weizi/ffs+fp+sam/ffs/jr714.txt \
  model_conf_threshold:=0.3
```

To rebuild the prob-enabled engine from an existing post-runner ONNX:

```bash
/home/hc/anaconda3/envs/ffs/bin/python /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/expose_ffs_prob.py \
  /home/hc/model/ffs/23-36-37/post_runner.onnx \
  /home/hc/model/ffs/23-36-37/post_runner_with_prob.onnx
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/hc/model/ffs/23-36-37/post_runner_with_prob.onnx \
  --saveEngine=/home/hc/model/ffs/23-36-37/post_runner_with_prob_fp16_5060.engine \
  --fp16 --useCudaGraph --skipInference
```

## Depth Inspector

Use the standalone depth inspection tool to visually compare RGB and saved depth results frame by frame.
The viewer shows the RGB image and a colorized depth map side by side, and updates the hovered pixel's
RGB value and depth value in real time.

```bash
python3 src/ffs_pkg/scripts/depth_inspector.py \
  --rgb-dir /home/hc/weizi/dataset/jrnew-blue/rgb \
  --depth-dir /home/hc/weizi/dataset/jrnew-blue/depth \
  --depth-scale 1000.0
```

After installation you can also run:

```bash
./install/fast_foundation_stereo/lib/fast_foundation_stereo/depth_inspector.py \
  --rgb-dir /home/hc/weizi/dataset/jrnew-blue/rgb \
  --depth-dir /home/hc/weizi/dataset/jrnew-blue/depth_raw
```

Viewer controls:
- `A` or left arrow: previous frame
- `D` or right arrow: next frame
- `S`: save the current viewer screenshot next to the depth image
- `Q` or `Esc`: quit


The stereo node publishes:
- `/disparity/image_raw` as `32FC1`
- `/disparity/image_vis` as `bgr8` for direct viewing in RViz2 or `rqt_image_view`

The rectified depth node publishes:
- `/depth_image` as the selected primary depth topic (`32FC1`), controlled by `publish_filtered_depth`
- `/depth_image_raw` as the raw metric depth converted from disparity (`32FC1`)
- `/depth_image_filtered` as the confidence-filtered depth (`32FC1`)
- `/confidence_map` as the per-pixel confidence map in `[0, 1]` (`32FC1`)
- `/weight_map` as the confidence-derived weight map (`32FC1`)

Confidence defaults exposed in `disparity2pc_node`:
- `patch_radius=2`
- `sigma_photo=0.12`
- `sigma_e=0.15`
- `sigma_t=0.02`
- `alpha_edge=0.8`
- `w_photo=0.45`
- `w_tex=0.20`
- `w_grad=0.20`
- `w_tmp=0.15`
- `conf_threshold=0.35`
- `weight_gamma=1.5`

This pipeline computes depth in meters when `baseline` is configured in meters, so the default temporal scale is `sigma_t=0.02` meters. If a downstream integration converts depth to millimeters, use `sigma_t=20.0` instead.
