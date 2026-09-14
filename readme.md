# 整体启动脚本
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_fast.launch.py

# ffs使用说明

ros2 launch fast_foundation_stereo offline_stereo_depth.launch.py \
dataset_root:=/home/hc/dataset/myobject/oxi_fast_all \
caminfo_path:=/home/hc/weizi/ffs+fp+sam/ffs/jr714.txt

ros2 launch fast_foundation_stereo offline_stereo_depth.launch.py \
dataset_root:=/home/hc/dataset/myobject/weixing1 \
caminfo_path:=/home/hc/weizi/ffs+fp+sam/ffs/128091.txt
# 进行深度图和rgb图对应可视化
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/depth_inspector.py --rgb-dir /home/hc/dataset/myobject/weixing1/rgb --depth-dir /home/hc/dataset/myobject/weixing1/depth
# 进行极限校正可视化结果
python /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_stereo_rectification_vis.py --caminfo-path /home/hc/dataset/myobject/fluke/jr714.txt --left-dir /home/hc/dataset/myobject/fluke/rgb --right-dir /home/hc/dataset/myobject/fluke/camera2 --output-dir /home/hc/dataset/myobject/fluke/rect

# 生成extents.txt

python /home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/tools/export_trimesh_oriented_bounds.py /home/hc/dataset/myobject/weixing1/bundlesdf720/textured_mesh.obj

# bundlesdf 简化命令
source ~/anaconda3/bin/activate
conda activate bundlesdf

python /home/hc/weizi/BundleSDF/tools/simplify_mesh_surface.py \
  --input /home/hc/weizi/dataset/jrnew-blue/mesh \
  --output_dir /home/hc/weizi/dataset/jrnew-blue/mesh_simple \
  --method normal \
  --normal_ray_offset_ratio 2.0 \
  --normal_dilate_hops 2 \
  --min_component_faces 0 \
  --min_component_area 0 \
  --overwrite

# 运行单次sam命令
python /home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/sam.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --input_image /home/hc/dataset/blue715/rgb/0001.png \
  --output_dir "/home/hc/dataset/blue715/masks" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "blue carton " \
  --device "cuda"

# sim仿真命令
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
colcon build --packages-select sim_stereo_cpp

source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
ros2 launch sim_stereo_cpp stereo_render_cpp.launch.py



# 运行批量评估脚本命令
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py    /home/hc/dataset/myobject/lgj   --tracking_dataset_root /home/hc/dataset/myobject/lgj --bundle_shorter_side 720

 python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py   /home/hc/dataset/myobject/lgj   --skip_depth_benchmark   --skip_main_reconstruction   --skip_gaijin_reconstruction   --tracking_dataset_root /home/hc/dataset/myobject/lgj_fast2   --foundationpose_depth_source tracking_dataset   --overwrite  --fp_fps 20


python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py   /home/hc/dataset/myobject/patrick   --skip_depth_benchmark   --skip_main_reconstruction   --skip_gaijin_reconstruction   --tracking_dataset_root /home/hc/dataset/myobject/patrick_fastall_stride5   --foundationpose_depth_source tracking_dataset   --overwrite  --fp_fps 12

python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py   /home/hc/dataset/myobject/fluke_2  --skip_depth_benchmark   --skip_main_reconstruction   --skip_gaijin_reconstruction   --tracking_dataset_root /home/hc/dataset/myobject/fluke_fast2   --foundationpose_depth_source tracking_dataset   --overwrite  --fp_fps 20    --fp_render_mode faces

python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py    /home/hc/dataset/myobject/oxi   --tracking_dataset_root /home/hc/dataset/myobject/oxi --bundle_shorter_side 720 --fp_fps 12  --fp_render_mode faces 

python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py    /home/hc/dataset/myobject/oxi  --skip_depth_benchmark   --skip_main_reconstruction   --skip_gaijin_reconstruction  --foundationpose_depth_source tracking_dataset   --overwrite --tracking_dataset_root /home/hc/dataset/myobject/oxi_fast  --fp_fps 20  --fp_render_mode faces  --allow_dirty_bundlesdf

 python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py   /home/hc/dataset/myobject/lgj   --skip_depth_benchmark   --skip_main_reconstruction   --skip_gaijin_reconstruction   --tracking_dataset_root /home/hc/dataset/myobject/lgj_fast --foundationpose_depth_source tracking_dataset   --overwrite  --fp_fps 20  --fp_render_mode faces --fp_q_omega 3e-2 

 cd /home/hc/weizi/ffs+fp+sam/sam
 # 生成dino 和sam engine文件

 ./build_engines.sh
# 生成dino onnx文件
cd ~/weizi/ffs+fp+sam/sam
./scripts/export_dino.sh
#
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_fast.launch.py
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py
# 单次c++部署sam测试
./build/grounded_sam_demo \
  --input /home/hc/weizi/dataset/jrnew-blue/rgb/0000.png \
  --output_dir demo_outputs \
  --dino_engine /home/hc/weizi/ffs+fp+sam/sam/engines/grounding_dino_fixed_prompt.engine \
  --sam_encoder_engine /home/hc/weizi/ffs+fp+sam/sam/engines/sam_image_encoder.engine \
  --sam_decoder_engine /home/hc/weizi/ffs+fp+sam/sam/engines/sam_mask_decoder.engine \
  --box_threshold 0.3 \
  --max_detections 16
# batch_evaluate_stack.py 使用说明

脚本路径：

```bash
/home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py
```

这个脚本用于批量完成 `FFS 双目深度评估 + BundleSDF 重建质量评估 + FoundationPose 跟踪质量评估`。它会先对一个或多个重建数据集生成多组深度图，再分别调用 BundleSDF 的 `main` 分支和改进分支进行重建和评估，自动选出重建质量最好的一组深度图，最后用改进前后的 mesh 去跑 FoundationPose 跟踪评估。

## 一、整体流程

默认执行顺序如下：

1. 运行 Fast Foundation Stereo 离线深度 benchmark。
2. 切换 `/home/hc/weizi/BundleSDF` 到 `main` 分支，对所有深度组进行 BundleSDF 重建和 mesh 评估。
3. 从 `reconstruction_main/summary.csv` 中自动选择重建质量最好的一组深度图。
4. 切换 BundleSDF 到 `gaijin` 分支，只使用第 3 步选出的最佳深度组进行改进版 BundleSDF 重建和评估。
5. 生成改进前后 mesh 的对比表。
6. 使用 `main` 和 `gaijin` 两个 mesh，分别运行 FoundationPose 的 `standard` 和 `adaptive_filter` 两种跟踪器。
7. 汇总每个数据集的状态、最佳深度组、重建指标和跟踪指标。

默认输出目录是每个数据集下面的：

```bash
<dataset_root>/evaluate
```

例如：

```bash
/home/hc/dataset/myobject/fluke_2/evaluate
```

## 二、输入数据集要求

### 1. 重建数据集

每个重建数据集默认需要包含：

```text
dataset_root/
  rgb/          # 左目 RGB 图像，默认参数 --left_subdir rgb
  camera2/      # 右目图像，默认参数 --right_subdir camera2
  masks/        # 目标物体 mask，默认参数 --mask_subdir masks
  cam_K.txt     # 左目相机内参，用于 BundleSDF 和部分评估
  caminfo.txt   # 双目标定文件，用于 FFS 离线深度；也可以叫 cam_info.txt
```

如果标定文件不是 `caminfo.txt` 或 `cam_info.txt`，需要用 `--caminfo_name` 指定，例如：

```bash
--caminfo_name jr714.txt
```

如果左右目或 mask 文件夹名称不同，可以用：

```bash
--left_subdir left_rgb \
--right_subdir right_rgb \
--mask_subdir object_masks
```

脚本会检查：

- `rgb/`、`camera2/`、`masks/` 是否存在。
- `cam_K.txt` 是否存在。
- `caminfo.txt`、`cam_info.txt` 或 `--caminfo_name` 指定的文件是否存在。
- 左右目图像文件名 stem 是否一致，例如 `000001.png` 和 `000001.png`。
- RGB 和 mask 是否至少有相同帧号。

### 2. FoundationPose 跟踪数据集

默认参数 `--foundationpose_depth_source tracking_dataset` 表示 FoundationPose 跟踪评估使用一个单独的 RGBD 跟踪序列，而不是重建阶段的双目数据集。

跟踪数据集默认需要包含：

```text
tracking_dataset_root/
  rgb/          # 跟踪 RGB 图像
  depth/        # 跟踪用深度图
  masks/        # 跟踪 mask
  cam_K.txt     # 跟踪相机内参
```

可以通过下面任意一种方式指定跟踪数据集。

方式 1：直接指定某个跟踪数据集：

```bash
--tracking_dataset_root /home/hc/dataset/myobject/fluke_2_tracking
```

方式 2：如果跟踪数据就在重建数据集下面的某个子目录：

```bash
--tracking_dataset_subdir tracking_rgbd
```

对应结构：

```text
dataset_root/
  tracking_rgbd/
    rgb/
    depth/
    masks/
    cam_K.txt
```

方式 3：批量评估多个物体时，用映射文件指定每个重建数据集对应的跟踪数据集：

```bash
--tracking_dataset_map /home/hc/dataset/myobject/tracking_map.csv
```

映射文件格式可以是 CSV：

```text
dataset,tracking_dataset
fluke_2,/home/hc/dataset/myobject/fluke_2_tracking
box_1,/home/hc/dataset/myobject/box_1_tracking
```

也可以是空格分隔：

```text
fluke_2 /home/hc/dataset/myobject/fluke_2_tracking
box_1 /home/hc/dataset/myobject/box_1_tracking
```

方式 4：如果重建数据集本身也包含 `depth/`，并且你就想直接用它作为 FoundationPose 跟踪数据集，需要显式加：

```bash
--allow_reconstruction_dataset_for_tracking
```

否则脚本默认不会把重建数据集本身当成跟踪数据集，这是为了避免不小心把重建阶段的深度图和跟踪阶段的深度图混用。

如果跟踪数据集的文件夹名称不同，可以改：

```bash
--tracking_left_subdir rgb \
--tracking_depth_subdir depth \
--tracking_mask_subdir masks \
--tracking_intrinsics_name cam_K.txt
```

## 三、最常用运行命令

### 1. 单个数据集完整评估

如果重建数据集和跟踪数据集是两个不同目录：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --tracking_dataset_root /home/hc/dataset/myobject/fluke_2_tracking \
  --batch_status_csv /home/hc/dataset/myobject/fluke_2/evaluate/batch_pipeline_status.csv
```

如果重建数据集本身就包含 `rgb/depth/masks/cam_K.txt`，并且想直接用它跑 FoundationPose 跟踪评估：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --allow_reconstruction_dataset_for_tracking \
  --batch_status_csv /home/hc/dataset/myobject/fluke_2/evaluate/batch_pipeline_status.csv
```

### 2. 先 dry-run 检查将要执行的命令

`--dry_run` 只打印命令，不真正执行，可以用来检查路径、conda 环境、BundleSDF 分支和参数是否正确。

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --allow_reconstruction_dataset_for_tracking \
  --dry_run
```

### 3. 指定 BundleSDF 输入图像短边长度

BundleSDF 的 `main` 和 `gaijin` 分支都会收到同一个短边参数：

```bash
--bundle_shorter_side 720
```

等价写法：

```bash
--bundlesdf_shorter_side 720
```

含义：

- `720`：把输入 RGB、depth、mask 按短边缩放到 720，再进行 BundleSDF 重建。
- `1080`：把输入按短边缩放到 1080，适合和原始 1920x1080 数据保持一致。
- `<=0`：不缩放，保持原始 RGB、depth、mask 分辨率。

例如完整使用 1080 短边重建和评估：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --allow_reconstruction_dataset_for_tracking \
  --bundle_shorter_side 1080 \
  --batch_status_csv /home/hc/dataset/myobject/fluke_2/evaluate/batch_pipeline_status_1080.csv
```

### 4. 只在已有评估结果基础上重跑改进版 BundleSDF

如果已经跑过深度 benchmark 和 `main` 分支重建，只想使用最佳深度组重跑 `gaijin` 分支，例如使用 `confidence_photo_only`，并指定 BundleSDF 短边为 1080：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --skip_depth_benchmark \
  --skip_main_reconstruction \
  --skip_foundationpose \
  --best_method confidence_photo_only \
  --bundle_shorter_side 1080 \
  --overwrite \
  --batch_status_csv /home/hc/dataset/myobject/fluke_2/evaluate/rerun_gaijin_1080_status.csv
```

这个命令会复用已有的：

```text
evaluate/depth_benchmark/
evaluate/reconstruction_main/summary.csv
```

并重新生成或覆盖：

```text
evaluate/bundlesdf_gaijin/confidence_photo_only/
evaluate/reconstruction_gaijin/confidence_photo_only.csv
evaluate/reconstruction_gaijin/summary.csv
evaluate/reconstruction_mesh_comparison.csv
```

如果还想接着用新的 `gaijin` mesh 重跑 FoundationPose 对比，不要加 `--skip_foundationpose`：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --skip_depth_benchmark \
  --skip_main_reconstruction \
  --best_method confidence_photo_only \
  --bundle_shorter_side 1080 \
  --overwrite \
  --allow_reconstruction_dataset_for_tracking \
  --batch_status_csv /home/hc/dataset/myobject/fluke_2/evaluate/rerun_gaijin_1080_with_fp_status.csv
```

### 5. 只跑深度 benchmark

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --skip_main_reconstruction \
  --skip_gaijin_reconstruction \
  --skip_foundationpose
```

注意：当前脚本在跳过 `main` 重建后仍然会尝试选择最佳深度组。如果只是想单独跑深度 benchmark，更稳妥的方式是直接运行 FFS 的离线 launch，或者在已有 `reconstruction_main/summary.csv` 的情况下配合 `--best_method` 使用。

### 6. 批量评估多个数据集

方式 1：命令行直接列出多个数据集：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  /home/hc/dataset/myobject/box_1 \
  --tracking_dataset_map /home/hc/dataset/myobject/tracking_map.csv
```

方式 2：用文本文件列出数据集：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  --dataset_list /home/hc/dataset/myobject/eval_list.txt \
  --tracking_dataset_map /home/hc/dataset/myobject/tracking_map.csv
```

`eval_list.txt` 每行一个重建数据集路径：

```text
/home/hc/dataset/myobject/fluke_2
/home/hc/dataset/myobject/box_1
/home/hc/dataset/myobject/cup_3
```

方式 3：扫描父目录下所有符合结构的数据集：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  --dataset_parent /home/hc/dataset/myobject \
  --sequence_glob "fluke_*" \
  --tracking_dataset_map /home/hc/dataset/myobject/tracking_map.csv
```

## 四、关键参数说明

### 1. 数据集和输出目录参数

```text
datasets
```

位置参数，可以传一个或多个重建数据集路径。如果不传，脚本默认使用 `/home/hc/weizi/dataset/jrnew-blue`。

```text
--evaluate_dir evaluate
```

每个数据集下面的总输出目录。默认是相对路径 `evaluate`，即：

```text
<dataset_root>/evaluate
```

也可以指定绝对路径：

```bash
--evaluate_dir /home/hc/dataset/eval_outputs/fluke_2_1080
```

```text
--depth_benchmark_dir
```

深度 benchmark 输出目录。默认留空，表示：

```text
<evaluate_dir>/depth_benchmark
```

如果手动指定相对路径，会解析到数据集目录下面；如果指定绝对路径，会直接使用该绝对路径。

```text
--batch_status_csv batch_pipeline_status.csv
```

批量任务总状态 CSV。默认相对于当前运行命令时的工作目录。建议显式写到 evaluate 目录，避免散落在项目根目录：

```bash
--batch_status_csv /home/hc/dataset/myobject/fluke_2/evaluate/batch_pipeline_status.csv
```

### 2. 跳过和覆盖参数

```text
--skip_depth_benchmark
```

跳过 FFS 离线深度 benchmark，复用已有 `evaluate/depth_benchmark/`。

```text
--skip_main_reconstruction
```

跳过 BundleSDF `main` 分支重建，复用已有 `evaluate/reconstruction_main/summary.csv` 和 `evaluate/bundlesdf_main/`。

```text
--skip_gaijin_reconstruction
```

跳过 BundleSDF `gaijin` 分支重建。

```text
--skip_foundationpose
```

跳过 FoundationPose 跟踪评估。

```text
--overwrite
```

即使输出已经存在也重新运行对应阶段。重跑某一阶段时通常需要加它，否则脚本看到 summary 已存在会认为该阶段已经完成。

```text
--dry_run
```

只打印将要执行的命令，不真正运行。调试路径和参数时非常有用。

```text
--stop_on_error
```

遇到错误立即停止。默认是不加该参数时记录错误，然后继续处理后续数据集。

### 3. BundleSDF 参数

```text
--bundlesdf_root /home/hc/weizi/BundleSDF
```

BundleSDF 仓库路径。脚本会在这个仓库里自动切换分支并调用 `batch_depth_benchmark.py`。

```text
--bundlesdf_main_branch main
--bundlesdf_gaijin_branch gaijin
```

改进前和改进后的 BundleSDF 分支名称。默认分别为 `main` 和 `gaijin`。

```text
--keep_bundlesdf_branch
```

默认情况下，脚本结束后会把 BundleSDF 仓库切回运行脚本前所在的分支。加上这个参数后，会停留在最后使用的分支。

```text
--allow_dirty_bundlesdf
```

允许 BundleSDF 仓库在有未提交修改时切换分支。一般不建议加，除非确认当前修改不会影响切分支。

```text
--bundle_shorter_side 720
```

BundleSDF 输入 RGB、depth、mask 的短边缩放长度。这个参数会同时传给 `main` 和 `gaijin` 分支，用来保证两边重建分辨率策略一致。

```text
--bundle_device cuda
--bundle_stride 1
--bundle_debug_level 2
--bundle_use_gui 0
--bundle_use_segmenter 1
```

BundleSDF 运行相关参数。一般保持默认即可。

### 4. 最佳深度组选择参数

```text
--best_metric composite
```

默认使用综合排名选择最佳深度组。综合指标会参考：

- 越大越好：`psnr`、`ssim`、`mask_iou`、`coverage_5mm`、`coverage_1cm`
- 越小越好：`lpips`、`depth_rmse`、`obs_to_mesh_mean_dist`

脚本会对每个指标分别排名，再取平均排名最好的 method 作为最佳深度组。

```text
--best_metric psnr
--best_metric_direction max
```

也可以指定单个指标选择最佳方法。对于脚本认识的指标，`--best_metric_direction auto` 会自动判断越大越好还是越小越好；如果是自定义列名，需要手动指定 `min` 或 `max`。

```text
--best_method confidence_photo_only
```

手动指定最佳深度组，跳过自动选择。适合在已有结果基础上重跑某个阶段。

### 5. FFS 深度 benchmark 参数

默认会执行：

```text
ros2 launch fast_foundation_stereo offline_depth_benchmark.launch.py
```

常用参数：

```text
--depth_launch_package fast_foundation_stereo
--depth_launch_file offline_depth_benchmark.launch.py
--engine_file_path [...]
--stereo_model_type FAST_FOUNDATION_STEREO
--input_image_width 1920
--input_image_height 1080
--model_input_width 640
--model_input_height 448
--min_depth_meters 0.1
--max_depth_meters 100.0
--depth_scale 1000.0
--max_pairs 1000
--threshold_sweep 0.20,0.30,0.35,0.40,0.50
```

如果需要向 ROS2 launch 额外传参，可以重复使用：

```bash
--depth_launch_arg name:=value
```

例如：

```bash
--depth_launch_arg some_param:=123 \
--depth_launch_arg another_param:=true
```

### 6. ROS 环境参数

默认情况下，脚本会自动 source：

```text
/opt/ros/<最新版本>/setup.bash
<ffs+fp+sam>/install/setup.bash
```

如果需要手动指定 ROS setup 文件，可以使用：

```bash
--ros_setup /opt/ros/humble/setup.bash \
--ros_setup /home/hc/weizi/ffs+fp+sam/install/setup.bash
```

### 7. conda 环境参数

默认使用：

```text
--conda_prefix ~/anaconda3
--bundlesdf_conda_env bundlesdf
--foundationpose_conda_env fp_ros
```

脚本实际调用形式类似：

```bash
~/anaconda3/bin/conda run --no-capture-output -n bundlesdf python ...
~/anaconda3/bin/conda run --no-capture-output -n fp_ros python ...
```

如果本机环境名不同，需要显式修改。

### 8. FoundationPose 参数

```text
--foundationpose_root /home/hc/weizi/FoundationPoseROS2
```

FoundationPose 评估脚本所在仓库路径。脚本会调用：

```text
evaluate_pose_tracking_no_gt.py
```

```text
--foundationpose_depth_source tracking_dataset
```

FoundationPose 使用的深度来源。可选值：

- `tracking_dataset`：默认值，使用单独 RGBD 跟踪数据集里的 `depth/`。
- `selected_depth`：使用 FFS benchmark 选中的原始深度组。
- `main_filtered_depth`：使用 BundleSDF main 重建输出的 `depth_filtered/`。
- `variant_filtered_depth`：main mesh 用 main 的 `depth_filtered/`，gaijin mesh 用 gaijin 的 `depth_filtered/`。

FoundationPose 会对两个 mesh variant 都跑两种 runner：

```text
main / standard
main / adaptive_filter
gaijin / standard
gaijin / adaptive_filter
```

常用跟踪参数：

```text
--fp_shorter_side 360
--fp_num_points 2000
--fp_depth_min 0.001
--fp_depth_max 0.0
--fp_render_mode points
--fp_icp_max_iterations 20
--fp_icp_max_correspondence_dist 0.05
--fp_est_refine_iter 5
--fp_track_refine_iter 2
```

`--fp_render_mode` 控制 no-GT 指标里 mesh 投影渲染的方式：

- `points`：默认值，用采样点投影，适合 1000 帧以上的长序列，速度稳定。
- `faces`：使用三角面软件光栅化，更接近完整 mesh 投影，但在 1080p 长序列上会非常慢，外层日志可能长时间停在同一行。

需要调试或控制评估规模时可以加：

```bash
--fp_verbose \
--fp_debug 1 \
--fp_progress_interval 20 \
--fp_max_frames 100
```

`--fp_overwrite_metrics` 用于只重算 FoundationPose 的 no-GT 指标，保留已有 `ob_in_cam` 位姿并自动传递 `--skip_existing_pred`。这适合已经跑完 tracking，但后来修改了 mask、评估参数或渲染模式，只想刷新 `summary.json` 和 `per_frame_metrics.csv` 的情况。

## 五、输出目录和文件含义

默认输出结构如下：

```text
evaluate/
  depth_benchmark/
  bundlesdf_main/
  reconstruction_main/
  bundlesdf_gaijin/
  reconstruction_gaijin/
  reconstruction_mesh_comparison.csv
  best_depth_method.json
  foundationpose_scenes/
  foundationpose/
  foundationpose_tracking_summary.json
  foundationpose_tracking_summary.csv
  pipeline_status.json
  logs/
    pipeline/
```

### 1. `depth_benchmark/`

FFS 离线深度 benchmark 的输出目录。里面通常包含多组不同后处理策略或置信度阈值生成的深度图目录。每个子目录就是一个 depth method，后续 BundleSDF 会把每个 method 当成一组候选深度输入。

例如：

```text
evaluate/depth_benchmark/confidence_photo_only/
evaluate/depth_benchmark/raw/
evaluate/depth_benchmark/conf_0.35/
```

实际 method 名以离线深度 benchmark 脚本输出为准。

### 2. `bundlesdf_main/`

BundleSDF `main` 分支的重建结果目录。每个 depth method 对应一个子目录：

```text
evaluate/bundlesdf_main/<method>/
```

常见关键文件或目录：

```text
textured_mesh.obj       # 该 method 重建出的 mesh，后续用于 FoundationPose 或 mesh 指标评估
depth_filtered/         # BundleSDF 内部滤波后的深度图
ob_in_cam/              # BundleSDF 估计出的物体位姿序列
```

具体文件会随 BundleSDF 自身输出而变化。

### 3. `reconstruction_main/`

BundleSDF `main` 分支的重建质量评估结果。

关键文件：

```text
summary.csv
```

每一行对应一个 depth method，列里是重建质量指标，例如：

```text
method
psnr
ssim
lpips
mask_iou
depth_rmse
coverage_5mm
coverage_1cm
obs_to_mesh_mean_dist
```

脚本默认从这个文件中选出最佳 depth method。

另外还可能包含：

```text
<method>.csv
```

表示某一个 depth method 更详细的逐帧或细分评估结果，具体列由 BundleSDF 的 `batch_depth_benchmark.py` 和评估逻辑决定。

### 4. `best_depth_method.json`

记录脚本选出的最佳深度组。

如果是自动选择，里面会包含：

```text
best_method
selection_mode
metrics
average_rank
row
all_scores
```

其中：

- `best_method`：最终选中的深度组名称。
- `selection_mode`：选择方式，常见是 `composite_rank`、`single_metric` 或 `manual`。
- `metrics`：综合排名使用了哪些指标，以及每个指标是越大越好还是越小越好。
- `average_rank`：最佳 method 的平均排名。
- `row`：该 method 在 `reconstruction_main/summary.csv` 中的原始指标。
- `all_scores`：所有 method 的平均排名，方便检查为什么某组被选中。

如果你传了 `--best_method`，这里会记录为手动选择。

### 5. `bundlesdf_gaijin/`

BundleSDF 改进分支的重建结果目录。默认只会重建最佳 depth method：

```text
evaluate/bundlesdf_gaijin/<best_method>/
```

关键文件与 `bundlesdf_main/<method>/` 类似，例如：

```text
textured_mesh.obj
depth_filtered/
ob_in_cam/
```

### 6. `reconstruction_gaijin/`

BundleSDF 改进分支的重建质量评估结果。

关键文件：

```text
summary.csv
<best_method>.csv
```

因为改进分支默认只跑最佳 method，所以这里通常只有一个 method 的结果。

如果这里的指标全是 `NaN`，常见原因包括：

- BundleSDF 输出没有有效 mesh 或有效点。
- 深度图、mask、相机内参与重建分辨率不匹配。
- `main` 和 `gaijin` 的输入缩放策略不一致。
- 需要显式指定一致的 `--bundle_shorter_side`，例如 `720` 或 `1080`。

### 7. `reconstruction_mesh_comparison.csv`

改进前后 BundleSDF mesh 的对比表。

通常包含两行：

```text
variant,method,...
main,<best_method>,...
gaijin,<best_method>,...
```

用途：

- 快速比较 `main` 和 `gaijin` 在同一最佳深度组上的重建质量。
- 检查改进分支是否优于原始分支。
- 检查是否出现 `NaN`、有效点数为 0 等异常。

### 8. `foundationpose_scenes/`

FoundationPose 评估前准备好的场景目录。脚本会在这里创建软链接，把 RGB、depth、mask 和 `cam_K.txt` 整理成 FoundationPose 评估脚本需要的输入结构。

目录名包含：

```text
<method>__<mesh_variant>__<depth_source>__<tracking_dataset_name>
```

例如：

```text
confidence_photo_only__gaijin__tracking_dataset__fluke_2
```

它一般不是最终指标，而是 FoundationPose 的输入桥接目录。

### 9. `foundationpose/`

FoundationPose 跟踪评估输出目录，按 mesh variant 和 runner 分开：

```text
evaluate/foundationpose/main/standard/
evaluate/foundationpose/main/adaptive_filter/
evaluate/foundationpose/gaijin/standard/
evaluate/foundationpose/gaijin/adaptive_filter/
```

每个目录下会有 FoundationPose 自己的 debug 和评估结果。脚本判断是否完成时主要检查：

```text
foundationpose_debug/summary.json
foundationpose_debug/per_frame_metrics.csv
```

或者 adaptive filter 对应：

```text
foundationpose_adaptive_filter_debug/summary.json
foundationpose_adaptive_filter_debug/per_frame_metrics.csv
```

其中：

- `summary.json`：该 runner 的总体跟踪指标。
- `per_frame_metrics.csv`：逐帧跟踪指标。

### 10. `foundationpose_tracking_summary.json`

FoundationPose 四组实验的总汇总文件，推荐优先查看这个 JSON。它使用缩进格式保存，复制到聊天窗口、文档或脚本里都比 CSV 更方便。

顶层通常包含：

```text
dataset
evaluate_root
tracking_dataset
generated_at
summary
results
results_list
csv_path
```

其中 `summary` 是最方便人工查看的短摘要列表，每一项通常包含：

```text
mesh_variant
runner
tracking_dataset
num_frames_total
num_frames_evaluated
render_mode
mask_iou_mean
mask_iou_median
depth_error_mean_m
depth_error_median_m
depth_error_mean_cm
chamfer_distance_mean
icp_residual_mean
translation_increment_mean_m
translation_increment_mean_cm
rotation_increment_mean_deg
mesh_path
pred_pose
summary_path
per_frame_csv
```

`results` 按下面结构保存完整指标，适合精确查某一组实验：

```text
results.main.standard
results.main.adaptive_filter
results.gaijin.standard
results.gaijin.adaptive_filter
```

`results_list` 是完整指标的列表形式，适合脚本遍历处理。

用途：

- 比较 `main` mesh 和 `gaijin` mesh 对 FoundationPose 跟踪质量的影响。
- 比较 `standard` 和 `adaptive_filter` runner 的稳定性。
- 快速定位每组实验的详细输出目录。

### 11. `foundationpose_tracking_summary.csv`

FoundationPose 四组实验的 CSV 兼容汇总表，字段与 JSON 中的 `results_list` 基本一致。它适合用表格软件或 Pandas 读取，但不如 JSON 方便人工复制查看。

### 12. `pipeline_status.json`

单个数据集的流水线状态文件。脚本每完成一个阶段都会更新它。

常见字段：

```text
dataset
output_dir
tracking_dataset
depth_benchmark_status
main_reconstruction_status
best_method
gaijin_reconstruction_status
foundationpose_status
elapsed_seconds
message
```

状态值常见有：

- `pending`：尚未执行。
- `completed`：已执行完成。
- `already_complete`：检测到已有输出，跳过该阶段。
- `skipped_by_option`：用户通过 `--skip_*` 主动跳过。
- `dry_run`：dry-run 模式。
- `failed`：FoundationPose 某些 runner 执行失败。

如果某个阶段报错，错误信息会写入 `message`。

### 12. `logs/pipeline/`

所有长时间运行子进程的日志目录。脚本会把子进程 stdout/stderr 写入这里，方便中断后检查。

常见日志：

```text
depth_benchmark.log
bundlesdf_main.log
bundlesdf_gaijin.log
foundationpose_main_standard.log
foundationpose_main_adaptive_filter.log
foundationpose_gaijin_standard.log
foundationpose_gaijin_adaptive_filter.log
```

如果脚本长时间卡住、输出 `NaN`、某个阶段失败，优先查看这里对应阶段的日志。

### 13. `batch_pipeline_status.csv`

批量运行的总状态 CSV。默认写到运行命令时的当前目录：

```text
batch_pipeline_status.csv
```

建议运行时用 `--batch_status_csv` 显式指定到某个 evaluate 目录。它每一行对应一个数据集，字段与 `pipeline_status.json` 类似，适合批量任务结束后快速查看哪些数据集失败。

## 六、恢复和重跑建议

### 1. 已有输出时的默认行为

如果不加 `--overwrite`，脚本会尽量复用已有结果：

- 深度 benchmark：检查 `depth_benchmark/` 下是否有足够数量的完整深度组。
- BundleSDF：检查对应 `reconstruction_*/summary.csv` 是否存在 method 行。
- FoundationPose：检查 `summary.json` 和 `per_frame_metrics.csv` 是否存在。

因此，中断后再次运行同一命令，通常会从未完成的阶段继续。

### 2. 只重跑某一阶段

常用策略是配合 `--skip_*` 和 `--overwrite`。

只重跑 gaijin 重建：

```bash
--skip_depth_benchmark \
--skip_main_reconstruction \
--skip_foundationpose \
--best_method <method> \
--overwrite
```

只重跑 FoundationPose：

```bash
--skip_depth_benchmark \
--skip_main_reconstruction \
--skip_gaijin_reconstruction \
--best_method <method> \
--overwrite
```

只重算 FoundationPose 评估指标，不重新跑 tracking、不重新生成 `ob_in_cam`：

```bash
--skip_depth_benchmark \
--skip_main_reconstruction \
--skip_gaijin_reconstruction \
--best_method <method> \
--fp_overwrite_metrics
```

注意：如果跳过了 `main` 重建，但没有传 `--best_method`，脚本仍然需要从 `reconstruction_main/summary.csv` 自动选择最佳 method；如果这个文件不存在，就会失败。

### 3. 公平比较 BundleSDF main 和 gaijin

比较两个分支时要保证输入分辨率策略一致，建议显式指定：

```bash
--bundle_shorter_side 720
```

或者：

```bash
--bundle_shorter_side 1080
```

如果想完全重新生成一套 1080 短边结果，建议使用新的输出目录，避免和已有 720 结果混在一起：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py \
  /home/hc/dataset/myobject/fluke_2 \
  --allow_reconstruction_dataset_for_tracking \
  --evaluate_dir evaluate_1080 \
  --bundle_shorter_side 1080 \
  --batch_status_csv /home/hc/dataset/myobject/fluke_2/evaluate_1080/batch_pipeline_status.csv
```

### 4. 出现 NaN 时先检查什么

优先检查：

```bash
cat /home/hc/dataset/myobject/fluke_2/evaluate/pipeline_status.json
cat /home/hc/dataset/myobject/fluke_2/evaluate/best_depth_method.json
cat /home/hc/dataset/myobject/fluke_2/evaluate/reconstruction_main/summary.csv
cat /home/hc/dataset/myobject/fluke_2/evaluate/reconstruction_gaijin/summary.csv
cat /home/hc/dataset/myobject/fluke_2/evaluate/reconstruction_mesh_comparison.csv
```

再看对应日志：

```bash
less /home/hc/dataset/myobject/fluke_2/evaluate/logs/pipeline/bundlesdf_gaijin.log
less /home/hc/dataset/myobject/fluke_2/evaluate/logs/pipeline/foundationpose_gaijin_standard.log
```

如果 `reconstruction_gaijin/summary.csv` 某行都是 `NaN`，重点确认：

- `evaluate/bundlesdf_gaijin/<method>/textured_mesh.obj` 是否存在且正常。
- `evaluate/bundlesdf_gaijin/<method>/depth_filtered/` 是否有有效深度图。
- `cam_K.txt` 是否和 RGB/depth/mask 的实际分辨率匹配。
- `--bundle_shorter_side` 是否与想比较的分支保持一致。

## 七、查看帮助

脚本自带完整参数帮助：

```bash
python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_evaluate_stack.py --help
```
