ros2 launch fast_foundation_stereo offline_stereo_depth.launch.py \
dataset_root:=/home/hc/dataset/myobject/gold_fast2 \
caminfo_path:=/home/hc/dataset/myobject/penqi/jr714.txt

ros2 launch fast_foundation_stereo offline_stereo_depth.launch.py \
dataset_root:=/home/hc/picture \
caminfo_path:=/home/hc/dataset/myobject/penqi/jr714.txt

python3 /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/depth_inspector.py --rgb-dir /home/hc/dataset/myobject/fluke_2/rgb --depth-dir /home/hc/dataset/myobject/fluke_2/depth

python /home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/scripts/batch_stereo_rectification_vis.py --caminfo-path /home/hc/dataset/myobject/fluke/jr714.txt --left-dir /home/hc/dataset/myobject/fluke/rgb --right-dir /home/hc/dataset/myobject/fluke/camera2 --output-dir /home/hc/dataset/myobject/fluke/rect

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

python /home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/tools/export_trimesh_oriented_bounds.py /home/hc/weizi/dataset/jrnew-blue/mesh1/textured_mesh.obj --output-dir /home/hc/weizi/dataset/jrnew-blue/mesh1

ros2 launch foundationpose_cpp foundationpose_stereo_tracker_fast.launch.py \
  run_first_mask:=true \
  text_prompt:="blue carton" \
  bert_base_uncased_path:=/home/hc/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594

  ros2 launch foundationpose_cpp foundationpose_stereo_tracker.launch.py \
  run_first_mask:=true \
  text_prompt:="blue carton" \
  bert_base_uncased_path:=/home/hc/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594
