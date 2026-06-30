#!/usr/bin/env python3
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage


REPO_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_DIR / "GroundingDINO"))
sys.path.append(str(REPO_DIR / "segment_anything"))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, sam_hq_model_registry, sam_model_registry


def load_image_from_rgb(rgb_image: np.ndarray):
    image_pil = Image.fromarray(rgb_image).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_grounding_model(config_path: str, checkpoint_path: str,
                         bert_base_uncased_path: str, device: str):
    args = SLConfig.fromfile(config_path)
    args.device = device
    if bert_base_uncased_path:
        args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.to(device).eval()


@torch.inference_mode()
def get_grounding_output(model, image, caption, box_threshold, text_threshold, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    image = image.to(device)
    outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    keep = logits.max(dim=1)[0] > box_threshold
    logits = logits[keep]
    boxes = boxes[keep]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = []
    for logit in logits:
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        phrases.append(f"{phrase}({logit.max().item():.3f})")

    return boxes, phrases


def masks_to_binary_mask(masks: torch.Tensor, height: int, width: int) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    if masks.numel() == 0:
        return output

    for mask in masks:
        mask_np = mask.squeeze().detach().cpu().numpy().astype(bool)
        output[mask_np] = 255
    return output


def mono8_image_msg(mask: np.ndarray, header) -> RosImage:
    mask = np.ascontiguousarray(mask.astype(np.uint8))
    msg = RosImage()
    msg.header = header
    msg.height = mask.shape[0]
    msg.width = mask.shape[1]
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = mask.shape[1]
    msg.data = mask.tobytes()
    return msg


def ros_image_to_rgb_array(msg: RosImage) -> np.ndarray:
    encoding = msg.encoding.lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    channels = channels_by_encoding[encoding]
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)

    if encoding in ("rgb8", "8uc3"):
        rgb = image
    elif encoding == "bgr8":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding == "rgba8":
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif encoding in ("bgra8", "8uc4"):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image.reshape(msg.height, msg.width), cv2.COLOR_GRAY2RGB)

    return np.ascontiguousarray(rgb)


class FirstMaskGroundedSamNode(Node):
    def __init__(self):
        super().__init__("first_mask_grounded_sam")
        self.processed = False

        self.declare_parameter("image_topic", "/left/image_raw")
        self.declare_parameter("mask_topic", "/fisrt_mask")
        self.declare_parameter("publish_mask_topic", True)
        self.declare_parameter("output_dir", str(REPO_DIR / "ros2_outputs" / "first_mask"))
        self.declare_parameter("output_name", "first_mask.png")
        self.declare_parameter(
            "config_path",
            str(REPO_DIR / "GroundingDINO" / "groundingdino" / "config" /
                "GroundingDINO_SwinT_OGC.py"),
        )
        self.declare_parameter("grounded_checkpoint", str(REPO_DIR / "groundingdino_swint_ogc.pth"))
        self.declare_parameter("sam_version", "vit_b")
        self.declare_parameter("sam_checkpoint", str(REPO_DIR / "sam_vit_b_01ec64.pth"))
        self.declare_parameter("sam_hq_checkpoint", "")
        self.declare_parameter("use_sam_hq", False)
        self.declare_parameter("text_prompt", "blue object")
        self.declare_parameter("box_threshold", 0.3)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("bert_base_uncased_path", "")

        self.image_topic = self.get_parameter("image_topic").value
        self.mask_topic = self.get_parameter("mask_topic").value
        self.publish_mask_topic = bool(self.get_parameter("publish_mask_topic").value)
        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser()
        self.output_name = self.get_parameter("output_name").value
        self.config_path = self.get_parameter("config_path").value
        self.grounded_checkpoint = self.get_parameter("grounded_checkpoint").value
        self.sam_version = self.get_parameter("sam_version").value
        self.sam_checkpoint = self.get_parameter("sam_checkpoint").value
        self.sam_hq_checkpoint = self.get_parameter("sam_hq_checkpoint").value
        self.use_sam_hq = self.get_parameter("use_sam_hq").value
        self.text_prompt = self.get_parameter("text_prompt").value
        self.box_threshold = float(self.get_parameter("box_threshold").value)
        self.text_threshold = float(self.get_parameter("text_threshold").value)
        self.device = self.get_parameter("device").value
        self.bert_base_uncased_path = self.get_parameter("bert_base_uncased_path").value

        self.output_dir.mkdir(parents=True, exist_ok=True)

        mask_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.mask_pub = None
        if self.publish_mask_topic:
            self.mask_pub = self.create_publisher(RosImage, self.mask_topic, mask_qos)

        self.get_logger().info("Loading GroundingDINO and SAM models...")
        self.model = load_grounding_model(
            self.config_path,
            self.grounded_checkpoint,
            self.bert_base_uncased_path,
            self.device,
        )
        if self.use_sam_hq:
            sam = sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint)
        else:
            sam = sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint)
        self.predictor = SamPredictor(sam.to(self.device))
        self.get_logger().info(
            f"Models ready. Waiting for first frame on {self.image_topic}; prompt='{self.text_prompt}'"
        )

        self.image_sub = self.create_subscription(
            RosImage,
            self.image_topic,
            self.image_callback,
            image_qos,
        )

    def image_callback(self, msg: RosImage):
        if self.processed:
            return
        self.processed = True

        try:
            rgb = self.ros_image_to_rgb(msg)
            mask = self.segment_first_frame(rgb)
            path = self.save_mask(mask)
            self.publish_mask(mask, msg.header)
            if self.publish_mask_topic:
                self.get_logger().info(
                    f"Published first mask to {self.mask_topic} and saved {path}. Node is now idle."
                )
            else:
                self.get_logger().info(f"Saved first mask to {path}. Node is now idle.")
        except Exception as exc:
            self.processed = False
            self.get_logger().error(f"Failed to process first frame: {exc}")
            return

        self.release_models()
        self.destroy_subscription(self.image_sub)

    def ros_image_to_rgb(self, msg: RosImage) -> np.ndarray:
        return ros_image_to_rgb_array(msg)

    @torch.inference_mode()
    def segment_first_frame(self, rgb: np.ndarray) -> np.ndarray:
        image_pil, image_tensor = load_image_from_rgb(rgb)
        boxes, phrases = get_grounding_output(
            self.model,
            image_tensor,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold,
            self.device,
        )

        height, width = rgb.shape[:2]
        if boxes.size(0) == 0:
            self.get_logger().warn("No object detected; publishing an empty mask.")
            return np.zeros((height, width), dtype=np.uint8)

        self.predictor.set_image(rgb)
        width_pil, height_pil = image_pil.size
        scale = torch.tensor([width_pil, height_pil, width_pil, height_pil], dtype=torch.float32)
        boxes = boxes * scale
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes,
            rgb.shape[:2],
        ).to(self.device)
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        self.get_logger().info(f"Detected {masks.shape[0]} mask(s): {', '.join(phrases)}")
        return masks_to_binary_mask(masks, height, width)

    def save_mask(self, mask: np.ndarray) -> Path:
        name = self.output_name
        if not name.lower().endswith(".png"):
            name = f"{name}.png"
        path = self.output_dir / name
        if not cv2.imwrite(str(path), mask):
            raise RuntimeError(f"Failed to write mask to {path}")
        return path

    def publish_mask(self, mask: np.ndarray, header):
        if self.mask_pub is not None:
            self.mask_pub.publish(mono8_image_msg(mask, header))

    def release_models(self):
        try:
            if self.model is not None:
                self.model.to("cpu")
            if self.predictor is not None and hasattr(self.predictor, "model"):
                self.predictor.model.to("cpu")
        except Exception as exc:
            self.get_logger().warn(f"Model release warning: {exc}")

        self.model = None
        self.predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    rclpy.init()
    node = FirstMaskGroundedSamNode()
    try:
        while rclpy.ok() and not node.processed:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
