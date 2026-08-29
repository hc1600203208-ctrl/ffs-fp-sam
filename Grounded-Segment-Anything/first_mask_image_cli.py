#!/usr/bin/env python3
import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_DIR / "GroundingDINO"))
sys.path.append(str(REPO_DIR / "segment_anything"))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, sam_hq_model_registry, sam_model_registry


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_rgb_image_file(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3:
        raise RuntimeError(f"Unsupported image shape from {image_path}: {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    raise RuntimeError(f"Unsupported image channel count from {image_path}: {image.shape[2]}")


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


def load_grounding_model(config_path: str,
                         checkpoint_path: str,
                         bert_base_uncased_path: str,
                         device: str):
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


def save_mask(mask: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.ascontiguousarray(mask.astype(np.uint8))
    if not cv2.imwrite(str(output_path), mask):
        raise RuntimeError(f"Failed to write mask to {output_path}")


@torch.inference_mode()
def segment_first_frame(rgb: np.ndarray, args) -> tuple[np.ndarray, list[str]]:
    image_pil, image_tensor = load_image_from_rgb(rgb)
    grounding_model = load_grounding_model(
        args.config_path,
        args.grounded_checkpoint,
        args.bert_base_uncased_path,
        args.device,
    )

    if args.use_sam_hq:
        sam = sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint)
    else:
        sam = sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint)
    predictor = SamPredictor(sam.to(args.device))

    try:
        boxes, phrases = get_grounding_output(
            grounding_model,
            image_tensor,
            args.text_prompt,
            args.box_threshold,
            args.text_threshold,
            args.device,
        )

        height, width = rgb.shape[:2]
        if boxes.size(0) == 0:
            return np.zeros((height, width), dtype=np.uint8), []

        predictor.set_image(rgb)
        width_pil, height_pil = image_pil.size
        scale = torch.tensor([width_pil, height_pil, width_pil, height_pil], dtype=torch.float32)
        boxes = boxes * scale
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes,
            rgb.shape[:2],
        ).to(args.device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return masks_to_binary_mask(masks, height, width), phrases
    finally:
        grounding_model.to("cpu")
        predictor.model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one first-frame binary mask with GroundingDINO + SAM from an image file."
    )
    parser.add_argument("--image_path", required=True, help="Input first-frame image path.")
    parser.add_argument("--output_path", required=True, help="Output mask image path, usually first_mask.png.")
    parser.add_argument(
        "--config_path",
        default=str(REPO_DIR / "GroundingDINO" / "groundingdino" / "config" /
                    "GroundingDINO_SwinT_OGC.py"),
    )
    parser.add_argument("--grounded_checkpoint", default=str(REPO_DIR / "groundingdino_swint_ogc.pth"))
    parser.add_argument("--sam_version", default="vit_b")
    parser.add_argument("--sam_checkpoint", default=str(REPO_DIR / "sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam_hq_checkpoint", default="")
    parser.add_argument("--use_sam_hq", type=str_to_bool, default=False)
    parser.add_argument("--text_prompt", default="blue object")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bert_base_uncased_path", default="")
    parser.add_argument(
        "--fail_on_empty_mask",
        type=str_to_bool,
        default=False,
        help="Exit non-zero if GroundingDINO finds no boxes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path).expanduser()
    output_path = Path(args.output_path).expanduser()

    rgb = load_rgb_image_file(image_path)
    mask, phrases = segment_first_frame(rgb, args)
    if not phrases and args.fail_on_empty_mask:
        raise RuntimeError(
            f"No object detected for prompt '{args.text_prompt}' in {image_path}; mask not saved."
        )

    save_mask(mask, output_path)
    result = {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "positive_pixels": int(np.count_nonzero(mask)),
        "phrases": phrases,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
