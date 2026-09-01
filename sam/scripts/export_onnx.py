#!/usr/bin/env python3
import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "Grounded-Segment-Anything"
sys.path.append(str(REPO / "GroundingDINO"))
sys.path.append(str(REPO / "segment_anything"))

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from segment_anything import sam_model_registry
from groundingdino.util.misc import NestedTensor, inverse_sigmoid
from groundingdino.models.GroundingDINO.backbone.swin_transformer import window_partition


@dataclass
class PromptExportSpec:
    prompt: str
    max_text_len: int = 256


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_grounding_model(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    bert_base_uncased_path: str = "",
):
    args = SLConfig.fromfile(str(config_path))
    args.device = device
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    if bert_base_uncased_path:
        args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
        if hasattr(module, "use_transformer_ckpt"):
            module.use_transformer_ckpt = False
    model.eval().to(device)
    return model

class FixedPositionEmbedding(torch.nn.Module):
    def __init__(self, position_embedding, feature_shapes, device):
        super().__init__()
        for index, (height, width) in enumerate(feature_shapes):
            feature = torch.zeros(1, 1, height, width, device=device)
            mask = torch.zeros(1, height, width, device=device, dtype=torch.bool)
            with torch.no_grad():
                position = position_embedding(NestedTensor(feature, mask)).detach()
            self.register_buffer(f"position_{index}", position)

    def forward(self, tensor_list):
        height = tensor_list.tensors.shape[-2]
        width = tensor_list.tensors.shape[-1]
        for index in range(len(self._buffers)):
            position = getattr(self, f"position_{index}")
            if position.shape[-2] == height and position.shape[-1] == width:
                return position.expand(tensor_list.tensors.shape[0], -1, -1, -1)
        raise RuntimeError(f"Unexpected feature shape: {height}x{width}")


def _make_swin_attention_mask(height: int, width: int, window_size: int, shift_size: int):
    if shift_size <= 0:
        return None

    padded_h = int(np.ceil(height / window_size)) * window_size
    padded_w = int(np.ceil(width / window_size)) * window_size
    image_mask = torch.zeros((1, padded_h, padded_w, 1), dtype=torch.float32)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    count = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            image_mask[:, h_slice, w_slice, :] = count
            count += 1

    mask_windows = window_partition(image_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(
        attention_mask == 0, float(0.0)
    )
    return attention_mask


def _fixed_swin_basic_layer_forward(self, x, height, width):
    attention_mask = self.fixed_attention_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=x.device, dtype=x.dtype)

    for block in self.blocks:
        block.H, block.W = height, width
        x = block(x, attention_mask)

    if self.downsample is not None:
        x_down = self.downsample(x, height, width)
        out_h, out_w = (height + 1) // 2, (width + 1) // 2
        return x, height, width, x_down, out_h, out_w

    return x, height, width, x, height, width


def freeze_swin_attention_masks(model, image_height: int, image_width: int):
    patch_h = (image_height + 3) // 4
    patch_w = (image_width + 3) // 4
    current_h = patch_h
    current_w = patch_w

    for layer in getattr(model.backbone[0], "layers", []):
        attention_mask = _make_swin_attention_mask(
            current_h, current_w, layer.window_size, layer.shift_size
        )
        if attention_mask is None:
            layer.fixed_attention_mask = None
        else:
            layer.register_buffer("fixed_attention_mask", attention_mask, persistent=False)
        layer.forward = types.MethodType(_fixed_swin_basic_layer_forward, layer)

        if layer.downsample is not None:
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2


class GroundingDINOExportWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        caption: str,
        mask_input_type: str,
        fixed_mask: bool,
        dino_height: int,
        dino_width: int,
    ):
        super().__init__()
        self.model = model
        self.mask_input_type = mask_input_type
        self.fixed_mask = fixed_mask
        self.dino_height = dino_height
        self.dino_width = dino_width
        self.caption = caption.lower().strip()
        if not self.caption.endswith("."):
            self.caption += "."
        self.specical_tokens = self.model.specical_tokens

        tokenized = self.model.tokenizer([self.caption], padding="longest", return_tensors="pt")
        tokenized = tokenized.to(next(self.model.parameters()).device)
        (
            self.text_self_attention_masks,
            self.position_ids,
            _,
        ) = self._build_text_meta(tokenized)
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = self.text_self_attention_masks
        tokenized_for_encoder["position_ids"] = self.position_ids
        bert_output = self.model.bert(**tokenized_for_encoder)
        self.register_buffer("encoded_text", self.model.feat_map(bert_output["last_hidden_state"]).detach())
        self.register_buffer("text_token_mask", tokenized.attention_mask.bool().detach())
        self.register_buffer("position_ids_buffer", self.position_ids.detach())
        self.register_buffer(
            "text_self_attention_masks_buffer", self.text_self_attention_masks.detach()
        )

    def _build_text_meta(self, tokenized):
        from groundingdino.models.GroundingDINO.bertwarper import (
            generate_masks_with_special_tokens_and_transfer_map,
        )

        return generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.model.tokenizer
        )

    def forward(self, images, masks=None):
        if isinstance(images, (list, tuple)):
            images = torch.stack(images, dim=0)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if self.fixed_mask:
            masks = torch.zeros(
                (images.shape[0], self.dino_height, self.dino_width),
                device=images.device,
                dtype=torch.bool,
            )
        else:
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if self.mask_input_type == "float":
                masks = masks > 0.5

        text_dict = {
            "encoded_text": self.encoded_text.expand(images.shape[0], -1, -1),
            "text_token_mask": self.text_token_mask.expand(images.shape[0], -1),
            "position_ids": self.position_ids_buffer.expand(images.shape[0], -1),
            "text_self_attention_masks": self.text_self_attention_masks_buffer.expand(images.shape[0], -1, -1),
        }

        samples = NestedTensor(images, masks.bool())
        features, poss = self.model.backbone(samples)

        srcs = []
        masks_out = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.model.input_proj[l](src))
            masks_out.append(mask)
        if self.model.num_feature_levels > len(srcs):
            srcs_len = len(srcs)
            for l in range(srcs_len, self.model.num_feature_levels):
                if l == srcs_len:
                    src = self.model.input_proj[l](features[-1].tensors)
                else:
                    src = self.model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks_out.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.model.transformer(
            srcs, masks_out, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.model.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(self.model.class_embed, hs)]
        )
        return outputs_class[-1], outputs_coord_list[-1]


def export_grounding_dino(args):
    model = load_grounding_model(
        args.config_path,
        args.grounded_checkpoint,
        args.device,
        args.bert_base_uncased_path,
    )
    if args.fixed_mask:
        freeze_swin_attention_masks(model, args.dino_height, args.dino_width)
        dummy_mask_for_shapes = torch.zeros(
            1, args.dino_height, args.dino_width, device=args.device, dtype=torch.bool
        )
        with torch.no_grad():
            features = model.backbone(
                NestedTensor(
                    torch.zeros(
                        1, 3, args.dino_height, args.dino_width, device=args.device
                    ),
                    dummy_mask_for_shapes,
                )
            )[0]
        feature_shapes = [
            (feature.tensors.shape[-2], feature.tensors.shape[-1]) for feature in features
        ]
        last_height, last_width = feature_shapes[-1]
        feature_shapes.append(((last_height + 1) // 2, (last_width + 1) // 2))
        model.backbone[1] = FixedPositionEmbedding(
            model.backbone[1], feature_shapes, args.device
        )
    wrapper = GroundingDINOExportWrapper(
        model,
        args.text_prompt,
        args.mask_input_type,
        args.fixed_mask,
        args.dino_height,
        args.dino_width,
    )
    wrapper.eval()

    dummy_image = torch.zeros(1, 3, args.dino_height, args.dino_width, device=args.device)
    mask_dtype = torch.float32 if args.mask_input_type == "float" else torch.bool
    dummy_mask = torch.zeros(
        1, args.dino_height, args.dino_width, device=args.device, dtype=mask_dtype
    )

    output_path = Path(args.output)
    _ensure_dir(output_path.parent)
    if args.fixed_mask:
        export_inputs = (dummy_image,)
        input_names = ["images"]
        dynamic_axes = None if args.static_batch else {
            "images": {0: "batch"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }
    else:
        export_inputs = (dummy_image, dummy_mask)
        input_names = ["images", "masks"]
        dynamic_axes = None if args.static_batch else {
            "images": {0: "batch"},
            "masks": {0: "batch"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }
    torch.onnx.export(
        wrapper,
        export_inputs,
        str(output_path),
        input_names=input_names,
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        dynamo=False,
    )

    metadata = {
        "prompt": args.text_prompt,
        "config_path": str(args.config_path),
        "checkpoint": str(args.grounded_checkpoint),
        "dino_input": [args.dino_height, args.dino_width],
        "mask_input_type": args.mask_input_type,
        "fixed_mask": args.fixed_mask,
        "static_batch": args.static_batch,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def export_sam(args):
    sam = sam_model_registry[args.sam_version](checkpoint=str(args.sam_checkpoint))
    sam.eval().to(args.device)
    onnx_model = SamMaskDecoderWrapper(sam).eval()

    image_embedding_h, image_embedding_w = sam.prompt_encoder.image_embedding_size
    dummy_image_embeddings = torch.randn(
        1,
        sam.prompt_encoder.embed_dim,
        image_embedding_h,
        image_embedding_w,
        device=args.device,
    )
    dummy_point_coords = torch.tensor([[[0.0, 0.0], [args.sam_input - 1.0, args.sam_input - 1.0]]], device=args.device)
    dummy_point_labels = torch.tensor([[2.0, 3.0]], device=args.device)
    dummy_mask_input = torch.zeros(1, 1, 256, 256, device=args.device)
    dummy_has_mask_input = torch.zeros(1, device=args.device)

    output_path = Path(args.output)
    _ensure_dir(output_path.parent)
    torch.onnx.export(
        onnx_model,
        (
            dummy_image_embeddings,
            dummy_point_coords,
            dummy_point_labels,
            dummy_mask_input,
            dummy_has_mask_input,
        ),
        str(output_path),
        input_names=[
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
        ],
        output_names=["low_res_masks", "scores"],
        dynamic_axes={
            "image_embeddings": {0: "batch"},
            "point_coords": {0: "batch", 1: "num_points"},
            "point_labels": {0: "batch", 1: "num_points"},
            "mask_input": {0: "batch"},
            "has_mask_input": {0: "batch"},
            "low_res_masks": {0: "batch"},
            "scores": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )


class SamMaskDecoderWrapper(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.mask_decoder = sam.mask_decoder
        self.model = sam
        self.img_size = sam.image_encoder.img_size

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )
        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[i].weight * (
                point_labels == i
            )
        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input.reshape(-1, 1, 1, 1) * self.model.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input.reshape(-1, 1, 1, 1)
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)
        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )
        return masks[:, 0:1, :, :], scores[:, 0:1]


class SamImageEncoderWrapper(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam

    def forward(self, x):
        return self.sam.image_encoder(x)


def export_sam_image_encoder(args):
    sam = sam_model_registry[args.sam_version](checkpoint=str(args.sam_checkpoint))
    sam.eval().to(args.device)
    wrapper = SamImageEncoderWrapper(sam).eval()
    dummy = torch.randn(1, 3, args.sam_input, args.sam_input, device=args.device)
    output_path = Path(args.output)
    _ensure_dir(output_path.parent)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes={"image": {0: "batch"}, "image_embeddings": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grounding_dino", "sam", "sam_encoder"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config_path", default=str(REPO / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--grounded_checkpoint", default=str(REPO / "groundingdino_swint_ogc.pth"))
    parser.add_argument("--bert_base_uncased_path", default="")
    parser.add_argument("--text_prompt", default="blue carton")
    parser.add_argument("--sam_version", default="vit_b")
    parser.add_argument("--sam_checkpoint", default=str(REPO / "sam_vit_b_01ec64.pth"))
    parser.add_argument("--dino_height", type=int, default=800)
    parser.add_argument("--dino_width", type=int, default=1066)
    parser.add_argument("--mask_input_type", choices=["float", "bool"], default="float")
    parser.add_argument("--fixed_mask", action="store_true")
    parser.add_argument("--static_batch", action="store_true")
    parser.add_argument("--sam_input", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "grounding_dino":
        export_grounding_dino(args)
    elif args.mode == "sam":
        export_sam(args)
    else:
        export_sam_image_encoder(args)


if __name__ == "__main__":
    main()
