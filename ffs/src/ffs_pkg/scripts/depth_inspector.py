#!/usr/bin/env python3

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Depth Inspector"
INFO_BAR_HEIGHT = 80
PANEL_GAP = 12


@dataclass(frozen=True)
class ImagePair:
    stem: str
    rgb_path: Path
    depth_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize RGB and depth image pairs, and inspect RGB/depth values "
            "under the mouse cursor in real time."
        )
    )
    parser.add_argument("--rgb-dir", required=True, help="Directory containing RGB images.")
    parser.add_argument("--depth-dir", required=True, help="Directory containing depth images.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale used to convert uint16 depth PNG values to meters. Default: 1000.0",
    )
    parser.add_argument(
        "--min-depth-meters",
        type=float,
        default=None,
        help="Lower bound for depth visualization. Default: inferred from valid pixels in each frame.",
    )
    parser.add_argument(
        "--max-depth-meters",
        type=float,
        default=None,
        help="Upper bound for depth visualization. Default: inferred from valid pixels in each frame.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1800,
        help="Maximum display window width in pixels. Default: 1800",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1000,
        help="Maximum display window height in pixels. Default: 1000",
    )
    return parser.parse_args()


def collect_pairs(rgb_dir: Path, depth_dir: Path) -> List[ImagePair]:
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")

    rgb_map = {
        path.stem: path
        for path in sorted(rgb_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }
    depth_map = {
        path.stem: path
        for path in sorted(depth_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }

    matched_stems = sorted(set(rgb_map.keys()) & set(depth_map.keys()))
    if not matched_stems:
        raise RuntimeError(
            f"No matched RGB/depth image pairs found under {rgb_dir} and {depth_dir}."
        )

    return [ImagePair(stem, rgb_map[stem], depth_map[stem]) for stem in matched_stems]


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load RGB image: {path}")
    return image


def load_depth_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load depth image: {path}")
    if image.ndim != 2:
        raise RuntimeError(f"Depth image must be single-channel: {path}")
    return image


def depth_to_meters(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / depth_scale
    if depth.dtype == np.uint8:
        return depth.astype(np.float32) / depth_scale
    if depth.dtype in (np.float32, np.float64):
        return depth.astype(np.float32)
    raise RuntimeError(f"Unsupported depth dtype: {depth.dtype}")


def compute_visualization_range(
    depth_meters: np.ndarray,
    min_depth_meters: Optional[float],
    max_depth_meters: Optional[float],
) -> Tuple[float, float]:
    valid = depth_meters[np.isfinite(depth_meters) & (depth_meters > 0.0)]
    if valid.size == 0:
        return 0.1, 1.0

    if min_depth_meters is None:
        min_depth_meters = float(np.min(valid))
    if max_depth_meters is None:
        max_depth_meters = float(np.max(valid))

    if max_depth_meters <= min_depth_meters:
        max_depth_meters = min_depth_meters + 1e-3

    return min_depth_meters, max_depth_meters


def colorize_depth(
    depth_meters: np.ndarray,
    min_depth_meters: Optional[float],
    max_depth_meters: Optional[float],
) -> Tuple[np.ndarray, Tuple[float, float]]:
    vis_min, vis_max = compute_visualization_range(
        depth_meters,
        min_depth_meters,
        max_depth_meters,
    )
    valid_mask = np.isfinite(depth_meters) & (depth_meters >= vis_min) & (depth_meters <= vis_max)
    normalized = np.zeros(depth_meters.shape, dtype=np.uint8)

    if np.any(valid_mask):
        depth_range = max(vis_max - vis_min, 1e-6)
        normalized_valid = 1.0 - ((depth_meters[valid_mask] - vis_min) / depth_range)
        normalized[valid_mask] = np.clip(np.round(normalized_valid * 255.0), 0, 255).astype(np.uint8)

    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    color[~valid_mask] = (0, 0, 0)
    return color, (vis_min, vis_max)


def resize_for_display(
    rgb_bgr: np.ndarray,
    depth_vis_bgr: np.ndarray,
    window_width: int,
    window_height: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rgb_h, rgb_w = rgb_bgr.shape[:2]
    depth_h, depth_w = depth_vis_bgr.shape[:2]
    combined_width = rgb_w + depth_w + PANEL_GAP
    combined_height = max(rgb_h, depth_h)

    available_width = max(window_width, 1)
    available_height = max(window_height - INFO_BAR_HEIGHT, 1)
    scale = min(1.0, available_width / combined_width, available_height / combined_height)

    if scale < 1.0:
        rgb_bgr = cv2.resize(
            rgb_bgr,
            (max(1, int(round(rgb_w * scale))), max(1, int(round(rgb_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        depth_vis_bgr = cv2.resize(
            depth_vis_bgr,
            (max(1, int(round(depth_w * scale))), max(1, int(round(depth_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    return rgb_bgr, depth_vis_bgr, scale


class DepthInspector:
    def __init__(self, args: argparse.Namespace, pairs: List[ImagePair]) -> None:
        self.args = args
        self.pairs = pairs
        self.index = 0
        self.mouse_position: Optional[Tuple[int, int]] = None
        self.current_canvas: Optional[np.ndarray] = None
        self.current_rgb: Optional[np.ndarray] = None
        self.current_depth_raw: Optional[np.ndarray] = None
        self.current_depth_meters: Optional[np.ndarray] = None
        self.current_depth_vis: Optional[np.ndarray] = None
        self.current_depth_range: Tuple[float, float] = (0.1, 1.0)
        self.current_scale: float = 1.0
        self.current_rgb_display_shape: Tuple[int, int] = (0, 0)
        self.current_depth_display_shape: Tuple[int, int] = (0, 0)

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.mouse_position = (x, y)

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            self.render_current_frame()
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord("d"), ord("l"), 83):
                self.step(1)
            elif key in (ord("a"), ord("h"), 81):
                self.step(-1)
            elif key == ord("s"):
                self.save_screenshot()

        cv2.destroyAllWindows()

    def step(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.pairs)
        self.mouse_position = None

    def save_screenshot(self) -> None:
        if self.current_canvas is None:
            return
        screenshot_path = self.pairs[self.index].depth_path.parent / (
            self.pairs[self.index].stem + "_inspect.png"
        )
        cv2.imwrite(str(screenshot_path), self.current_canvas)
        print(f"Saved screenshot to {screenshot_path}")

    def render_current_frame(self) -> None:
        pair = self.pairs[self.index]
        self.current_rgb = load_rgb_image(pair.rgb_path)
        self.current_depth_raw = load_depth_image(pair.depth_path)
        self.current_depth_meters = depth_to_meters(self.current_depth_raw, self.args.depth_scale)
        self.current_depth_vis, self.current_depth_range = colorize_depth(
            self.current_depth_meters,
            self.args.min_depth_meters,
            self.args.max_depth_meters,
        )

        rgb_display, depth_display, self.current_scale = resize_for_display(
            self.current_rgb,
            self.current_depth_vis,
            self.args.window_width,
            self.args.window_height,
        )

        self.current_rgb_display_shape = rgb_display.shape[:2]
        self.current_depth_display_shape = depth_display.shape[:2]
        canvas_height = INFO_BAR_HEIGHT + max(rgb_display.shape[0], depth_display.shape[0])
        canvas_width = rgb_display.shape[1] + PANEL_GAP + depth_display.shape[1]
        canvas = np.full((canvas_height, canvas_width, 3), 18, dtype=np.uint8)

        canvas[INFO_BAR_HEIGHT : INFO_BAR_HEIGHT + rgb_display.shape[0], 0 : rgb_display.shape[1]] = rgb_display
        depth_x0 = rgb_display.shape[1] + PANEL_GAP
        canvas[
            INFO_BAR_HEIGHT : INFO_BAR_HEIGHT + depth_display.shape[0],
            depth_x0 : depth_x0 + depth_display.shape[1],
        ] = depth_display

        self.draw_overlay(canvas, pair)
        self.current_canvas = canvas
        cv2.imshow(WINDOW_NAME, canvas)

    def draw_overlay(self, canvas: np.ndarray, pair: ImagePair) -> None:
        vis_min, vis_max = self.current_depth_range
        header = (
            f"[{self.index + 1}/{len(self.pairs)}] {pair.stem} | "
            f"RGB: {pair.rgb_path.name} | Depth: {pair.depth_path.name}"
        )
        help_text = "Keys: A/Left prev, D/Right next, S save screenshot, Q/Esc quit"
        info_text = (
            f"Depth vis range: {vis_min:.3f}m - {vis_max:.3f}m | "
            f"Hover on either panel to inspect pixel values"
        )

        cv2.putText(canvas, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, help_text, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, info_text, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)

        hover_info = self.get_hover_info()
        if hover_info is not None:
            cv2.rectangle(canvas, (0, INFO_BAR_HEIGHT - 24), (canvas.shape[1], INFO_BAR_HEIGHT), (50, 50, 50), -1)
            cv2.putText(
                canvas,
                hover_info,
                (12, INFO_BAR_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def get_hover_info(self) -> Optional[str]:
        if self.mouse_position is None:
            return None
        if self.current_rgb is None or self.current_depth_meters is None or self.current_depth_raw is None:
            return None

        x, y = self.mouse_position
        if y < INFO_BAR_HEIGHT:
            return None

        local_y = y - INFO_BAR_HEIGHT
        rgb_h, rgb_w = self.current_rgb_display_shape
        depth_h, depth_w = self.current_depth_display_shape
        depth_x0 = rgb_w + PANEL_GAP

        if 0 <= x < rgb_w and 0 <= local_y < rgb_h:
            rgb_x = self.display_to_source_index(x, rgb_w, self.current_rgb.shape[1])
            rgb_y = self.display_to_source_index(local_y, rgb_h, self.current_rgb.shape[0])
            depth_x = self.map_coordinate(
                rgb_x,
                self.current_rgb.shape[1],
                self.current_depth_meters.shape[1],
            )
            depth_y = self.map_coordinate(
                rgb_y,
                self.current_rgb.shape[0],
                self.current_depth_meters.shape[0],
            )
        elif depth_x0 <= x < depth_x0 + depth_w and 0 <= local_y < depth_h:
            depth_local_x = x - depth_x0
            depth_x = self.display_to_source_index(
                depth_local_x,
                depth_w,
                self.current_depth_meters.shape[1],
            )
            depth_y = self.display_to_source_index(
                local_y,
                depth_h,
                self.current_depth_meters.shape[0],
            )
            rgb_x = self.map_coordinate(
                depth_x,
                self.current_depth_meters.shape[1],
                self.current_rgb.shape[1],
            )
            rgb_y = self.map_coordinate(
                depth_y,
                self.current_depth_meters.shape[0],
                self.current_rgb.shape[0],
            )
        else:
            return None

        rgb_bgr = self.current_rgb[rgb_y, rgb_x]
        rgb_values = (int(rgb_bgr[2]), int(rgb_bgr[1]), int(rgb_bgr[0]))

        depth_raw_value = self.current_depth_raw[depth_y, depth_x]
        depth_meters_value = float(self.current_depth_meters[depth_y, depth_x])
        depth_valid = math.isfinite(depth_meters_value) and depth_meters_value > 0.0
        depth_text = f"{depth_meters_value:.4f} m" if depth_valid else "invalid"

        return (
            f"RGB(x={rgb_x}, y={rgb_y}) = {rgb_values} | "
            f"Depth(x={depth_x}, y={depth_y}) raw={depth_raw_value} -> {depth_text}"
        )

    @staticmethod
    def display_to_source_index(display_index: int, display_size: int, source_size: int) -> int:
        if display_size <= 1 or source_size <= 1:
            return 0
        ratio = display_index / max(display_size - 1, 1)
        return int(round(ratio * (source_size - 1)))

    @staticmethod
    def map_coordinate(index: int, src_size: int, dst_size: int) -> int:
        if src_size <= 1 or dst_size <= 1:
            return 0
        ratio = index / max(src_size - 1, 1)
        return int(round(ratio * (dst_size - 1)))


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(Path(args.rgb_dir), Path(args.depth_dir))
    inspector = DepthInspector(args, pairs)
    inspector.run()


if __name__ == "__main__":
    main()
