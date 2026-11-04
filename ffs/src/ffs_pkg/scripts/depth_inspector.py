#!/usr/bin/env python3

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Depth Inspector"
ROI_WINDOW_NAME = "Depth ROI Detail"
INFO_BAR_HEIGHT = 98
PANEL_GAP = 12


@dataclass(frozen=True)
class ImageGroup:
    stem: str
    rgb_path: Path
    depth_paths: Tuple[Path, ...]


@dataclass(frozen=True)
class Panel:
    depth_index: Optional[int]
    x: int
    y: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize RGB and depth image pairs, and inspect RGB/depth values "
            "under the mouse cursor in real time."
        )
    )
    parser.add_argument("--rgb-dir", required=True, help="Directory containing RGB images.")
    parser.add_argument(
        "--depth-dir", required=True, nargs="+", metavar="DIR",
        help="One or more depth directories. The first defines the shared color ranges.",
    )
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
        "--roi-focus-span-meters",
        type=float,
        default=0.5,
        help="Initial depth span around a right-clicked target in the ROI. Default: 0.5 m",
    )
    parser.add_argument(
        "--roi-target-gap-meters",
        type=float,
        default=0.5,
        help="Largest depth gap within the automatically selected target cluster. Default: 0.5 m",
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
    args = parser.parse_args()
    if args.depth_scale <= 0:
        parser.error("--depth-scale must be greater than zero")
    if args.roi_focus_span_meters <= 0:
        parser.error("--roi-focus-span-meters must be greater than zero")
    if args.roi_target_gap_meters <= 0:
        parser.error("--roi-target-gap-meters must be greater than zero")
    return args


def collect_pairs(rgb_dir: Path, depth_dirs: Sequence[Path]) -> List[ImageGroup]:
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
    if not depth_dirs:
        raise ValueError("At least one depth directory is required")

    rgb_map = {
        path.stem: path
        for path in sorted(rgb_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }
    depth_maps = []
    for depth_dir in depth_dirs:
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")
        depth_maps.append({
            path.stem: path
            for path in sorted(depth_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        })

    matched = set(rgb_map)
    for depth_map in depth_maps:
        matched &= set(depth_map)
    matched_stems = sorted(matched)
    if not matched_stems:
        raise RuntimeError(
            "No common image stems found in the RGB directory and all depth directories."
        )
    print(f"Matched {len(matched_stems)} frames across RGB and {len(depth_dirs)} depth directories")
    return [ImageGroup(stem, rgb_map[stem], tuple(m[stem] for m in depth_maps))
            for stem in matched_stems]


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


def compute_roi_visualization_range(
    depth_roi: np.ndarray, target_gap_meters: float
) -> Tuple[float, float, int]:
    valid = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0.0)]
    if valid.size == 0:
        return 0.1, 1.0, 0

    # The target occupies most of the ROI, so its depth contains the median.
    # Sparse transition depths can bridge a foreground/background gap; stop
    # when either the depth gap or the local depth population becomes too small.
    median = float(np.median(valid))
    depth_min, depth_max = float(np.min(valid)), float(np.max(valid))
    bin_width = max(0.005, min(0.05, target_gap_meters / 10.0))
    bin_count = min(4096, max(1, math.ceil((depth_max - depth_min) / bin_width)))
    counts, edges = np.histogram(valid, bins=bin_count,
                                 range=(depth_min, depth_max + bin_width))
    center = min(int(np.searchsorted(edges, median, side="right") - 1), bin_count - 1)
    nearby_peak = int(np.max(counts[max(0, center - 2):min(bin_count, center + 3)]))
    min_population = max(1, math.ceil(nearby_peak * 0.05))
    occupied = np.flatnonzero(counts)
    center_pos = int(np.searchsorted(occupied, center))
    left_pos = right_pos = center_pos
    actual_bin_width = float(edges[1] - edges[0])
    while left_pos > 0:
        previous = int(occupied[left_pos - 1])
        current = int(occupied[left_pos])
        if ((current - previous) * actual_bin_width > target_gap_meters + actual_bin_width
                or counts[previous] < min_population):
            break
        left_pos -= 1
    while right_pos + 1 < occupied.size:
        current = int(occupied[right_pos])
        following = int(occupied[right_pos + 1])
        if ((following - current) * actual_bin_width > target_gap_meters + actual_bin_width
                or counts[following] < min_population):
            break
        right_pos += 1
    low_edge = edges[int(occupied[left_pos])]
    high_edge = edges[int(occupied[right_pos]) + 1]
    target = valid[(valid >= low_edge) & (valid < high_edge)]
    vis_min, vis_max = float(np.min(target)), float(np.max(target))
    target_count = int(target.size)
    # A nearly flat surface still needs a finite color scale.
    if vis_max - vis_min < 0.02:
        center = (vis_min + vis_max) / 2.0
        vis_min, vis_max = center - 0.01, center + 0.01
    return max(0.0, vis_min), vis_max, target_count


def colorize_roi_depth(
    depth_roi: np.ndarray, vis_min: float, vis_max: float
) -> np.ndarray:
    valid = np.isfinite(depth_roi) & (depth_roi > 0.0)
    in_range = valid & (depth_roi >= vis_min) & (depth_roi <= vis_max)
    normalized = np.zeros(depth_roi.shape, dtype=np.uint8)
    if np.any(in_range):
        values = 1.0 - (depth_roi[in_range] - vis_min) / max(vis_max - vis_min, 1e-6)
        normalized[in_range] = np.clip(np.round(values * 255.0), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    color[valid & ~in_range] = (48, 48, 48)
    color[~valid] = (0, 0, 0)
    return color


class DepthInspector:
    def __init__(self, args: argparse.Namespace, pairs: List[ImageGroup]) -> None:
        self.args = args
        self.pairs = pairs
        self.index = 0
        self.loaded_index: Optional[int] = None
        self.current_rgb: Optional[np.ndarray] = None
        self.current_depth_raws: List[np.ndarray] = []
        self.current_depth_meters: List[np.ndarray] = []
        self.current_depth_range: Tuple[float, float] = (0.1, 1.0)
        self.current_panels: List[Panel] = []
        self.base_canvas: Optional[np.ndarray] = None
        self.current_canvas: Optional[np.ndarray] = None
        self.current_roi_canvas: Optional[np.ndarray] = None
        self.current_roi_range: Optional[Tuple[float, float]] = None
        self.roi_depth_bounds: Optional[Tuple[int, int, int, int]] = None
        self.roi_range_override: Optional[Tuple[float, float]] = None
        self.roi_dirty = False
        self.roi_window_open = False
        self.mouse_position: Optional[Tuple[int, int]] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.drag_panel: Optional[Panel] = None
        depth_dirs = [path.parent for path in pairs[0].depth_paths]
        common = Path(os.path.commonpath(depth_dirs)) if len(depth_dirs) > 1 else None
        self.depth_labels = [
            f"Depth {i + 1}{' (reference)' if i == 0 else ''}: "
            f"{(path.relative_to(common) if common is not None and path != common else path.name)}"
            for i, path in enumerate(depth_dirs)
        ]

    @staticmethod
    def map_coordinate(index: int, src_size: int, dst_size: int) -> int:
        if src_size <= 1 or dst_size <= 1:
            return 0
        return int(round(index * (dst_size - 1) / (src_size - 1)))

    @staticmethod
    def map_bounds(
        bounds: Tuple[int, int, int, int],
        source_shape: Tuple[int, int],
        destination_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds
        src_h, src_w = source_shape
        dst_h, dst_w = destination_shape
        dx0 = min(dst_w - 1, max(0, math.floor(x0 * dst_w / src_w)))
        dy0 = min(dst_h - 1, max(0, math.floor(y0 * dst_h / src_h)))
        dx1 = min(dst_w, max(dx0 + 1, math.ceil(x1 * dst_w / src_w)))
        dy1 = min(dst_h, max(dy0 + 1, math.ceil(y1 * dst_h / src_h)))
        return dx0, dy0, dx1, dy1

    @staticmethod
    def put_text(canvas: np.ndarray, text: str, x: int, y: int,
                 color: Tuple[int, int, int] = (230, 230, 230), scale: float = 0.55) -> None:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def fit_text(text: str, width: int, scale: float = 0.55) -> str:
        if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= width:
            return text
        if ": " in text and "/" in text:
            title, path = text.split(": ", 1)
            text = f"{title}: .../{path.rsplit('/', 1)[-1]}"
            if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= width:
                return text
        while text and cv2.getTextSize(text + "...", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
            text = text[:-1]
        return text + "..."

    def layout_panels(
        self, images: Sequence[np.ndarray], labels: Sequence[str],
        max_width: int, max_height: int, top: int, footer: int = 0,
        subtitles: Optional[Sequence[str]] = None, zoom: bool = False,
    ) -> Tuple[np.ndarray, List[Panel]]:
        count = len(images)
        columns = min(3, count)
        rows = math.ceil(count / columns)
        cell_width = max(1, (max_width - 24 - PANEL_GAP * (columns - 1)) // columns)
        cell_height = max(1, (max_height - top - footer - 12 - PANEL_GAP * (rows - 1)) // rows)
        label_height = 26
        subtitle_height = 24 if subtitles is not None else 0
        image_height = max(1, cell_height - label_height - subtitle_height)
        canvas_width = 24 + columns * cell_width + PANEL_GAP * (columns - 1)
        canvas_height = top + rows * cell_height + PANEL_GAP * (rows - 1) + footer + 12
        canvas = np.full((canvas_height, canvas_width, 3), 18, dtype=np.uint8)
        panels = []
        for i, (image, label) in enumerate(zip(images, labels)):
            row, col = divmod(i, columns)
            cell_x = 12 + col * (cell_width + PANEL_GAP)
            cell_y = top + row * (cell_height + PANEL_GAP)
            self.put_text(canvas, self.fit_text(label, cell_width - 4), cell_x, cell_y + 19,
                          (180, 255, 180) if i == 1 else (230, 230, 230))
            height, width = image.shape[:2]
            factor = min(cell_width / width, image_height / height)
            if zoom:
                factor = min(factor, 40.0)
            else:
                factor = min(factor, 4.0)
            display_width = max(1, int(round(width * factor)))
            display_height = max(1, int(round(height * factor)))
            interpolation = cv2.INTER_NEAREST if zoom and i > 0 else (
                cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
            )
            display = cv2.resize(image, (display_width, display_height), interpolation=interpolation)
            image_x = cell_x + (cell_width - display_width) // 2
            image_y = cell_y + label_height + (image_height - display_height) // 2
            canvas[image_y:image_y + display_height, image_x:image_x + display_width] = display
            panels.append(Panel(None if i == 0 else i - 1,
                                image_x, image_y, display_width, display_height))
            if subtitles is not None:
                self.put_text(canvas, self.fit_text(subtitles[i], cell_width - 4, 0.47),
                              cell_x, cell_y + cell_height - 5, (190, 190, 190), 0.47)
        return canvas, panels

    def load_current_frame(self) -> None:
        group = self.pairs[self.index]
        self.current_rgb = load_rgb_image(group.rgb_path)
        self.current_depth_raws = [load_depth_image(path) for path in group.depth_paths]
        self.current_depth_meters = [depth_to_meters(raw, self.args.depth_scale)
                                     for raw in self.current_depth_raws]
        self.current_depth_range = compute_visualization_range(
            self.current_depth_meters[0], self.args.min_depth_meters,
            self.args.max_depth_meters,
        )
        colored = [colorize_depth(depth, *self.current_depth_range)[0]
                   for depth in self.current_depth_meters]
        self.base_canvas, self.current_panels = self.layout_panels(
            [self.current_rgb, *colored], ["RGB", *self.depth_labels],
            self.args.window_width, self.args.window_height, INFO_BAR_HEIGHT,
        )
        self.loaded_index = self.index

    def panel_at(self, x: int, y: int) -> Optional[Panel]:
        for panel in self.current_panels:
            if panel.x <= x < panel.x + panel.width and panel.y <= y < panel.y + panel.height:
                return panel
        return None

    @staticmethod
    def clamp_to_panel(x: int, y: int, panel: Panel) -> Tuple[int, int]:
        return (min(max(x, panel.x), panel.x + panel.width - 1),
                min(max(y, panel.y), panel.y + panel.height - 1))

    def panel_source_shape(self, panel: Panel) -> Tuple[int, int]:
        assert self.current_rgb is not None
        if panel.depth_index is None:
            return self.current_rgb.shape[:2]
        return self.current_depth_meters[panel.depth_index].shape

    def canvas_to_reference(self, x: int, y: int) -> Optional[Tuple[Panel, int, int, int, int]]:
        panel = self.panel_at(x, y)
        if panel is None:
            return None
        source_h, source_w = self.panel_source_shape(panel)
        source_x = self.map_coordinate(x - panel.x, panel.width, source_w)
        source_y = self.map_coordinate(y - panel.y, panel.height, source_h)
        ref_h, ref_w = self.current_depth_meters[0].shape
        ref_x = self.map_coordinate(source_x, source_w, ref_w)
        ref_y = self.map_coordinate(source_y, source_h, ref_h)
        return panel, source_x, source_y, ref_x, ref_y

    def selection_to_depth_bounds(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        first = self.canvas_to_reference(*start)
        last = self.canvas_to_reference(*end)
        if first is None or last is None or first[0] != last[0]:
            return None
        x0, x1 = sorted((first[3], last[3]))
        y0, y1 = sorted((first[4], last[4]))
        return x0, y0, x1 + 1, y1 + 1

    def focus_roi_at(self, x: int, y: int) -> None:
        if self.roi_depth_bounds is None:
            return
        position = self.canvas_to_reference(x, y)
        if position is None:
            return
        _, _, _, ref_x, ref_y = position
        x0, y0, x1, y1 = self.roi_depth_bounds
        if not (x0 <= ref_x < x1 and y0 <= ref_y < y1):
            return
        reference = self.current_depth_meters[0]
        patch = reference[max(y0, ref_y - 2):min(y1, ref_y + 3),
                          max(x0, ref_x - 2):min(x1, ref_x + 3)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return
        center = float(np.median(valid))
        half_span = self.args.roi_focus_span_meters / 2.0
        self.roi_range_override = (max(0.0, center - half_span), center + half_span)
        self.roi_dirty = True

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.mouse_position = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_panel = self.panel_at(x, y)
            if self.drag_panel is not None:
                self.drag_start = (x, y)
                self.drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_panel is not None:
            self.drag_current = self.clamp_to_panel(x, y, self.drag_panel)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_panel is not None:
            end = self.clamp_to_panel(x, y, self.drag_panel)
            if (self.drag_start is not None and abs(end[0] - self.drag_start[0]) >= 4
                    and abs(end[1] - self.drag_start[1]) >= 4):
                bounds = self.selection_to_depth_bounds(self.drag_start, end)
                if bounds is not None:
                    self.roi_depth_bounds = bounds
                    self.roi_range_override = None
                    self.roi_dirty = True
            self.drag_start = None
            self.drag_current = None
            self.drag_panel = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.focus_roi_at(x, y)

    def draw_region_overlay(self, canvas: np.ndarray) -> None:
        if self.roi_depth_bounds is not None:
            ref_shape = self.current_depth_meters[0].shape
            for panel in self.current_panels:
                source_shape = self.panel_source_shape(panel)
                x0, y0, x1, y1 = self.map_bounds(self.roi_depth_bounds, ref_shape, source_shape)
                start = (panel.x + self.map_coordinate(x0, source_shape[1], panel.width),
                         panel.y + self.map_coordinate(y0, source_shape[0], panel.height))
                end = (panel.x + self.map_coordinate(x1 - 1, source_shape[1], panel.width),
                       panel.y + self.map_coordinate(y1 - 1, source_shape[0], panel.height))
                cv2.rectangle(canvas, start, end, (255, 255, 0), 2)
        if self.drag_start is not None and self.drag_current is not None:
            cv2.rectangle(canvas, self.drag_start, self.drag_current, (0, 255, 255), 1)

    def render_roi_detail(self) -> np.ndarray:
        assert self.current_rgb is not None and self.roi_depth_bounds is not None
        reference_shape = self.current_depth_meters[0].shape
        rgb_bounds = self.map_bounds(self.roi_depth_bounds, reference_shape, self.current_rgb.shape[:2])
        rx0, ry0, rx1, ry1 = rgb_bounds
        images = [self.current_rgb[ry0:ry1, rx0:rx1]]
        crops = []
        for depth in self.current_depth_meters:
            x0, y0, x1, y1 = self.map_bounds(self.roi_depth_bounds, reference_shape, depth.shape)
            crops.append(depth[y0:y1, x0:x1])
        reference_crop = crops[0]
        if self.roi_range_override is None:
            vis_min, vis_max, _ = compute_roi_visualization_range(
                reference_crop, self.args.roi_target_gap_meters
            )
            mode = "reference target cluster"
        else:
            vis_min, vis_max = self.roi_range_override
            mode = "reference target focus"
        self.current_roi_range = (vis_min, vis_max)
        images.extend(colorize_roi_depth(crop, vis_min, vis_max) for crop in crops)
        subtitles = [f"RGB crop: {rx1 - rx0} x {ry1 - ry0} pixels"]
        for crop in crops:
            valid = np.isfinite(crop) & (crop > 0)
            colored = valid & (crop >= vis_min) & (crop <= vis_max)
            subtitles.append(f"Colored: {np.count_nonzero(colored)}/{np.count_nonzero(valid)} valid pixels")
        canvas, _ = self.layout_panels(
            images, ["RGB", *self.depth_labels],
            min(self.args.window_width, 1500), min(self.args.window_height, 1000),
            112, footer=85, subtitles=subtitles, zoom=True,
        )
        x0, y0, x1, y1 = self.roi_depth_bounds
        self.put_text(canvas, f"ROI on reference: x={x0}:{x1}, y={y0}:{y1}", 12, 25)
        self.put_text(canvas, f"Shared depth color: {vis_min:.3f} - {vis_max:.3f} m | {mode}",
                      12, 52, (180, 255, 180))
        self.put_text(canvas, "All depth panels use the reference range | Gray: outside range | Black: invalid",
                      12, 79, (200, 200, 200), 0.53)
        bar_y = canvas.shape[0] - 71
        bar_width = min(512, canvas.shape[1] - 24)
        gradient = np.linspace(255, 0, bar_width, dtype=np.uint8)[None, :]
        bar = cv2.applyColorMap(np.repeat(gradient, 18, axis=0), cv2.COLORMAP_JET)
        canvas[bar_y:bar_y + 18, 12:12 + bar_width] = bar
        self.put_text(canvas, f"near {vis_min:.3f} m", 12, bar_y + 44, scale=0.5)
        self.put_text(canvas, f"{(vis_min + vis_max) / 2:.3f} m", 12 + bar_width // 2 - 42,
                      bar_y + 44, scale=0.5)
        self.put_text(canvas, f"far {vis_max:.3f} m", 12 + bar_width - 110,
                      bar_y + 44, scale=0.5)
        return canvas

    def get_hover_info(self) -> Optional[str]:
        if self.mouse_position is None:
            return None
        position = self.canvas_to_reference(*self.mouse_position)
        if position is None:
            return None
        panel, sx, sy, rx, ry = position
        assert self.current_rgb is not None
        if panel.depth_index is None:
            rgb = self.current_rgb[sy, sx]
            rgb_values = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            label = "Reference"
            raw = self.current_depth_raws[0][ry, rx]
            meters = float(self.current_depth_meters[0][ry, rx])
            prefix = f"RGB({sx},{sy})={rgb_values} | "
            dx, dy = rx, ry
        else:
            i = panel.depth_index
            label = f"Depth {i + 1}"
            raw = self.current_depth_raws[i][sy, sx]
            meters = float(self.current_depth_meters[i][sy, sx])
            prefix = ""
            dx, dy = sx, sy
        value = f"{meters:.4f} m" if math.isfinite(meters) and meters > 0 else "invalid"
        return f"{prefix}{label}({dx},{dy}) raw={raw} -> {value}"

    def draw_overlay(self, canvas: np.ndarray) -> None:
        group = self.pairs[self.index]
        self.put_text(canvas, f"[{self.index + 1}/{len(self.pairs)}] {group.stem} | "
                      f"{len(group.depth_paths)} depth sources | first is reference", 12, 24, scale=0.6)
        self.put_text(canvas, "Drag: shared ROI | Right click: reference focus | +/-: span | "
                      "C: auto | R: clear | S: save | A/D: frames | Q: quit",
                      12, 49, (180, 220, 255), 0.52)
        vis_min, vis_max = self.current_depth_range
        info = f"Full shared range: {vis_min:.3f} - {vis_max:.3f} m"
        if self.current_roi_range is not None:
            lo, hi = self.current_roi_range
            info += f" | ROI shared range: {lo:.3f} - {hi:.3f} m"
        self.put_text(canvas, info, 12, 73, (180, 255, 180), 0.53)
        hover = self.get_hover_info()
        if hover is not None:
            self.put_text(canvas, self.fit_text(hover, canvas.shape[1] - 24, 0.52),
                          12, 93, (255, 255, 255), 0.52)

    def render_current_frame(self) -> None:
        if self.loaded_index != self.index:
            self.load_current_frame()
        assert self.base_canvas is not None
        if self.roi_depth_bounds is not None and (self.roi_dirty or self.current_roi_canvas is None):
            self.current_roi_canvas = self.render_roi_detail()
            self.roi_dirty = False
        canvas = self.base_canvas.copy()
        self.draw_region_overlay(canvas)
        self.draw_overlay(canvas)
        self.current_canvas = canvas
        cv2.imshow(WINDOW_NAME, canvas)
        if self.current_roi_canvas is not None:
            if not self.roi_window_open:
                cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(ROI_WINDOW_NAME, self.current_roi_canvas)
            self.roi_window_open = True

    def clear_roi(self) -> None:
        self.roi_depth_bounds = None
        self.roi_range_override = None
        self.current_roi_range = None
        self.current_roi_canvas = None
        self.roi_dirty = False
        self.drag_start = None
        self.drag_current = None
        self.drag_panel = None
        if self.roi_window_open:
            try:
                cv2.destroyWindow(ROI_WINDOW_NAME)
            except cv2.error:
                pass
            self.roi_window_open = False

    def step(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.pairs)
        self.mouse_position = None
        self.clear_roi()

    def adjust_roi_span(self, factor: float) -> None:
        if self.roi_depth_bounds is None or self.current_roi_range is None:
            return
        vis_min, vis_max = self.current_roi_range
        center = (vis_min + vis_max) / 2.0
        half_span = max((vis_max - vis_min) * factor / 2.0, 0.001)
        self.roi_range_override = (max(0.0, center - half_span), center + half_span)
        self.roi_dirty = True

    def save_screenshot(self) -> None:
        if self.current_canvas is None:
            return
        group = self.pairs[self.index]
        output_dir = group.depth_paths[0].parent
        screenshot_path = output_dir / f"{group.stem}_inspect.png"
        cv2.imwrite(str(screenshot_path), self.current_canvas)
        print(f"Saved screenshot to {screenshot_path}")
        if self.current_roi_canvas is not None:
            roi_path = output_dir / f"{group.stem}_roi_inspect.png"
            cv2.imwrite(str(roi_path), self.current_roi_canvas)
            print(f"Saved ROI detail to {roi_path}")

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
            elif key == ord("r"):
                self.clear_roi()
            elif key == ord("c"):
                self.roi_range_override = None
                self.roi_dirty = True
            elif key in (ord("+"), ord("=")):
                self.adjust_roi_span(0.5)
            elif key in (ord("-"), ord("_")):
                self.adjust_roi_span(2.0)
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    groups = collect_pairs(Path(args.rgb_dir), [Path(path) for path in args.depth_dir])
    DepthInspector(args, groups).run()


if __name__ == "__main__":
    main()
