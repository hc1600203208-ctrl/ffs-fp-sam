#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CAMINFO_PATH = "/home/hc/weizi/dataset/jrnew-blue/caminfo.txt"
DEFAULT_LEFT_DIR = "/home/hc/weizi/dataset/jrnew-blue/rgb"
DEFAULT_RIGHT_DIR = "/home/hc/weizi/dataset/jrnew-blue/camera2"
DEFAULT_CONDA_HOME = "~/anaconda3"
EXPECTED_CALIBRATION_VALUES = 41

WINDOW_NAME = "Stereo Rectification Alignment"
HEADER_HEIGHT = 42
ROW_GAP = 14
PANEL_GAP = 18


@dataclass(frozen=True)
class StereoImagePair:
    key: str
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class StereoCalibration:
    k_left: "np.ndarray"
    dist_left: "np.ndarray"
    k_right: "np.ndarray"
    dist_right: "np.ndarray"
    rotation: "np.ndarray"
    translation: "np.ndarray"
    baseline: float


@dataclass(frozen=True)
class StereoRectificationMaps:
    image_size: Tuple[int, int]
    left_map_x: "np.ndarray"
    left_map_y: "np.ndarray"
    right_map_x: "np.ndarray"
    right_map_y: "np.ndarray"
    r1: "np.ndarray"
    r2: "np.ndarray"
    p1: "np.ndarray"
    p2: "np.ndarray"
    q: "np.ndarray"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch visualizer for stereo epipolar rectification alignment. "
            "The calibration parsing and rectification flow mirror "
            "offline_stereo_depth_node.cpp."
        )
    )
    parser.add_argument(
        "--caminfo-path",
        default=DEFAULT_CAMINFO_PATH,
        help=f"Stereo calibration txt file. Default: {DEFAULT_CAMINFO_PATH}",
    )
    parser.add_argument(
        "--left-dir",
        default=DEFAULT_LEFT_DIR,
        help=f"Directory containing left camera images. Default: {DEFAULT_LEFT_DIR}",
    )
    parser.add_argument(
        "--right-dir",
        default=DEFAULT_RIGHT_DIR,
        help=f"Directory containing right camera images. Default: {DEFAULT_RIGHT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for visualization outputs. Default: "
            "<left-dir-parent>/rectification_vis"
        ),
    )
    parser.add_argument(
        "--match-by",
        choices=("filename", "stem"),
        default="filename",
        help=(
            "How to match left/right images. filename matches the C++ offline "
            "node behavior. Default: filename"
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this zero-based index in the matched pair list.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum number of pairs to process. 0 means all pairs. Default: 0",
    )
    parser.add_argument(
        "--line-step",
        type=int,
        default=80,
        help="Source-image pixel interval between horizontal epipolar guide lines. Default: 80",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=1,
        help="Guide line thickness before display scaling. Default: 1",
    )
    parser.add_argument(
        "--max-vis-width",
        type=int,
        default=2400,
        help=(
            "Maximum saved visualization width. 0 keeps full resolution. "
            "Default: 2400"
        ),
    )
    parser.add_argument(
        "--save-rectified",
        action="store_true",
        help="Also save rectified left/right images into output subdirectories.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show each visualization with cv2.imshow while processing.",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=0,
        help=(
            "cv2.waitKey delay when --show is set. 0 waits for a key on each "
            "pair. Default: 0"
        ),
    )
    parser.add_argument(
        "--expected-conda-env",
        default="sam",
        help="Expected conda environment name for runtime warning. Default: sam",
    )
    parser.add_argument(
        "--conda-home",
        default=DEFAULT_CONDA_HOME,
        help=f"Conda installation directory used in warning text. Default: {DEFAULT_CONDA_HOME}",
    )
    parser.add_argument(
        "--skip-conda-check",
        action="store_true",
        help="Do not warn when the current Python does not look like the expected conda env.",
    )
    return parser.parse_args()


def warn_if_wrong_conda_env(expected_env: str, conda_home: Path) -> None:
    if not expected_env:
        return

    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix_name = Path(sys.prefix).name
    running_expected_env = conda_default_env == expected_env or prefix_name == expected_env
    if running_expected_env:
        return

    conda_sh = conda_home / "etc" / "profile.d" / "conda.sh"
    print(
        (
            f"Warning: current Python does not look like conda env '{expected_env}'. "
            f"Recommended run method:\n"
            f"  source {conda_sh} && conda activate {expected_env}\n"
            f"  python {Path(__file__).resolve()} ..."
        ),
        file=sys.stderr,
    )


def import_runtime_dependencies(expected_env: str, conda_home: Path) -> None:
    global cv2, np

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        conda_sh = conda_home / "etc" / "profile.d" / "conda.sh"
        raise SystemExit(
            "Failed to import OpenCV/numpy. Run this script inside the requested "
            f"conda environment:\n"
            f"  source {conda_sh} && conda activate {expected_env}\n"
            f"  python {Path(__file__).resolve()} ..."
        ) from exc

    cv2 = cv2_module
    np = np_module


def load_stereo_calibration_from_txt(path: Path) -> StereoCalibration:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")

    values: List[float] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.split("#", 1)[0]
            for token in line.split():
                values.append(float(token))

    if len(values) < EXPECTED_CALIBRATION_VALUES:
        raise RuntimeError(
            "Calibration file has insufficient numeric values. "
            f"Expected at least {EXPECTED_CALIBRATION_VALUES}, got {len(values)}."
        )

    idx = 0
    k_left = np.array(values[idx : idx + 9], dtype=np.float64).reshape(3, 3)
    idx += 9
    baseline = float(values[idx])
    idx += 1
    dist_left = np.array(values[idx : idx + 5], dtype=np.float64).reshape(1, 5)
    idx += 5
    k_right = np.array(values[idx : idx + 9], dtype=np.float64).reshape(3, 3)
    idx += 9
    dist_right = np.array(values[idx : idx + 5], dtype=np.float64).reshape(1, 5)
    idx += 5
    rotation = np.array(values[idx : idx + 9], dtype=np.float64).reshape(3, 3)
    idx += 9
    translation = np.array(values[idx : idx + 3], dtype=np.float64).reshape(3, 1)

    if baseline <= 0.0:
        baseline = abs(float(translation[0, 0]))
    if baseline <= 0.0:
        baseline = float(np.linalg.norm(translation))
    if baseline <= 0.0:
        raise RuntimeError("Baseline parsed from calibration file is invalid.")

    return StereoCalibration(
        k_left=k_left,
        dist_left=dist_left,
        k_right=k_right,
        dist_right=dist_right,
        rotation=rotation,
        translation=translation,
        baseline=baseline,
    )


def compute_stereo_rectification_maps(
    calibration: StereoCalibration,
    image_size: Tuple[int, int],
) -> StereoRectificationMaps:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise RuntimeError("Image size for stereo rectification must be positive.")

    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        calibration.k_left,
        calibration.dist_left,
        calibration.k_right,
        calibration.dist_right,
        image_size,
        calibration.rotation,
        calibration.translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
        newImageSize=image_size,
    )

    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        calibration.k_left,
        calibration.dist_left,
        r1,
        p1,
        image_size,
        cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        calibration.k_right,
        calibration.dist_right,
        r2,
        p2,
        image_size,
        cv2.CV_32FC1,
    )

    return StereoRectificationMaps(
        image_size=image_size,
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        r1=r1,
        r2=r2,
        p1=p1,
        p2=p2,
        q=q,
    )


def collect_image_pairs(left_dir: Path, right_dir: Path, match_by: str) -> List[StereoImagePair]:
    if not left_dir.is_dir():
        raise FileNotFoundError(f"Left image directory does not exist: {left_dir}")
    if not right_dir.is_dir():
        raise FileNotFoundError(f"Right image directory does not exist: {right_dir}")

    left_images = collect_images(left_dir, match_by)
    right_images = collect_images(right_dir, match_by)
    matched_keys = sorted(set(left_images.keys()) & set(right_images.keys()))
    if not matched_keys:
        raise RuntimeError(
            f"No matched stereo image pairs found under {left_dir} and {right_dir}. "
            "If extensions differ, try --match-by stem."
        )

    return [
        StereoImagePair(key=key, left_path=left_images[key], right_path=right_images[key])
        for key in matched_keys
    ]


def collect_images(directory: Path, match_by: str) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        key = path.name if match_by == "filename" else path.stem
        if key in images:
            raise RuntimeError(
                f"Duplicate image match key '{key}' in {directory}. "
                "Use exact filename matching or remove duplicate extensions."
            )
        images[key] = path
    return images


def load_color_image(path: Path) -> "np.ndarray":
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return image


def rectify_pair(
    left_image: "np.ndarray",
    right_image: "np.ndarray",
    maps: StereoRectificationMaps,
) -> Tuple["np.ndarray", "np.ndarray"]:
    left_rectified = cv2.remap(
        left_image,
        maps.left_map_x,
        maps.left_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rectified = cv2.remap(
        right_image,
        maps.right_map_x,
        maps.right_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_rectified, right_rectified


def compute_canvas_scale(image_width: int, max_vis_width: int) -> float:
    if max_vis_width <= 0:
        return 1.0

    full_width = image_width * 2 + PANEL_GAP
    if full_width <= max_vis_width:
        return 1.0
    return max_vis_width / float(full_width)


def resize_for_canvas(image: "np.ndarray", scale: float) -> "np.ndarray":
    if abs(scale - 1.0) < 1e-6:
        return image.copy()

    height, width = image.shape[:2]
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def make_pair_panel(
    left_image: "np.ndarray",
    right_image: "np.ndarray",
    title: str,
    line_step: int,
    line_thickness: int,
) -> "np.ndarray":
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + PANEL_GAP + right_image.shape[1]
    panel = np.full((HEADER_HEIGHT + height, width, 3), 18, dtype=np.uint8)

    cv2.putText(
        panel,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    y0 = HEADER_HEIGHT
    panel[y0 : y0 + left_image.shape[0], 0 : left_image.shape[1]] = left_image
    right_x0 = left_image.shape[1] + PANEL_GAP
    panel[y0 : y0 + right_image.shape[0], right_x0 : right_x0 + right_image.shape[1]] = right_image

    draw_epipolar_guides(panel, y0, height, width, line_step, line_thickness)
    cv2.putText(
        panel,
        "left",
        (12, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "right",
        (right_x0 + 12, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def draw_epipolar_guides(
    image: "np.ndarray",
    y_offset: int,
    height: int,
    width: int,
    line_step: int,
    line_thickness: int,
) -> None:
    if line_step <= 0:
        return

    colors = [
        (0, 255, 255),
        (0, 210, 80),
        (255, 190, 60),
        (255, 100, 190),
        (80, 180, 255),
    ]
    thickness = max(1, line_thickness)
    start_y = max(1, line_step // 2)
    for line_index, local_y in enumerate(range(start_y, height, line_step)):
        color = colors[line_index % len(colors)]
        y = y_offset + local_y
        cv2.line(image, (0, y), (width - 1, y), color, thickness, cv2.LINE_AA)


def make_visualization(
    pair: StereoImagePair,
    pair_number: int,
    pair_count: int,
    left_raw: "np.ndarray",
    right_raw: "np.ndarray",
    left_rectified: "np.ndarray",
    right_rectified: "np.ndarray",
    calibration: StereoCalibration,
    maps: StereoRectificationMaps,
    line_step: int,
    line_thickness: int,
    max_vis_width: int,
) -> "np.ndarray":
    height, width = left_raw.shape[:2]
    scale = compute_canvas_scale(width, max_vis_width)
    scaled_line_step = max(1, int(round(line_step * scale))) if line_step > 0 else 0
    scaled_line_thickness = max(1, int(round(line_thickness * scale)))

    left_raw_vis = resize_for_canvas(left_raw, scale)
    right_raw_vis = resize_for_canvas(right_raw, scale)
    left_rectified_vis = resize_for_canvas(left_rectified, scale)
    right_rectified_vis = resize_for_canvas(right_rectified, scale)

    rectified_baseline = compute_rectified_baseline(maps)
    raw_title = (
        f"Raw stereo pair [{pair_number}/{pair_count}] | "
        f"{pair.left_path.name} / {pair.right_path.name} | "
        f"{width}x{height}"
    )
    rectified_title = (
        "Rectified stereo pair | "
        f"baseline={calibration.baseline:.6f}m | "
        f"rectified_baseline={rectified_baseline:.6f}m | "
        f"fx={maps.p1[0, 0]:.3f}"
    )

    raw_panel = make_pair_panel(
        left_raw_vis,
        right_raw_vis,
        raw_title,
        scaled_line_step,
        scaled_line_thickness,
    )
    rectified_panel = make_pair_panel(
        left_rectified_vis,
        right_rectified_vis,
        rectified_title,
        scaled_line_step,
        scaled_line_thickness,
    )

    canvas_width = max(raw_panel.shape[1], rectified_panel.shape[1])
    canvas_height = raw_panel.shape[0] + ROW_GAP + rectified_panel.shape[0]
    canvas = np.full((canvas_height, canvas_width, 3), 12, dtype=np.uint8)
    paste_centered(canvas, raw_panel, 0)
    paste_centered(canvas, rectified_panel, raw_panel.shape[0] + ROW_GAP)
    return canvas


def paste_centered(canvas: "np.ndarray", panel: "np.ndarray", y: int) -> None:
    x = (canvas.shape[1] - panel.shape[1]) // 2
    canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel


def compute_rectified_baseline(maps: StereoRectificationMaps) -> float:
    fx = float(maps.p2[0, 0])
    if abs(fx) <= 1e-12:
        return 0.0
    return abs(float(maps.p2[0, 3]) / fx)


def resolve_output_dir(left_dir: Path, configured_output_dir: Optional[str]) -> Path:
    if configured_output_dir:
        return Path(configured_output_dir).expanduser()
    return left_dir.parent / "rectification_vis"


def save_rectified_images(
    output_dir: Path,
    pair: StereoImagePair,
    left_rectified: "np.ndarray",
    right_rectified: "np.ndarray",
) -> None:
    left_dir = output_dir / "rectified_left"
    right_dir = output_dir / "rectified_right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    left_output = left_dir / pair.left_path.name
    right_output = right_dir / pair.right_path.name
    if not cv2.imwrite(str(left_output), left_rectified):
        raise RuntimeError(f"Failed to save rectified left image: {left_output}")
    if not cv2.imwrite(str(right_output), right_rectified):
        raise RuntimeError(f"Failed to save rectified right image: {right_output}")


def process_pair(
    pair: StereoImagePair,
    pair_number: int,
    pair_count: int,
    calibration: StereoCalibration,
    maps_by_size: Dict[Tuple[int, int], StereoRectificationMaps],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    left_raw = load_color_image(pair.left_path)
    right_raw = load_color_image(pair.right_path)
    if left_raw.shape[:2] != right_raw.shape[:2]:
        raise RuntimeError(
            "Stereo image sizes do not match: "
            f"{pair.left_path} is {left_raw.shape[1]}x{left_raw.shape[0]}, "
            f"{pair.right_path} is {right_raw.shape[1]}x{right_raw.shape[0]}"
        )

    image_size = (left_raw.shape[1], left_raw.shape[0])
    if image_size not in maps_by_size:
        maps_by_size[image_size] = compute_stereo_rectification_maps(calibration, image_size)
        print(f"Rectification maps prepared for {image_size[0]}x{image_size[1]} images.")

    maps = maps_by_size[image_size]
    left_rectified, right_rectified = rectify_pair(left_raw, right_raw, maps)
    visualization = make_visualization(
        pair=pair,
        pair_number=pair_number,
        pair_count=pair_count,
        left_raw=left_raw,
        right_raw=right_raw,
        left_rectified=left_rectified,
        right_rectified=right_rectified,
        calibration=calibration,
        maps=maps,
        line_step=args.line_step,
        line_thickness=args.line_thickness,
        max_vis_width=args.max_vis_width,
    )

    output_path = output_dir / f"{pair_number:06d}_{pair.left_path.stem}_rectification.png"
    if not cv2.imwrite(str(output_path), visualization):
        raise RuntimeError(f"Failed to save visualization: {output_path}")

    if args.save_rectified:
        save_rectified_images(output_dir, pair, left_rectified, right_rectified)

    if args.show:
        cv2.imshow(WINDOW_NAME, visualization)
        key = cv2.waitKey(max(args.wait_ms, 0)) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt

    return output_path


def slice_pairs(
    pairs: List[StereoImagePair],
    start_index: int,
    max_pairs: int,
) -> List[StereoImagePair]:
    if start_index < 0:
        raise RuntimeError("--start-index must be non-negative.")
    if max_pairs < 0:
        raise RuntimeError("--max-pairs must be non-negative.")

    selected = pairs[start_index:]
    if max_pairs > 0:
        selected = selected[:max_pairs]
    if not selected:
        raise RuntimeError("No image pairs selected for processing.")
    return selected


def main() -> None:
    args = parse_args()
    conda_home = Path(args.conda_home).expanduser()
    if not args.skip_conda_check:
        warn_if_wrong_conda_env(args.expected_conda_env, conda_home)
    import_runtime_dependencies(args.expected_conda_env, conda_home)

    caminfo_path = Path(args.caminfo_path).expanduser()
    left_dir = Path(args.left_dir).expanduser()
    right_dir = Path(args.right_dir).expanduser()
    output_dir = resolve_output_dir(left_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_stereo_calibration_from_txt(caminfo_path)
    pairs = collect_image_pairs(left_dir, right_dir, args.match_by)
    selected_pairs = slice_pairs(pairs, args.start_index, args.max_pairs)

    print(f"Calibration: {caminfo_path}")
    print(f"Left images:  {left_dir}")
    print(f"Right images: {right_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Matched pairs: {len(pairs)}, selected pairs: {len(selected_pairs)}")

    maps_by_size: Dict[Tuple[int, int], StereoRectificationMaps] = {}
    processed = 0
    failed = 0
    try:
        for offset, pair in enumerate(selected_pairs, start=1):
            pair_number = args.start_index + offset
            try:
                output_path = process_pair(
                    pair=pair,
                    pair_number=pair_number,
                    pair_count=len(pairs),
                    calibration=calibration,
                    maps_by_size=maps_by_size,
                    output_dir=output_dir,
                    args=args,
                )
            except Exception as exc:
                failed += 1
                print(f"[{pair_number}/{len(pairs)}] failed: {pair.key}: {exc}", file=sys.stderr)
                continue

            processed += 1
            print(f"[{pair_number}/{len(pairs)}] saved {output_path}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(f"Done. processed={processed}, failed={failed}, output_dir={output_dir}")
    if processed == 0 or failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
