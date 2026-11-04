from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

import depth_inspector as inspector_module
from depth_inspector import (
    DepthInspector,
    collect_pairs,
    colorize_roi_depth,
    compute_roi_visualization_range,
)


def test_roi_colors_span_the_majority_target_and_ignore_far_background() -> None:
    target = np.linspace(1.0, 1.3, 700, dtype=np.float32)
    background = np.full(300, 20.0, dtype=np.float32)
    depth_roi = np.concatenate((target, background)).reshape(20, 50)

    vis_min, vis_max, target_count = compute_roi_visualization_range(depth_roi, 0.1)
    color = colorize_roi_depth(depth_roi, vis_min, vis_max).reshape(-1, 3)

    assert target_count == 700
    assert abs(vis_min - 1.0) < 1e-5
    assert abs(vis_max - 1.3) < 1e-5
    assert np.all(color[700:] == (48, 48, 48))
    assert not np.array_equal(color[0], color[699])


def test_roi_median_cluster_excludes_near_outliers_too() -> None:
    depth_roi = np.concatenate((
        np.full(5, 0.2, dtype=np.float32),
        np.linspace(1.0, 1.3, 70, dtype=np.float32),
        np.full(25, 10.0, dtype=np.float32),
    ))

    vis_min, vis_max, target_count = compute_roi_visualization_range(depth_roi, 0.1)

    assert target_count == 70
    assert abs(vis_min - 1.0) < 1e-5
    assert abs(vis_max - 1.3) < 1e-5


def test_target_gap_can_include_a_depth_discontinuity_on_the_object() -> None:
    depth_roi = np.concatenate((
        np.full(60, 1.0, dtype=np.float32),
        np.full(20, 1.18, dtype=np.float32),
        np.full(20, 10.0, dtype=np.float32),
    ))

    _, _, narrow_count = compute_roi_visualization_range(depth_roi, 0.1)
    vis_min, vis_max, wide_count = compute_roi_visualization_range(depth_roi, 0.2)

    assert narrow_count == 60
    assert wide_count == 80
    assert abs(vis_min - 1.0) < 1e-5
    assert abs(vis_max - 1.18) < 1e-5


def test_sparse_transition_does_not_connect_target_to_background() -> None:
    target = np.linspace(1.0, 1.3, 10000, dtype=np.float32)
    sparse_transition = np.arange(1.31, 4.9, 0.01, dtype=np.float32)
    background = np.full(1000, 5.0, dtype=np.float32)
    depth_roi = np.concatenate((target, sparse_transition, background))

    vis_min, vis_max, target_count = compute_roi_visualization_range(depth_roi, 0.5)

    assert target_count == target.size
    assert abs(vis_min - 1.0) < 1e-5
    assert abs(vis_max - 1.3) < 1e-5


def test_multiple_depth_directories_share_roi_and_reference_colors(
    tmp_path: Path, monkeypatch,
) -> None:
    rgb_dir = tmp_path / "rgb"
    depth_dirs = [tmp_path / f"method_{i}" for i in range(4)]
    rgb_dir.mkdir()
    for directory in depth_dirs:
        directory.mkdir()

    cv2.imwrite(str(rgb_dir / "frame.png"), np.full((120, 160, 3), 100, np.uint8))
    reference = np.full((120, 160), 20000, np.uint16)
    reference[40:80, 50:90] = np.linspace(1000, 1300, 40, dtype=np.uint16)[None, :]
    cv2.imwrite(str(depth_dirs[0] / "frame.png"), reference)
    for directory in depth_dirs[1:]:
        other = np.full((60, 80), 20000, np.uint16)
        other[20:40, 25:45] = 2000
        cv2.imwrite(str(directory / "frame.png"), other)
    cv2.imwrite(str(rgb_dir / "unmatched.png"), np.zeros((2, 2, 3), np.uint8))

    groups = collect_pairs(rgb_dir, depth_dirs)
    assert len(groups) == 1
    assert groups[0].stem == "frame"
    assert groups[0].depth_paths == tuple(directory / "frame.png" for directory in depth_dirs)

    args = Namespace(depth_scale=1000.0, min_depth_meters=None, max_depth_meters=None,
                     roi_focus_span_meters=0.5, roi_target_gap_meters=0.5,
                     window_width=1200, window_height=900)
    viewer = DepthInspector(args, groups)
    monkeypatch.setattr(cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    color_ranges = []
    original_colorize = inspector_module.colorize_roi_depth

    def capture_range(depth_roi, vis_min, vis_max):
        color_ranges.append((vis_min, vis_max))
        return original_colorize(depth_roi, vis_min, vis_max)

    monkeypatch.setattr(inspector_module, "colorize_roi_depth", capture_range)
    viewer.render_current_frame()
    assert len(viewer.current_panels) == 5
    assert viewer.current_depth_range == (1.0, 20.0)

    # Select on the second depth panel, which has half the reference resolution.
    panel = viewer.current_panels[2]
    start = (panel.x + viewer.map_coordinate(25, 80, panel.width),
             panel.y + viewer.map_coordinate(20, 60, panel.height))
    end = (panel.x + viewer.map_coordinate(44, 80, panel.width),
           panel.y + viewer.map_coordinate(39, 60, panel.height))
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, *start, 0, None)
    viewer.on_mouse(cv2.EVENT_LBUTTONUP, *end, 0, None)
    viewer.render_current_frame()

    assert viewer.roi_depth_bounds is not None
    x0, y0, x1, y1 = viewer.roi_depth_bounds
    assert 49 <= x0 <= 51 and 89 <= x1 <= 91
    assert 39 <= y0 <= 41 and 79 <= y1 <= 81
    assert len(color_ranges) == 4
    assert len(set(color_ranges)) == 1
    vis_min, vis_max = viewer.current_roi_range
    assert 0.99 <= vis_min <= 1.01 and 1.29 <= vis_max <= 1.31
    assert viewer.current_roi_canvas is not None

    viewer.focus_roi_at(*start)
    viewer.render_current_frame()
    assert abs(viewer.current_roi_range[1] - viewer.current_roi_range[0] - 0.5) < 1e-5
    viewer.save_screenshot()
    assert (depth_dirs[0] / "frame_inspect.png").exists()
    assert (depth_dirs[0] / "frame_roi_inspect.png").exists()
