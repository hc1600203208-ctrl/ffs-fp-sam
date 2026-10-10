#!/usr/bin/env python3
"""Compare raw and confidence-filtered depth against one reconstruction mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "points": int(values.size),
        "mean_m": float(values.mean()),
        "p25_m": float(np.quantile(values, 0.25)),
        "median_m": float(np.median(values)),
        "p75_m": float(np.quantile(values, 0.75)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
        "over_10mm_fraction": float(np.mean(values > 0.01)),
        "over_100mm_fraction": float(np.mean(values > 0.1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/home/hc/weizi/dataset/jrnew-blue"))
    parser.add_argument("--evaluate_utils", type=Path, default=Path("/home/hc/weizi/BundleSDF_confidence_main/evaluate"))
    parser.add_argument("--points_per_frame", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.points_per_frame <= 0:
        parser.error("--points_per_frame must be positive")

    sys.path.insert(0, str(args.evaluate_utils.resolve()))
    from geometry_utils import MeshDistanceQuery, backproject_depth, camera_points_to_object
    from io_utils import load_depth, load_intrinsics, load_mask, load_pose
    from render_utils import combine_geometry, load_mesh_parts

    root = args.dataset.resolve()
    benchmark = root / "evaluate/depth_benchmark"
    raw_dir = benchmark / "raw_depth"
    filtered_dir = benchmark / "confidence_full_threshold_sweep/threshold_0_35"
    reference = root / "evaluate/bundlesdf_main/confidence_full_threshold_sweep/threshold_0_35"
    output = root / "evaluate/reconstruction_depth_error_distribution"
    raw_files = {p.stem: p for p in raw_dir.glob("*.png")}
    filtered_files = {p.stem: p for p in filtered_dir.glob("*.png")}
    pose_files = {p.stem: p for p in (reference / "ob_in_cam").glob("*.txt")}
    mask_files = {p.stem: p for p in (root / "masks").glob("*.png")}
    ids = sorted(raw_files.keys() & filtered_files.keys() & pose_files.keys() & mask_files.keys())
    if not ids:
        raise RuntimeError("No frames shared by raw depth, filtered depth, masks and reference poses")
    if len(ids) != len(raw_files) or len(ids) != len(filtered_files):
        raise RuntimeError("Depth frame sets do not match the reference pose/mask frame sets")

    intrinsics = load_intrinsics(root / "cam_K.txt")
    mesh = combine_geometry(load_mesh_parts(reference / "textured_mesh.obj"))
    query = MeshDistanceQuery(mesh, method="sampled", sample_points=200_000, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    raw_errors, filtered_errors, removed_errors = [], [], []
    valid_raw_pixels = valid_filtered_pixels = 0
    for index, frame_id in enumerate(ids, 1):
        raw = load_depth(raw_files[frame_id], 1000.0)
        filtered = load_depth(filtered_files[frame_id], 1000.0)
        mask = load_mask(mask_files[frame_id])
        if raw.shape != filtered.shape or raw.shape != mask.shape:
            raise ValueError(f"Frame {frame_id}: depth/mask shape mismatch")
        raw_valid = mask & (raw > 0)
        filtered_valid = mask & (filtered > 0)
        if np.any(filtered_valid & ~raw_valid) or not np.array_equal(raw[filtered_valid], filtered[filtered_valid]):
            raise ValueError(f"Frame {frame_id}: filtered depth is not an unchanged subset of raw depth")
        valid_raw_pixels += int(raw_valid.sum())
        valid_filtered_pixels += int(filtered_valid.sum())
        candidates = np.flatnonzero(raw_valid)
        sampled = rng.choice(candidates, size=min(args.points_per_frame, len(candidates)), replace=False)
        sampled_mask = np.zeros(raw.shape, dtype=bool)
        sampled_mask.flat[sampled] = True
        pose = load_pose(pose_files[frame_id])
        raw_points = backproject_depth(raw, sampled_mask, intrinsics)
        # backproject_depth traverses row-major order, so sort the sampled indices too.
        retained = filtered.flat[np.sort(sampled)] > 0
        raw_distance = query.distances(camera_points_to_object(raw_points, pose))
        raw_errors.append(raw_distance)
        removed_errors.append(raw_distance[~retained])
        if np.any(retained):
            filtered_points = backproject_depth(filtered, sampled_mask & filtered_valid, intrinsics)
            filtered_errors.append(query.distances(camera_points_to_object(filtered_points, pose)))
        if index % 25 == 0 or index == len(ids):
            print(f"Processed {index}/{len(ids)} frames", flush=True)

    raw_error = np.concatenate(raw_errors)
    filtered_error = np.concatenate(filtered_errors)
    removed_error = np.concatenate(removed_errors)
    summary = {
        "reference_mesh": str(reference / "textured_mesh.obj"),
        "reference_pose_directory": str(reference / "ob_in_cam"),
        "raw_depth_directory": str(raw_dir),
        "filtered_depth_directory": str(filtered_dir),
        "distance_method": "nearest of 200000 seeded mesh-surface samples",
        "frames": len(ids),
        "sample_limit_per_frame": args.points_per_frame,
        "seed": args.seed,
        "valid_raw_mask_pixels": valid_raw_pixels,
        "valid_filtered_mask_pixels": valid_filtered_pixels,
        "retained_pixel_fraction": valid_filtered_pixels / valid_raw_pixels,
        "raw": describe(raw_error),
        "filtered": describe(filtered_error),
        "removed": describe(removed_error),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "point_errors_m.npz", raw=raw_error, filtered=filtered_error, removed=removed_error)

    blue, red = "#2877b9", "#cb4a48"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    bins = np.linspace(0, 0.1, 101)
    for values, label, color in ((raw_error, "Raw Depth", blue), (filtered_error, "Filtered Depth", red)):
        axes[0].hist(values, bins=bins, weights=np.full(len(values), 1 / (len(values) * (bins[1] - bins[0]))),
                     alpha=0.5, label=label, color=color)
    axes[0].set(xlabel="Error (m)", ylabel="Density (log scale)", title="Point-to-mesh error histogram", xlim=(0, 0.1))
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[0].text(0.98, 0.96, f">0.1 m: Raw {summary['raw']['over_100mm_fraction']:.1%}\nFiltered {summary['filtered']['over_100mm_fraction']:.1%}",
                 ha="right", va="top", transform=axes[0].transAxes, fontsize=9)

    box = axes[1].boxplot([raw_error, filtered_error], tick_labels=["Raw Depth", "Filtered Depth"],
                          showfliers=True, widths=0.5, patch_artist=True,
                          flierprops={"marker": ".", "markersize": 1, "alpha": 0.08, "rasterized": True})
    for patch, color in zip(box["boxes"], (blue, red)):
        patch.set(facecolor=color, alpha=0.5)
    axes[1].set(xlabel="Depth group", ylabel="Error (m, log scale)", title="Median, IQR, whiskers and fliers")
    axes[1].set_yscale("log")

    for values, label, color in ((raw_error, "Raw Depth", blue), (filtered_error, "Filtered Depth", red)):
        sorted_values = np.sort(values)
        axes[2].plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color=color, label=label, linewidth=2)
    axes[2].set(xlabel="Error threshold (m)", ylabel="Cumulative probability", title="Error CDF", xlim=(0, 0.1), ylim=(0, 1))
    axes[2].legend(frameon=False)
    zoom = axes[2].inset_axes([0.43, 0.2, 0.53, 0.45])
    for values, color in ((raw_error, blue), (filtered_error, red)):
        sorted_values = np.sort(values)
        zoom.plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=1.5)
    zoom.set(xlim=(0, 0.01), ylim=(0.85, 1), title="0-10 mm detail")
    zoom.tick_params(labelsize=8)
    zoom.grid(alpha=0.2)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Raw vs confidence-filtered depth | {len(ids)} frames | retained pixels {summary['retained_pixel_fraction']:.1%}")
    fig.savefig(output / "comparison.png", dpi=200)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
