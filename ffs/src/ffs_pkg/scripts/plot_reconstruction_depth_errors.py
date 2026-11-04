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


def plot_comparison(raw_error: np.ndarray, filtered_error: np.ndarray, summary: dict, output: Path) -> None:
    """Redraw cached measurements without changing the point samples or reference."""
    blue, red = "#2877b9", "#cb4a48"
    groups = ((raw_error, "Raw Depth", blue), (filtered_error, "Filtered Depth", red))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.19, top=0.82, wspace=0.27)
    bins = np.linspace(0, 0.1, 101)
    for values, label, color in groups:
        axes[0].hist(values, bins=bins, weights=np.full(len(values), 1 / (len(values) * np.diff(bins)[0])),
                     alpha=0.45, label=label, color=color)
    axes[0].set(xlabel="Point-to-mesh error (m)", ylabel="Probability density (log scale)",
                title="(a) Long-tail error distribution", xlim=(0, 0.1), yscale="log")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(0.97, 0.80,
                 f"Outside displayed range (>0.1 m)\nRaw: {np.mean(raw_error > 0.1):.2%}\n"
                 f"Filtered: {np.mean(filtered_error > 0.1):.2%}",
                 ha="right", va="top", transform=axes[0].transAxes, fontsize=10)

    box_ax = axes[1]
    box = box_ax.boxplot([raw_error * 1000, filtered_error * 1000],
                         labels=["Raw Depth", "Filtered Depth"], showfliers=False,
                         widths=0.30, patch_artist=True,
                         medianprops={"color": "#202020", "linewidth": 2.5})
    for patch, (_, _, color) in zip(box["boxes"], groups):
        patch.set(facecolor=color, edgecolor=color, alpha=0.45)
    box_ax.set(xlabel="Depth group", ylabel="Point-to-mesh error (mm)",
               title="(b) Quartiles, median and outliers", xlim=(0.13, 2.87), ylim=(0, 6))
    for position, (values, _, color) in enumerate(groups, 1):
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75]) * 1000
        fliers = values[(values * 1000 > q3 + 1.5 * (q3 - q1)) & (values <= 0.006)] * 1000
        rng = np.random.default_rng(position)
        shown = rng.choice(fliers, size=min(220, len(fliers)), replace=False)
        box_ax.scatter(position + rng.uniform(-0.12, 0.12, len(shown)), shown,
                       s=5, color=color, alpha=0.30, linewidths=0, rasterized=True)
        side = -1 if position == 1 else 1
        edge = position + side * 0.15
        label_x = position + side * 0.27
        for value, label in ((q1, "Q1"), (median, "Median"), (q3, "Q3")):
            box_ax.plot([edge, label_x], [value, value], color=color, linewidth=1.2)
            box_ax.text(label_x + side * 0.02, value, f"{label} {value:.3f}",
                        ha="right" if side < 0 else "left", va="center", fontsize=9)
        bracket_x = position - side * 0.26
        box_ax.annotate("", xy=(bracket_x, q1), xytext=(bracket_x, q3),
                        arrowprops={"arrowstyle": "|-|", "color": color, "linewidth": 1.5})
        box_ax.text(bracket_x - side * 0.07, (q1 + q3) / 2,
                    f"IQR\n{q3 - q1:.3f} mm", ha="center", va="center", fontsize=9,
                    color=color)

    for values, label, color in groups:
        sorted_values = np.sort(values)
        axes[2].plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color=color,
                     label=label, linewidth=2.2)
    axes[2].set(xlabel="Error threshold (m)", ylabel="Cumulative probability",
                title="(c) Fraction below an error threshold", xlim=(0, 0.01), ylim=(0, 1.02))
    axes[2].legend(frameon=False, loc="lower right")
    threshold = 0.005
    axes[2].axvline(threshold, color="#777777", linestyle=":", linewidth=1)
    for values, _, color in groups:
        axes[2].plot(threshold, np.mean(values <= threshold), "o", color=color, markersize=5)
    axes[2].text(0.96, 0.63,
                 f"Error \u2264 5 mm\nRaw: {np.mean(raw_error <= threshold):.2%}\n"
                 f"Filtered: {np.mean(filtered_error <= threshold):.2%}",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Raw vs confidence-filtered depth: central spread and long-tail errors", y=0.97, fontsize=17)
    fig.text(0.5, 0.91, f"{summary['frames']} frames | retained object pixels: {summary['retained_pixel_fraction']:.1%} | "
             "same reference mesh, poses and sampled points", ha="center", fontsize=11)
    fig.text(0.05, 0.085,
             "Box = Q1\u2013Q3 (middle 50%); thick line = median; IQR = Q3 \u2212 Q1. "
             "Whiskers end at the most extreme observations within Q1 \u2212 1.5\u00d7IQR and Q3 + 1.5\u00d7IQR.", fontsize=10)
    fig.text(0.05, 0.045,
             "Dots = sampled fliers; boxplot zooms to 0–6 mm. Beyond 6 mm: "
             f"Raw {np.mean(raw_error > 0.006):.2%}, Filtered {np.mean(filtered_error > 0.006):.2%}. "
             "Histogram/CDF use all errors, normalized per group.", fontsize=10)
    fig.savefig(output / "comparison_annotated.png", dpi=240)
    fig.savefig(output / "comparison_annotated.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/home/hc/weizi/dataset/jrnew-blue"))
    parser.add_argument("--evaluate_utils", type=Path, default=Path("/home/hc/weizi/BundleSDF_confidence_main/evaluate"))
    parser.add_argument("--points_per_frame", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_only", action="store_true", help="Redraw the saved point_errors_m.npz without recomputing distances.")
    args = parser.parse_args()
    if args.points_per_frame <= 0:
        parser.error("--points_per_frame must be positive")

    output = args.dataset.resolve() / "evaluate/reconstruction_depth_error_distribution"
    if args.plot_only:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        with np.load(output / "point_errors_m.npz") as cached:
            plot_comparison(cached["raw"], cached["filtered"], summary, output)
        print(f"Redrawn cached measurements in {output}")
        return

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

    plot_comparison(raw_error, filtered_error, summary, output)
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
