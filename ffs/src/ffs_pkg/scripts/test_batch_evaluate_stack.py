#!/usr/bin/env python3

import json
from argparse import Namespace
from itertools import product
from pathlib import Path

import numpy as np

from batch_evaluate_stack import (
    FOUNDATIONPOSE_RUNNERS,
    MESH_VARIANTS,
    RATF_ABLATIONS,
    DatasetLayout,
    align_bundletrack_poses,
    build_arg_parser,
    build_bundletrack_evaluation_command,
    foundationpose_output_dir,
    parse_foundationpose_failure_starts,
    tracking_depth_inputs,
    write_bundletrack_comparison,
)


def test_foundationpose_experiment_matrix() -> None:
    root = Path("/tmp/tracking")
    args = Namespace(
        tracking_depth_subdir="depth",
        tracking_filtered_depth_subdir="depth_filter",
        include_tracking_filtered_depth=False,
    )
    layout = DatasetLayout(
        *([Path("/tmp")] * 8),
        foundationpose_output_root=Path("/tmp/fp"),
        logs_root=Path("/tmp"),
    )
    depth_inputs = tracking_depth_inputs(root, args)

    assert depth_inputs == (("raw", root / "depth"),)
    outputs = {
        foundationpose_output_dir(layout, variant, runner, depth_variant)
        for variant, runner, (depth_variant, _) in product(
            MESH_VARIANTS, FOUNDATIONPOSE_RUNNERS, depth_inputs
        )
    }
    assert len(outputs) == 4
    assert Path("/tmp/fp/main/standard") in outputs

    args.include_tracking_filtered_depth = True
    depth_inputs = tracking_depth_inputs(root, args)
    assert depth_inputs == (
        ("raw", root / "depth"),
        ("confidence_filtered", root / "depth_filter"),
    )
    outputs = {
        foundationpose_output_dir(layout, variant, runner, depth_variant)
        for variant, runner, (depth_variant, _) in product(
            MESH_VARIANTS, FOUNDATIONPOSE_RUNNERS, depth_inputs
        )
    }
    assert len(outputs) == 8
    assert Path("/tmp/fp/main/standard/confidence_filtered") in outputs


def test_ratf_ablation_matrix() -> None:
    assert RATF_ABLATIONS == (
        ("baseline", False, False, False),
        ("without_view_weighting", False, True, True),
        ("without_depth_consistency", True, False, True),
        ("without_hole_filling", True, True, False),
        ("full_ratf", True, True, True),
    )


def test_skip_laf_confidence_option() -> None:
    args = build_arg_parser().parse_args(["--run_confidence_reconstruction_comparison", "--skip_laf_confidence"])
    assert args.run_confidence_reconstruction_comparison
    assert args.skip_laf_confidence


def test_bundletrack_uses_existing_pose_evaluator() -> None:
    args = build_arg_parser().parse_args([])
    assert args.bundletrack_output_dir == ""
    assert args.bundletrack_reference_pose_dir == ""
    command = build_bundletrack_evaluation_command(
        tracking_dataset_root=Path("/tmp/tracking"),
        mesh_path=Path("/tmp/mesh.obj"),
        pred_pose_dir=Path("/tmp/poses"),
        output_dir=Path("/tmp/evaluation"),
        args=args,
        exclude_frame_ids_file=Path("/tmp/failed.txt"),
    )

    assert command[command.index("--depth_dir") + 1] == "/tmp/tracking/depth"
    assert command[command.index("--pred_pose") + 1] == "/tmp/poses"
    assert command[command.index("--exclude_frame_ids_file") + 1] == "/tmp/failed.txt"
    assert "--run_foundationpose" not in command


def test_foundationpose_failure_start_groups() -> None:
    assert parse_foundationpose_failure_starts(
        ["main:standard=0044", "gaijin/adaptive_filter=0100"]
    ) == {
        "main:standard": "0044",
        "gaijin:adaptive_filter": "0100",
    }


def test_bundletrack_pose_alignment_uses_only_first_reference(tmp_path: Path) -> None:
    pose_dir = tmp_path / "raw"
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "aligned"
    pose_dir.mkdir()
    reference_dir.mkdir()

    bundle_first = np.eye(4)
    bundle_second = np.eye(4)
    bundle_second[0, 3] = 0.1
    reference_first = np.eye(4)
    reference_first[2, 3] = 0.5
    np.savetxt(pose_dir / "0001.txt", bundle_first)
    np.savetxt(pose_dir / "0002.txt", bundle_second)
    np.savetxt(reference_dir / "0001.txt", reference_first)

    align_bundletrack_poses(pose_dir, reference_dir, output_dir)

    assert np.allclose(np.loadtxt(output_dir / "0001.txt"), reference_first)
    expected_second = reference_first.copy()
    expected_second[0, 3] = 0.1
    assert np.allclose(np.loadtxt(output_dir / "0002.txt"), expected_second)


def test_bundletrack_comparison_uses_gaijin_adaptive_filter(tmp_path: Path) -> None:
    output_root = tmp_path / "bundletrack"
    bundletrack_summary = output_root / "evaluation" / "summary.json"
    foundationpose_summary = tmp_path / "foundationpose_summary.json"
    bundletrack_summary.parent.mkdir(parents=True)
    bundletrack_summary.write_text(json.dumps({"mask_iou_mean": 0.8}), encoding="utf-8")
    foundationpose_summary.write_text(
        json.dumps(
            {
                "results": {
                    "gaijin": {
                        "adaptive_filter": {
                            "raw": {"mask_iou_mean": 0.9},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    comparison_path = write_bundletrack_comparison(
        output_root=output_root,
        tracking_dataset_root=tmp_path / "tracking",
        mesh_path=tmp_path / "mesh.obj",
        depth_dir=tmp_path / "tracking" / "depth",
        render_mode="faces",
        foundationpose_summary_path=foundationpose_summary,
        bundletrack_summary_path=bundletrack_summary,
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["bundletrack"]["mask_iou_mean"] == 0.8
    assert comparison["foundationpose"] == [{"mask_iou_mean": 0.9}]


if __name__ == "__main__":
    test_foundationpose_experiment_matrix()
    test_ratf_ablation_matrix()
    test_bundletrack_uses_existing_pose_evaluator()
