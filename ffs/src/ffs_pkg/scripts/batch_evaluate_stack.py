#!/usr/bin/env python3
"""Batch evaluation orchestrator for the FFS + BundleSDF + FoundationPose stack.

The script expects each reconstruction sequence to have this layout:

  dataset/
    rgb/
    camera2/
    masks/
    cam_K.txt
    caminfo.txt or cam_info.txt

FoundationPose tracking is evaluated on a separate RGBD sequence for the same
object. Pass it with --tracking_dataset_root or --tracking_dataset_map. Each
tracking sequence must contain:

  tracking_dataset/
    rgb/
    depth/
    depth_filter/  # optional; enabled with --include_tracking_filtered_depth
    masks/
    cam_K.txt

For every dataset it can:
  1. run the Fast Foundation Stereo offline depth benchmark;
  2. switch BundleSDF to the baseline branch and reconstruct/evaluate every
     depth benchmark group;
  3. pick the best depth group from the reconstruction summary;
  4. switch BundleSDF to the improved branch and reconstruct/evaluate only the
     best depth group;
  5. run/evaluate FoundationPose standard and adaptive-filter runners with both
     baseline and improved meshes, using raw tracking depth by default and
     optionally confidence-filtered tracking depth.
  6. optionally run/evaluate BundleTrack on the raw tracking sequence as a
     comparison group, scored with the improved mesh and the same no-GT metrics.

Use --bundle_shorter_side to keep the BundleSDF main and gaijin branches on the
same RGB/depth/mask resize policy before comparing their meshes.

Outputs are grouped under <dataset>/evaluate by default, including depth
benchmark groups, BundleSDF reconstruction artifacts, FoundationPose scenes,
metrics, and pipeline logs. Long-running child process output is also mirrored
into log files so interrupted batches are easy to resume and inspect.

Example:

  python3 batch_evaluate_stack.py /home/hc/weizi/dataset/jrnew-blue \
    --tracking_dataset_root /path/to/jrnew-blue-rgbd-tracking
"""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASET = "/home/hc/weizi/dataset/jrnew-blue"
DEFAULT_BUNDLESDF_ROOT = "/home/hc/weizi/BundleSDF"
DEFAULT_FOUNDATIONPOSE_ROOT = "/home/hc/weizi/FoundationPoseROS2"
DEFAULT_CONDA_PREFIX = "~/anaconda3"
DEFAULT_ENGINE_FILE_PATH = (
    '["/home/hc/model/ffs/23-36-37/feature_runner_fp16_5060.engine", '
    '"/home/hc/model/ffs/23-36-37/post_runner_fp16_5060.engine"]'
)

RECONSTRUCTION_MAX_METRICS = (
    "psnr",
    "ssim",
    "mask_iou",
    "coverage_5mm",
    "coverage_1cm",
)
RECONSTRUCTION_MIN_METRICS = (
    "lpips",
    "depth_rmse",
    "obs_to_mesh_mean_dist",
)
RECONSTRUCTION_METRIC_DIRECTIONS = {
    **{name: "max" for name in RECONSTRUCTION_MAX_METRICS},
    **{name: "min" for name in RECONSTRUCTION_MIN_METRICS},
}

FOUNDATIONPOSE_RUNNERS = ("standard", "adaptive_filter")
MESH_VARIANTS = ("main", "gaijin")
RATF_ABLATIONS = (
    ("baseline", False, False, False),
    ("without_view_weighting", False, True, True),
    ("without_depth_consistency", True, False, True),
    ("without_hole_filling", True, True, False),
    ("full_ratf", True, True, True),
)


def parse_foundationpose_failure_starts(values: Sequence[str]) -> Dict[str, str]:
    valid_groups = {
        f"{variant}:{runner}"
        for variant in MESH_VARIANTS
        for runner in FOUNDATIONPOSE_RUNNERS
    }
    parsed: Dict[str, str] = {}
    for value in values:
        group, separator, frame_id = value.partition("=")
        group = group.replace("/", ":")
        if separator != "=" or group not in valid_groups or not frame_id:
            raise ValueError(
                "--fp_failure_start_frame must be GROUP=FRAME, where GROUP is one of: "
                + ", ".join(sorted(valid_groups))
            )
        parsed[group] = frame_id
    return parsed


@dataclass
class DatasetResult:
    dataset: str
    output_dir: str
    tracking_dataset: str = ""
    depth_benchmark_status: str = "pending"
    main_reconstruction_status: str = "pending"
    best_method: str = ""
    gaijin_reconstruction_status: str = "pending"
    foundationpose_status: str = "pending"
    bundletrack_status: str = "pending"
    elapsed_seconds: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: Path
    evaluate_root: Path
    depth_benchmark_root: Path
    main_bundle_root: Path
    main_reconstruction_eval_root: Path
    gaijin_bundle_root: Path
    gaijin_reconstruction_eval_root: Path
    foundationpose_scene_root: Path
    foundationpose_output_root: Path
    logs_root: Path


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def resolve_under(base: Path, value: str | Path) -> Path:
    path = expand_path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: path.name,
    )


def image_stems(directory: Path) -> List[str]:
    return [path.stem for path in image_files(directory)]


def image_file_map(directory: Path) -> Dict[str, Path]:
    return {path.stem: path for path in image_files(directory)}


def natural_sort_key(text: str) -> Tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    )


def discover_depth_method_dirs(benchmark_root: Path) -> List[Path]:
    if not benchmark_root.is_dir():
        return []
    methods = []
    for directory in sorted([benchmark_root, *benchmark_root.rglob("*")]):
        if directory.is_dir() and image_files(directory):
            methods.append(directory)
    return methods


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def conda_executable(conda_prefix: str) -> str:
    candidate = Path(conda_prefix).expanduser() / "bin" / "conda"
    if candidate.exists():
        return str(candidate)
    return "conda"


def default_ros_setup_files() -> List[Path]:
    files: List[Path] = []
    ros_roots = sorted(Path("/opt/ros").glob("*/setup.bash")) if Path("/opt/ros").is_dir() else []
    if ros_roots:
        files.append(ros_roots[-1])

    workspace_setup = Path(__file__).resolve().parents[3] / "install" / "setup.bash"
    if workspace_setup.exists():
        files.append(workspace_setup)
    return files


def sourced_command(command: Sequence[str], setup_files: Sequence[Path]) -> List[str]:
    parts = []
    for setup_file in setup_files:
        parts.append(f"source {shlex.quote(str(setup_file))}")
    parts.append(command_text(command))
    return ["bash", "-lc", " && ".join(parts)]


def run_logged(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    label: str,
    stall_timeout_seconds: float,
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] command: {command_text(command)}", flush=True)
    if dry_run:
        print(f"[{label}] dry run; command not executed", flush=True)
        return 0

    environment = os.environ.copy()
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment["PYTHONUNBUFFERED"] = "1"

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log_file.write(command_text(command) + "\n")
        log_file.flush()

        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            start_new_session=True,
        )
        assert process.stdout is not None
        stdout_fd = process.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        last_report = time.monotonic()
        last_output = last_report
        last_line = ""
        line_buffer = ""
        stalled = False

        def update_last_line(text: str) -> None:
            nonlocal last_line, line_buffer
            line_buffer += text
            lines = line_buffer.splitlines()
            if lines:
                last_line = lines[-1].strip()
            newline_index = max(line_buffer.rfind("\n"), line_buffer.rfind("\r"))
            if newline_index >= 0:
                line_buffer = line_buffer[newline_index + 1 :]
            elif len(line_buffer) > 1000:
                line_buffer = line_buffer[-1000:]

        def drain_stdout(now: float) -> None:
            nonlocal last_output
            while True:
                try:
                    chunk = os.read(stdout_fd, 65536)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if not text:
                    continue
                log_file.write(text)
                log_file.flush()
                update_last_line(text)
                last_output = now

        def terminate_process_group(message: str) -> None:
            if process.poll() is not None:
                return
            print(f"[{label}] {message}", flush=True)
            log_file.write(f"\n[PIPELINE] {message}\n")
            log_file.flush()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

        try:
            while process.poll() is None:
                now = time.monotonic()
                events = selector.select(timeout=5.0)
                if events:
                    drain_stdout(now)

                if last_line and now - last_report >= 60.0:
                    print(f"[{label}] still running; latest log: {last_line[:220]}", flush=True)
                    last_report = now

                if stall_timeout_seconds > 0 and now - last_output >= stall_timeout_seconds:
                    stalled = True
                    message = (
                        f"No child-process log output for {stall_timeout_seconds / 60.0:.1f} "
                        "minutes; terminating the process group"
                    )
                    terminate_process_group(message)
                    break
        except BaseException:
            terminate_process_group("Parent process interrupted; terminating the child process group")
            raise
        finally:
            drain_stdout(time.monotonic())
            final_text = decoder.decode(b"", final=True)
            if final_text:
                log_file.write(final_text)
                log_file.flush()
                update_last_line(final_text)
            selector.close()
        return_code = 124 if stalled else process.wait()

    print(f"[{label}] exit code: {return_code}; log: {log_path}", flush=True)
    return return_code


def run_capture(command: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


class GitBranchGuard:
    def __init__(
        self,
        repo_root: Path,
        restore_original: bool,
        allow_dirty: bool,
        dry_run: bool,
    ) -> None:
        self.repo_root = repo_root
        self.restore_original = restore_original
        self.allow_dirty = allow_dirty
        self.dry_run = dry_run
        self.original_branch = self._current_branch()

    def _current_branch(self) -> str:
        try:
            branch = run_capture(
                ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                self.repo_root,
            )
            if branch:
                return branch
        except subprocess.CalledProcessError:
            pass
        return run_capture(["git", "rev-parse", "--short", "HEAD"], self.repo_root)

    def _dirty_status(self) -> str:
        return run_capture(["git", "status", "--porcelain"], self.repo_root)

    def checkout(self, branch: str) -> None:
        current = self._current_branch()
        if current == branch:
            print(f"[BundleSDF] already on branch {branch}", flush=True)
            return

        dirty = self._dirty_status()
        if dirty and not self.allow_dirty:
            raise RuntimeError(
                "BundleSDF worktree is dirty; refusing to switch branches. "
                "Commit/stash changes or pass --allow_dirty_bundlesdf."
            )

        print(f"[BundleSDF] checkout {branch} (from {current})", flush=True)
        if self.dry_run:
            return
        subprocess.run(["git", "checkout", branch], cwd=str(self.repo_root), check=True)

    def restore(self) -> None:
        if not self.restore_original:
            return
        current = self._current_branch()
        if current == self.original_branch:
            return
        dirty = self._dirty_status()
        if dirty and not self.allow_dirty:
            print(
                "[BundleSDF] not restoring original branch because the worktree is dirty",
                flush=True,
            )
            return
        print(f"[BundleSDF] restoring branch {self.original_branch}", flush=True)
        if not self.dry_run:
            subprocess.run(
                ["git", "checkout", self.original_branch],
                cwd=str(self.repo_root),
                check=True,
            )


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def metric_direction(metric: str, requested: str) -> str:
    if requested in {"min", "max"}:
        return requested
    if metric in RECONSTRUCTION_METRIC_DIRECTIONS:
        return RECONSTRUCTION_METRIC_DIRECTIONS[metric]
    raise ValueError(f"Cannot infer direction for metric {metric!r}; pass --best_metric_direction")


def rank_rows(rows: Sequence[Mapping[str, str]], metric: str, direction: str) -> Dict[int, float]:
    indexed_values = []
    for index, row in enumerate(rows):
        value = parse_float(row.get(metric))
        if math.isfinite(value):
            indexed_values.append((index, value))

    reverse = direction == "max"
    indexed_values.sort(key=lambda item: item[1], reverse=reverse)
    ranks: Dict[int, float] = {}
    rank = 1
    while rank <= len(indexed_values):
        start = rank - 1
        value = indexed_values[start][1]
        end = start
        while end + 1 < len(indexed_values) and indexed_values[end + 1][1] == value:
            end += 1
        averaged_rank = (start + 1 + end + 1) / 2.0
        for position in range(start, end + 1):
            ranks[indexed_values[position][0]] = averaged_rank
        rank = end + 2
    return ranks


def select_best_method(
    summary_csv: Path,
    best_metric: str,
    best_metric_direction: str,
) -> Tuple[str, Dict[str, Any]]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"BundleSDF summary not found: {summary_csv}")

    rows = [row for row in read_csv_dicts(summary_csv) if row.get("method")]
    if not rows:
        raise RuntimeError(f"No method rows found in {summary_csv}")

    if best_metric != "composite":
        direction = metric_direction(best_metric, best_metric_direction)
        scored = []
        for row in rows:
            value = parse_float(row.get(best_metric))
            if math.isfinite(value):
                scored.append((value, row))
        if not scored:
            raise RuntimeError(f"No finite values for metric {best_metric!r} in {summary_csv}")
        scored.sort(key=lambda item: item[0], reverse=direction == "max")
        best_row = scored[0][1]
        details = {
            "selection_mode": "single_metric",
            "metric": best_metric,
            "direction": direction,
            "value": parse_float(best_row.get(best_metric)),
            "summary_csv": str(summary_csv),
            "row": best_row,
        }
        return best_row["method"], details

    metrics: List[Tuple[str, str]] = []
    for metric in RECONSTRUCTION_MAX_METRICS:
        if metric in rows[0]:
            metrics.append((metric, "max"))
    for metric in RECONSTRUCTION_MIN_METRICS:
        if metric in rows[0]:
            metrics.append((metric, "min"))
    if not metrics:
        raise RuntimeError(f"No known reconstruction quality metrics found in {summary_csv}")

    scores = {index: [] for index in range(len(rows))}
    for metric, direction in metrics:
        ranks = rank_rows(rows, metric, direction)
        for index, rank in ranks.items():
            scores[index].append(rank)

    scored_rows = []
    for index, row in enumerate(rows):
        if scores[index]:
            average_rank = sum(scores[index]) / len(scores[index])
            scored_rows.append((average_rank, row.get("method", ""), index, row))
    if not scored_rows:
        raise RuntimeError(f"No finite metric values found in {summary_csv}")

    scored_rows.sort(key=lambda item: (item[0], item[1]))
    average_rank, _, best_index, best_row = scored_rows[0]
    details = {
        "selection_mode": "composite_rank",
        "metrics": [{"name": metric, "direction": direction} for metric, direction in metrics],
        "average_rank": average_rank,
        "summary_csv": str(summary_csv),
        "row": best_row,
        "all_scores": [
            {
                "method": row.get("method", ""),
                "average_rank": score,
                "ranked_metrics": len(scores[index]),
            }
            for score, _, index, row in scored_rows
        ],
        "best_row_index": best_index,
    }
    return best_row["method"], details


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def write_dataset_status(path: Path, result: DatasetResult) -> None:
    write_json(path, asdict(result))


def write_batch_status(path: Path, results: Sequence[DatasetResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DatasetResult.__dataclass_fields__.keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    temporary.replace(path)


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = target_path.resolve()
    if link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link_path}")
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def reset_generated_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def populate_frame_symlink_dir(
    link_dir: Path,
    files_by_stem: Mapping[str, Path],
    frame_ids: Sequence[str],
) -> None:
    reset_generated_path(link_dir)
    link_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        source = files_by_stem.get(frame_id)
        if source is not None:
            (link_dir / source.name).symlink_to(source.resolve())


def prepare_foundationpose_scene(
    scene_dir: Path,
    rgbd_root: Path,
    depth_dir: Path,
    rgb_subdir: str,
    mask_subdir: str,
    intrinsics_path: Path,
) -> Path:
    scene_dir.mkdir(parents=True, exist_ok=True)
    rgb_files = image_file_map(rgbd_root / rgb_subdir)
    depth_files = image_file_map(depth_dir)
    mask_files = image_file_map(rgbd_root / mask_subdir)

    rgb_depth_ids = sorted(set(rgb_files) & set(depth_files), key=natural_sort_key)
    if not rgb_depth_ids:
        raise RuntimeError(
            f"No overlapping RGB/depth frame stems for FoundationPose scene: "
            f"rgb={rgbd_root / rgb_subdir}, depth={depth_dir}"
        )

    initial_mask_ids = set(rgb_depth_ids) & set(mask_files)
    if not initial_mask_ids:
        raise RuntimeError(
            f"No mask frame overlaps RGB/depth frames for FoundationPose scene: "
            f"mask={rgbd_root / mask_subdir}"
        )

    start_frame_id = sorted(initial_mask_ids, key=natural_sort_key)[0]
    start_index = rgb_depth_ids.index(start_frame_id)
    frame_ids = rgb_depth_ids[start_index:]
    excluded_rgb = len(set(rgb_files) - set(frame_ids))
    excluded_depth = len(set(depth_files) - set(frame_ids))
    if excluded_rgb or excluded_depth:
        print(
            f"[FoundationPose] prepared aligned scene {scene_dir.name}: "
            f"{len(frame_ids)} RGB/depth frame(s), excluded "
            f"{excluded_rgb} RGB and {excluded_depth} depth frame(s)",
            flush=True,
        )

    populate_frame_symlink_dir(scene_dir / "rgb", rgb_files, frame_ids)
    populate_frame_symlink_dir(scene_dir / "depth", depth_files, frame_ids)
    populate_frame_symlink_dir(scene_dir / "masks", mask_files, frame_ids)
    ensure_symlink(scene_dir / "cam_K.txt", intrinsics_path)
    return scene_dir


def method_file_under(root: Path, method: str, suffix: str) -> Path:
    return root / f"{method}{suffix}"


def method_dir_under(root: Path, method: str) -> Path:
    return root / Path(method)


def mesh_path_for(layout: DatasetLayout, variant: str, method: str) -> Path:
    if variant == "main":
        return method_dir_under(layout.main_bundle_root, method) / "textured_mesh.obj"
    if variant == "gaijin":
        return method_dir_under(layout.gaijin_bundle_root, method) / "textured_mesh.obj"
    raise ValueError(f"Unknown mesh variant: {variant}")


def filtered_depth_path_for(layout: DatasetLayout, variant: str, method: str) -> Path:
    if variant == "main":
        return method_dir_under(layout.main_bundle_root, method) / "depth_filtered"
    if variant == "gaijin":
        return method_dir_under(layout.gaijin_bundle_root, method) / "depth_filtered"
    raise ValueError(f"Unknown mesh variant: {variant}")


def selected_depth_dir(layout: DatasetLayout, method: str) -> Path:
    return method_dir_under(layout.depth_benchmark_root, method)


def depth_dir_for_foundationpose(layout: DatasetLayout, method: str, variant: str, source: str) -> Path:
    if source == "selected_depth":
        return selected_depth_dir(layout, method)
    if source == "main_filtered_depth":
        return filtered_depth_path_for(layout, "main", method)
    if source == "variant_filtered_depth":
        return filtered_depth_path_for(layout, variant, method)
    raise ValueError(f"Unsupported foundationpose depth source: {source}")


def foundationpose_debug_dir(base_output: Path, runner: str) -> Path:
    if runner == "adaptive_filter":
        return base_output / "foundationpose_adaptive_filter_debug"
    return base_output / "foundationpose_debug"


def foundationpose_summary_path(base_output: Path, runner: str) -> Path:
    return foundationpose_debug_dir(base_output, runner) / "summary.json"


def foundationpose_output_dir(
    layout: DatasetLayout,
    variant: str,
    runner: str,
    depth_variant: str,
) -> Path:
    output_dir = layout.foundationpose_output_root / variant / runner
    if depth_variant == "confidence_filtered":
        output_dir /= depth_variant
    return output_dir


def tracking_depth_inputs(
    tracking_dataset_root: Path,
    args: argparse.Namespace,
) -> Tuple[Tuple[str, Path], ...]:
    depth_inputs = (("raw", tracking_dataset_root / args.tracking_depth_subdir),)
    if args.include_tracking_filtered_depth:
        depth_inputs += (
            (
                "confidence_filtered",
                tracking_dataset_root / args.tracking_filtered_depth_subdir,
            ),
        )
    return depth_inputs


def depth_benchmark_complete(
    dataset_root: Path,
    benchmark_root: Path,
    left_subdir: str,
    min_depth_groups: int,
) -> bool:
    expected = set(image_stems(dataset_root / left_subdir))
    if not expected:
        return False
    method_dirs = discover_depth_method_dirs(benchmark_root)
    complete_dirs = [
        directory
        for directory in method_dirs
        if set(image_stems(directory)) == expected
    ]
    return len(complete_dirs) >= min_depth_groups


def bundlesdf_summary_complete(summary_csv: Path) -> bool:
    if not summary_csv.is_file():
        return False
    try:
        rows = read_csv_dicts(summary_csv)
    except Exception:
        return False
    method_rows = [row for row in rows if row.get("method")]
    if not method_rows:
        return False
    return any(
        any(math.isfinite(parse_float(row.get(metric))) for metric in RECONSTRUCTION_METRIC_DIRECTIONS)
        for row in method_rows
    )


def foundationpose_complete(
    output_dir: Path,
    runner: str,
    failure_start_frame: str = "",
) -> bool:
    summary_path = foundationpose_summary_path(output_dir, runner)
    per_frame_path = summary_path.parent / "per_frame_metrics.csv"
    if not summary_path.is_file() or not per_frame_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("failure_start_frame", "") == failure_start_frame


def validate_dataset(dataset_root: Path, args: argparse.Namespace) -> Path:
    missing: List[str] = []
    for relative in (args.left_subdir, args.right_subdir, args.mask_subdir):
        if not (dataset_root / relative).is_dir():
            missing.append(relative + "/")
    if not (dataset_root / "cam_K.txt").is_file():
        missing.append("cam_K.txt")

    caminfo_candidates: List[Path] = []
    if args.caminfo_name:
        caminfo_candidates.append(dataset_root / args.caminfo_name)
    else:
        caminfo_candidates.extend([dataset_root / "caminfo.txt", dataset_root / "cam_info.txt"])
    caminfo_path = next((path for path in caminfo_candidates if path.is_file()), None)
    if caminfo_path is None:
        missing.append(args.caminfo_name or "caminfo.txt|cam_info.txt")

    if missing:
        raise FileNotFoundError(
            f"{dataset_root} is missing required dataset entries: {', '.join(missing)}"
        )

    left_ids = set(image_stems(dataset_root / args.left_subdir))
    right_ids = set(image_stems(dataset_root / args.right_subdir))
    mask_ids = set(image_stems(dataset_root / args.mask_subdir))
    if not left_ids:
        raise RuntimeError(f"No left images found in {dataset_root / args.left_subdir}")
    if not right_ids:
        raise RuntimeError(f"No right images found in {dataset_root / args.right_subdir}")
    if not mask_ids:
        raise RuntimeError(f"No masks found in {dataset_root / args.mask_subdir}")

    if left_ids != right_ids:
        missing_right = sorted(left_ids - right_ids)[:5]
        extra_right = sorted(right_ids - left_ids)[:5]
        raise RuntimeError(
            f"Left/right image stems differ in {dataset_root}; "
            f"missing_right={missing_right}, extra_right={extra_right}"
        )
    if not (left_ids & mask_ids):
        raise RuntimeError(f"RGB and mask stems do not overlap in {dataset_root}")

    assert caminfo_path is not None
    return caminfo_path


def looks_like_tracking_dataset(path: Path, args: argparse.Namespace) -> bool:
    return (
        (path / args.tracking_left_subdir).is_dir()
        and (path / args.tracking_depth_subdir).is_dir()
        and (path / args.tracking_mask_subdir).is_dir()
        and (path / args.tracking_intrinsics_name).is_file()
    )


def validate_tracking_dataset(tracking_root: Path, args: argparse.Namespace) -> Path:
    missing: List[str] = []
    required_subdirs = [
        args.tracking_left_subdir,
        args.tracking_depth_subdir,
        args.tracking_mask_subdir,
    ]
    if args.include_tracking_filtered_depth:
        required_subdirs.append(args.tracking_filtered_depth_subdir)
    for relative in required_subdirs:
        if not (tracking_root / relative).is_dir():
            missing.append(relative + "/")

    intrinsics_path = tracking_root / args.tracking_intrinsics_name
    if not intrinsics_path.is_file():
        missing.append(args.tracking_intrinsics_name)

    if missing:
        raise FileNotFoundError(
            f"{tracking_root} is missing required tracking RGBD entries: "
            f"{', '.join(missing)}"
        )

    rgb_ids = set(image_stems(tracking_root / args.tracking_left_subdir))
    depth_ids = set(image_stems(tracking_root / args.tracking_depth_subdir))
    mask_ids = set(image_stems(tracking_root / args.tracking_mask_subdir))
    if not rgb_ids:
        raise RuntimeError(f"No tracking RGB images found in {tracking_root / args.tracking_left_subdir}")
    if not depth_ids:
        raise RuntimeError(f"No tracking depth images found in {tracking_root / args.tracking_depth_subdir}")
    if not mask_ids:
        raise RuntimeError(f"No tracking masks found in {tracking_root / args.tracking_mask_subdir}")
    if not (rgb_ids & depth_ids & mask_ids):
        raise RuntimeError(
            f"Tracking RGB/depth/mask frame stems do not overlap in {tracking_root}"
        )
    if args.include_tracking_filtered_depth:
        filtered_depth_ids = set(image_stems(tracking_root / args.tracking_filtered_depth_subdir))
        if not filtered_depth_ids:
            raise RuntimeError(
                "No confidence-filtered tracking depth images found in "
                f"{tracking_root / args.tracking_filtered_depth_subdir}"
            )
        if not (rgb_ids & filtered_depth_ids & mask_ids):
            raise RuntimeError(
                "Tracking RGB/confidence-filtered-depth/mask frame stems do not overlap in "
                f"{tracking_root}"
            )

    return intrinsics_path


def load_tracking_dataset_map(path: Optional[str]) -> Dict[str, Path]:
    if not path:
        return {}

    map_path = expand_path(path).resolve()
    mapping: Dict[str, Path] = {}

    def key_aliases(raw_key: str) -> List[str]:
        aliases = [raw_key]
        key_path = Path(raw_key).expanduser()
        looks_like_path = key_path.is_absolute() or "/" in raw_key or "\\" in raw_key
        if looks_like_path:
            if not key_path.is_absolute():
                key_path = map_path.parent / key_path
            resolved = key_path.resolve()
            aliases.extend([str(resolved), resolved.as_posix(), resolved.name])
        else:
            aliases.append(key_path.name)
        return aliases

    for line_number, line in enumerate(map_path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue

        if "," in text:
            key, value = [part.strip() for part in text.split(",", 1)]
        else:
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid --tracking_dataset_map line {line_number}: {line!r}. "
                    "Expected 'dataset,tracking_dataset' or two whitespace-separated fields."
                )
            key, value = parts

        if key.lower() in {"dataset", "dataset_root", "sequence"} and value.lower() in {
            "tracking",
            "tracking_dataset",
            "tracking_dataset_root",
        }:
            continue

        tracking_path = expand_path(value)
        if not tracking_path.is_absolute():
            tracking_path = map_path.parent / tracking_path
        for alias in key_aliases(key):
            mapping[alias] = tracking_path.resolve()
    return mapping


def resolve_tracking_dataset(dataset_root: Path, args: argparse.Namespace) -> Path:
    resolved_dataset_root = dataset_root.expanduser().resolve()
    keys = {
        str(dataset_root),
        dataset_root.as_posix(),
        str(resolved_dataset_root),
        resolved_dataset_root.as_posix(),
        resolved_dataset_root.name,
    }
    for key in keys:
        if key in args.tracking_dataset_map_entries:
            return args.tracking_dataset_map_entries[key]

    candidates: List[Path] = []
    if args.tracking_dataset_subdir:
        candidates.append(resolve_under(dataset_root, args.tracking_dataset_subdir))

    if args.tracking_dataset_root:
        tracking_root = expand_path(args.tracking_dataset_root).resolve()
        if looks_like_tracking_dataset(tracking_root, args):
            candidates.append(tracking_root)
        suffixes = [
            args.tracking_dataset_suffix,
            "",
            "_tracking",
            "-tracking",
            "_rgbd",
            "-rgbd",
        ]
        for suffix in suffixes:
            if suffix is None:
                continue
            candidates.append(tracking_root / f"{dataset_root.name}{suffix}")

    candidates.extend(
        [
            dataset_root / "tracking",
            dataset_root / "tracking_rgbd",
            dataset_root / "foundationpose_eval",
            dataset_root / "foundationpose_rgbd",
        ]
    )
    if args.allow_reconstruction_dataset_for_tracking:
        candidates.append(dataset_root)

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if looks_like_tracking_dataset(candidate, args):
            return candidate

    raise FileNotFoundError(
        "No FoundationPose tracking RGBD dataset was found for "
        f"{dataset_root}. Pass --tracking_dataset_root, --tracking_dataset_subdir, "
        "or --tracking_dataset_map. Expected rgb/depth/masks/cam_K.txt by default."
    )


def build_layout(dataset_root: Path, args: argparse.Namespace) -> DatasetLayout:
    evaluate_root = resolve_under(dataset_root, args.evaluate_dir)
    if args.depth_benchmark_dir:
        depth_benchmark_root = resolve_under(dataset_root, args.depth_benchmark_dir)
    else:
        depth_benchmark_root = evaluate_root / "depth_benchmark"
    return DatasetLayout(
        dataset_root=dataset_root,
        evaluate_root=evaluate_root,
        depth_benchmark_root=depth_benchmark_root,
        main_bundle_root=evaluate_root / args.main_bundlesdf_dir,
        main_reconstruction_eval_root=evaluate_root / args.main_reconstruction_eval_dir,
        gaijin_bundle_root=evaluate_root / args.gaijin_bundlesdf_dir,
        gaijin_reconstruction_eval_root=evaluate_root / args.gaijin_reconstruction_eval_dir,
        foundationpose_scene_root=evaluate_root / "foundationpose_scenes",
        foundationpose_output_root=evaluate_root / "foundationpose",
        logs_root=evaluate_root / "logs" / "pipeline",
    )


def build_depth_command(
    dataset_root: Path,
    caminfo_path: Path,
    layout: DatasetLayout,
    args: argparse.Namespace,
) -> List[str]:
    launch_args = {
        "engine_file_path": args.engine_file_path,
        "model_type": args.stereo_model_type,
        "dataset_root": str(dataset_root),
        "left_subdir": args.left_subdir,
        "right_subdir": args.right_subdir,
        "caminfo_path": str(caminfo_path),
        "output_root": str(layout.depth_benchmark_root),
        "input_image_width": str(args.input_image_width),
        "input_image_height": str(args.input_image_height),
        "model_input_width": str(args.model_input_width),
        "model_input_height": str(args.model_input_height),
        "min_depth_meters": str(args.min_depth_meters),
        "max_depth_meters": str(args.max_depth_meters),
        "depth_scale": str(args.depth_scale),
        "save_input_resolution": str(args.save_input_resolution).lower(),
        "max_pairs": str(args.max_pairs),
        "median_kernel_size": str(args.median_kernel_size),
        "bilateral_d": str(args.bilateral_d),
        "bilateral_sigma_color": str(args.bilateral_sigma_color),
        "bilateral_sigma_space": str(args.bilateral_sigma_space),
        "use_temporal_confidence": str(args.use_temporal_confidence).lower(),
        "conf_threshold": str(args.conf_threshold),
        "threshold_sweep": args.threshold_sweep,
    }

    command = [
        "ros2",
        "launch",
        args.depth_launch_package,
        args.depth_launch_file,
        *[f"{key}:={value}" for key, value in launch_args.items()],
        *args.depth_launch_arg,
    ]
    return sourced_command(command, args.ros_setup_files)


def build_bundlesdf_command(
    dataset_root: Path,
    layout: DatasetLayout,
    output_root: Path,
    evaluation_root: Path,
    methods: Optional[Sequence[str]],
    args: argparse.Namespace,
    ratf_modules: Optional[Tuple[bool, bool, bool]] = None,
) -> List[str]:
    command = [
        conda_executable(args.conda_prefix),
        "run",
        "--no-capture-output",
        "-n",
        args.bundlesdf_conda_env,
        "python",
        str(args.bundlesdf_root / "batch_depth_benchmark.py"),
        "--dataset_root",
        str(dataset_root),
        "--repo_root",
        str(args.bundlesdf_root),
        "--depth_benchmark_dir",
        str(layout.depth_benchmark_root),
        "--bundlesdf_root",
        str(output_root),
        "--evaluation_root",
        str(evaluation_root),
        "--device",
        args.bundle_device,
        "--lpips_net",
        args.bundle_lpips_net,
        "--mesh_sample_points",
        str(args.bundle_mesh_sample_points),
        "--distance_method",
        args.bundle_distance_method,
        "--depth_scale",
        str(args.depth_scale),
        "--use_segmenter",
        str(args.bundle_use_segmenter),
        "--use_gui",
        str(args.bundle_use_gui),
        "--debug_level",
        str(args.bundle_debug_level),
        "--stride",
        str(args.bundle_stride),
        "--bundlesdf_shorter_side",
        str(args.bundle_shorter_side),
        "--stall_timeout_minutes",
        str(args.bundle_stall_timeout_minutes),
        "--max_retries",
        str(args.bundle_max_retries),
    ]
    if ratf_modules is not None:
        view_weighting, depth_consistency, hole_filling = ratf_modules
        command.extend(
            [
                "--texture_view_weighting",
                str(int(view_weighting)),
                "--texture_depth_filter",
                str(int(depth_consistency)),
                "--texture_hole_filling",
                str(int(hole_filling)),
            ]
        )
    if args.overwrite:
        command.append("--overwrite")
    if args.stop_on_error:
        command.append("--stop_on_error")
    if methods:
        command.extend(["--methods", *methods])
    return command


def build_foundationpose_command(
    scene_dir: Path,
    mesh_path: Path,
    output_dir: Path,
    runner: str,
    args: argparse.Namespace,
    failure_start_frame: str = "",
) -> List[str]:
    command = [
        conda_executable(args.conda_prefix),
        "run",
        "--no-capture-output",
        "-n",
        args.foundationpose_conda_env,
        "python",
        str(args.foundationpose_root / "evaluate_pose_tracking_no_gt.py"),
        "--rgb_dir",
        str(scene_dir / "rgb"),
        "--depth_dir",
        str(scene_dir / "depth"),
        "--mask_dir",
        str(scene_dir / "masks"),
        "--mesh_path",
        str(mesh_path),
        "--intrinsics",
        str(scene_dir / "cam_K.txt"),
        "--output_dir",
        str(output_dir),
        "--run_foundationpose",
        "--foundationpose_runner",
        runner,
        "--conda_prefix",
        args.conda_prefix,
        "--conda_env",
        args.foundationpose_conda_env,
        "--depth_scale",
        str(args.depth_scale),
        "--num_points",
        str(args.fp_num_points),
        "--mask_threshold",
        str(args.fp_mask_threshold),
        "--depth_min",
        str(args.fp_depth_min),
        "--depth_max",
        str(args.fp_depth_max),
        "--render_mode",
        args.fp_render_mode,
        "--render_max_faces",
        str(args.fp_render_max_faces),
        "--render_point_radius",
        str(args.fp_render_point_radius),
        "--max_observed_points",
        str(args.fp_max_observed_points),
        "--min_points",
        str(args.fp_min_points),
        "--icp_max_iterations",
        str(args.fp_icp_max_iterations),
        "--icp_tolerance",
        str(args.fp_icp_tolerance),
        "--icp_max_correspondence_dist",
        str(args.fp_icp_max_correspondence_dist),
        "--est_refine_iter",
        str(args.fp_est_refine_iter),
        "--track_refine_iter",
        str(args.fp_track_refine_iter),
        "--shorter_side",
        str(args.fp_shorter_side),
        "--debug",
        str(args.fp_debug),
        "--fps",
        str(args.fp_fps),
        "--q_pos",
        str(args.fp_q_pos),
        "--q_rot",
        str(args.fp_q_rot),
        "--q_vel",
        str(args.fp_q_vel),
        "--q_omega",
        str(args.fp_q_omega),
        "--sigma_pos_obs",
        str(args.fp_sigma_pos_obs),
        "--sigma_rot_obs",
        str(args.fp_sigma_rot_obs),
        "--sigma_pos_motion",
        str(args.fp_sigma_pos_motion),
        "--sigma_rot_motion",
        str(args.fp_sigma_rot_motion),
        "--min_quality",
        str(args.fp_min_quality),
        "--gate_threshold",
        str(args.fp_gate_threshold),
        "--reject_cov_increase",
        str(args.fp_reject_cov_increase),
        "--progress_interval",
        str(args.fp_progress_interval),
        "--max_frames",
        str(args.fp_max_frames),
    ]
    if not args.overwrite or args.fp_overwrite_metrics:
        command.append("--skip_existing_pred")
    if args.fp_use_open3d:
        command.append("--use_open3d")
    if args.fp_verbose:
        command.append("--verbose")
    if args.fp_disable_gating:
        command.append("--disable_gating")
    if args.fp_disable_adaptive_R:
        command.append("--disable_adaptive_R")
    if args.fp_save_track_vis:
        command.append("--save_track_vis")
    if args.fp_track_only:
        command.append("--track_only")
    if failure_start_frame:
        command.extend(["--failure_start_frame", failure_start_frame])
    return command


def build_bundletrack_evaluation_command(
    tracking_dataset_root: Path,
    mesh_path: Path,
    pred_pose_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    exclude_frame_ids_file: Optional[Path] = None,
) -> List[str]:
    command = [
        conda_executable(args.conda_prefix),
        "run",
        "--no-capture-output",
        "-n",
        args.foundationpose_conda_env,
        "python",
        str(args.foundationpose_root / "evaluate_pose_tracking_no_gt.py"),
        "--rgb_dir",
        str(tracking_dataset_root / args.tracking_left_subdir),
        "--depth_dir",
        str(tracking_dataset_root / args.tracking_depth_subdir),
        "--mask_dir",
        str(tracking_dataset_root / args.tracking_mask_subdir),
        "--mesh_path",
        str(mesh_path),
        "--pred_pose",
        str(pred_pose_dir),
        "--intrinsics",
        str(tracking_dataset_root / args.tracking_intrinsics_name),
        "--output_dir",
        str(output_dir),
        "--depth_scale",
        str(args.depth_scale),
        "--num_points",
        str(args.fp_num_points),
        "--mask_threshold",
        str(args.fp_mask_threshold),
        "--depth_min",
        str(args.fp_depth_min),
        "--depth_max",
        str(args.fp_depth_max),
        "--render_mode",
        args.fp_render_mode,
        "--render_max_faces",
        str(args.fp_render_max_faces),
        "--render_point_radius",
        str(args.fp_render_point_radius),
        "--max_observed_points",
        str(args.fp_max_observed_points),
        "--min_points",
        str(args.fp_min_points),
        "--icp_max_iterations",
        str(args.fp_icp_max_iterations),
        "--icp_tolerance",
        str(args.fp_icp_tolerance),
        "--icp_max_correspondence_dist",
        str(args.fp_icp_max_correspondence_dist),
        "--progress_interval",
        str(args.fp_progress_interval),
        "--max_frames",
        str(args.fp_max_frames),
    ]
    if args.fp_use_open3d:
        command.append("--use_open3d")
    if args.fp_verbose:
        command.append("--verbose")
    if exclude_frame_ids_file is not None:
        command.extend(["--exclude_frame_ids_file", str(exclude_frame_ids_file)])
    return command


def align_bundletrack_poses(
    pose_dir: Path,
    reference_pose_dir: Path,
    output_dir: Path,
) -> Path:
    import numpy as np

    pose_files = {path.stem: path for path in pose_dir.glob("*.txt")}
    reference_files = {path.stem: path for path in reference_pose_dir.glob("*.txt")}
    common_ids = sorted(set(pose_files) & set(reference_files), key=natural_sort_key)
    if not common_ids:
        raise RuntimeError(
            f"No shared pose ids between BundleTrack {pose_dir} and reference {reference_pose_dir}"
        )

    anchor_id = common_ids[0]
    bundle_anchor = np.loadtxt(pose_files[anchor_id], dtype=float).reshape(4, 4)
    reference_anchor = np.loadtxt(reference_files[anchor_id], dtype=float).reshape(4, 4)
    correction = np.linalg.inv(bundle_anchor) @ reference_anchor

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for frame_id, pose_path in sorted(pose_files.items(), key=lambda item: natural_sort_key(item[0])):
        pose = np.loadtxt(pose_path, dtype=float).reshape(4, 4)
        aligned_pose = pose @ correction
        if not np.isfinite(aligned_pose).all():
            raise RuntimeError(f"Non-finite aligned BundleTrack pose for frame {frame_id}")
        np.savetxt(output_dir / pose_path.name, aligned_pose, fmt="%.10f")

    metadata_path = output_dir.parent / "pose_alignment.json"
    write_json(
        metadata_path,
        {
            "method": "first_frame_rigid_alignment",
            "anchor_frame": anchor_id,
            "raw_pose_dir": str(pose_dir),
            "reference_pose_dir": str(reference_pose_dir),
            "aligned_pose_dir": str(output_dir),
            "correction_right_multiply": correction.tolist(),
            "num_poses": len(pose_files),
        },
    )
    return metadata_path


def run_depth_benchmark(
    dataset_root: Path,
    caminfo_path: Path,
    layout: DatasetLayout,
    args: argparse.Namespace,
) -> str:
    if args.skip_depth_benchmark:
        return "skipped_by_option"
    if (
        not args.overwrite
        and depth_benchmark_complete(
            dataset_root,
            layout.depth_benchmark_root,
            args.left_subdir,
            args.min_depth_groups,
        )
    ):
        print(f"[depth] existing complete benchmark found: {layout.depth_benchmark_root}", flush=True)
        return "already_complete"

    command = build_depth_command(dataset_root, caminfo_path, layout, args)
    return_code = run_logged(
        command,
        cwd=dataset_root,
        log_path=layout.logs_root / "depth_benchmark.log",
        label=f"{dataset_root.name}:depth_benchmark",
        stall_timeout_seconds=args.depth_stall_timeout_minutes * 60.0,
        dry_run=args.dry_run,
    )
    if return_code != 0:
        raise RuntimeError(f"Depth benchmark failed with exit code {return_code}")
    return "dry_run" if args.dry_run else "completed"


def run_confidence_reconstruction_comparison(
    dataset_root: Path,
    caminfo_path: Path,
    layout: DatasetLayout,
    branch_guard: GitBranchGuard,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    input_root = layout.evaluate_root / "depth_confidence_inputs"
    threshold_name = str(args.ffs_model_conf_threshold).replace(".", "_")
    laf_threshold_name = str(args.laf_conf_threshold).replace(".", "_")
    ffs_method = f"ffs_internal/threshold_{threshold_name}"
    laf_method = f"laf/threshold_{laf_threshold_name}"
    confnet_method = f"confnet/otb_threshold_{str(args.confnet_conf_threshold).replace('.', '_')}"
    ffs_depth = selected_depth_dir(layout, ffs_method)
    laf_depth = selected_depth_dir(layout, laf_method)
    confnet_depth = selected_depth_dir(layout, confnet_method)
    expected = set(image_stems(dataset_root / args.left_subdir))

    if args.overwrite or set(image_stems(ffs_depth)) != expected or not (
        set(image_stems(input_root / "raw_depth")) == expected
        and set(image_stems(input_root / "disparity")) == expected
        and set(image_stems(input_root / "laf_rgb")) == expected
        and {path.stem for path in (input_root / "topk_cost").glob("*.npy")} == expected
    ):
        launch_args = {
            "engine_file_path": args.confidence_engine_file_path,
            "dataset_root": str(dataset_root),
            "left_subdir": args.left_subdir,
            "right_subdir": args.right_subdir,
            "caminfo_path": str(caminfo_path),
            "output_subdir": str(input_root / "adaptive_depth"),
            "raw_depth_subdir": str(input_root / "raw_depth"),
            "vis_subdir": str(input_root / "vis"),
            "disparity_subdir": str(input_root / "disparity"),
            "topk_cost_subdir": str(input_root / "topk_cost"),
            "laf_rgb_subdir": str(input_root / "laf_rgb"),
            "model_confidence_subdir": str(input_root / "ffs_confidence"),
            "model_filtered_depth_subdir": str(ffs_depth),
            "model_conf_threshold": str(args.ffs_model_conf_threshold),
            "input_image_width": str(args.input_image_width),
            "input_image_height": str(args.input_image_height),
            "model_input_width": str(args.model_input_width),
            "model_input_height": str(args.model_input_height),
            "min_depth_meters": str(args.min_depth_meters),
            "max_depth_meters": str(args.max_depth_meters),
            "depth_scale": str(args.depth_scale),
            "save_input_resolution": str(args.save_input_resolution).lower(),
        }
        command = sourced_command(
            ["ros2", "launch", args.depth_launch_package, "offline_stereo_depth.launch.py",
             *[f"{key}:={value}" for key, value in launch_args.items()]],
            args.ros_setup_files,
        )
        code = run_logged(command, dataset_root, layout.logs_root / "depth_confidence_ffs.log",
                          f"{dataset_root.name}:confidence_ffs",
                          args.depth_stall_timeout_minutes * 60, args.dry_run)
        if code or (not args.dry_run and set(image_stems(ffs_depth)) != expected):
            raise RuntimeError("FFS internal confidence depth generation failed")

    if not args.skip_laf_confidence and (args.overwrite or set(image_stems(laf_depth)) != expected):
        command = [
            conda_executable(args.conda_prefix), "run", "--no-capture-output", "-n", args.laf_conda_env,
            "python", str(Path(__file__).with_name("filter_depth_laf.py")),
            "--laf_root", str(args.laf_root), "--input_root", str(input_root),
            "--output_dir", str(laf_depth),
            "--confidence_dir", str(input_root / "laf_confidence"),
            "--calibration", str(caminfo_path),
            "--threshold", str(args.laf_conf_threshold),
        ]
        if args.save_input_resolution:
            command.append("--aligned")
        code = run_logged(command, dataset_root, layout.logs_root / "depth_confidence_laf.log",
                          f"{dataset_root.name}:confidence_laf",
                          args.depth_stall_timeout_minutes * 60, args.dry_run)
        if code or (not args.dry_run and set(image_stems(laf_depth)) != expected):
            raise RuntimeError("LAF confidence depth generation failed")

    if args.overwrite or set(image_stems(confnet_depth)) != expected:
        command = [
            conda_executable(args.conda_prefix), "run", "--no-capture-output", "-n", args.confnet_conda_env,
            "python", str(Path(__file__).with_name("filter_depth_confnet.py")),
            "--confnet_root", str(args.confnet_root),
            "--checkpoint", str(args.confnet_checkpoint),
            "--dataset_root", str(dataset_root), "--right_subdir", args.right_subdir,
            "--input_root", str(input_root), "--output_dir", str(confnet_depth),
            "--calibration", str(caminfo_path),
            "--threshold", str(args.confnet_conf_threshold),
        ]
        if args.save_input_resolution:
            command.append("--aligned")
        code = run_logged(command, dataset_root, layout.logs_root / "depth_confidence_confnet.log",
                          f"{dataset_root.name}:confidence_confnet",
                          args.depth_stall_timeout_minutes * 60, args.dry_run)
        if code or (not args.dry_run and set(image_stems(confnet_depth)) != expected):
            raise RuntimeError("ConfNet OTB confidence depth generation failed")

    reconstruction_root = layout.evaluate_root / "reconstruction_confidence"
    methods = [ffs_method, confnet_method]
    if not args.skip_laf_confidence:
        methods.insert(1, laf_method)
    status = run_bundlesdf_batch(
        dataset_root, layout, branch_guard, args.bundlesdf_main_branch,
        layout.evaluate_root / "bundlesdf_confidence", reconstruction_root,
        methods, "confidence", args,
    )
    if not args.dry_run:
        existing = layout.main_reconstruction_eval_root / "summary.csv"
        new = reconstruction_root / "summary.csv"
        rows = read_csv_dicts(existing) + read_csv_dicts(new)
        comparison = layout.evaluate_root / "reconstruction_confidence_comparison.csv"
        with comparison.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[confidence] comparison: {comparison}", flush=True)
    return "completed", status


def run_bundlesdf_batch(
    dataset_root: Path,
    layout: DatasetLayout,
    branch_guard: GitBranchGuard,
    branch: str,
    output_root: Path,
    evaluation_root: Path,
    methods: Optional[Sequence[str]],
    label: str,
    args: argparse.Namespace,
    ratf_modules: Optional[Tuple[bool, bool, bool]] = None,
) -> str:
    if label == "main" and args.skip_main_reconstruction:
        return "skipped_by_option"
    if label == "gaijin" and args.skip_gaijin_reconstruction:
        return "skipped_by_option"

    summary_csv = evaluation_root / "summary.csv"
    method_summary_ok = bundlesdf_summary_complete(summary_csv)
    if methods and method_summary_ok:
        rows = read_csv_dicts(summary_csv)
        available = {row.get("method") for row in rows}
        method_summary_ok = all(method in available for method in methods)

    if not args.overwrite and method_summary_ok:
        print(f"[BundleSDF:{label}] summary already exists: {summary_csv}", flush=True)
        return "already_complete"

    branch_guard.checkout(branch)
    command = build_bundlesdf_command(
        dataset_root=dataset_root,
        layout=layout,
        output_root=output_root,
        evaluation_root=evaluation_root,
        methods=methods,
        args=args,
        ratf_modules=ratf_modules,
    )
    return_code = run_logged(
        command,
        cwd=args.bundlesdf_root,
        log_path=layout.logs_root / f"bundlesdf_{label}.log",
        label=f"{dataset_root.name}:bundlesdf_{label}",
        stall_timeout_seconds=args.bundle_outer_stall_timeout_minutes * 60.0,
        dry_run=args.dry_run,
    )
    if return_code != 0:
        raise RuntimeError(f"BundleSDF {label} batch failed with exit code {return_code}")
    return "dry_run" if args.dry_run else "completed"


def write_ratf_ablation_summary(
    layout: DatasetLayout,
    method: str,
    args: argparse.Namespace,
) -> Optional[Path]:
    rows: List[Dict[str, Any]] = []
    for name, view_weighting, depth_consistency, hole_filling in RATF_ABLATIONS:
        summary_path = layout.evaluate_root / args.ratf_ablation_eval_dir / name / "summary.csv"
        if not summary_path.is_file():
            continue
        result = next((row for row in read_csv_dicts(summary_path) if row.get("method") == method), None)
        if result is not None:
            rows.append(
                {
                    "ablation": name,
                    "view_weighting": int(view_weighting),
                    "depth_consistency": int(depth_consistency),
                    "hole_filling": int(hole_filling),
                    **result,
                }
            )
    if not rows:
        return None

    output_path = layout.evaluate_root / "ratf_ablation_summary.csv"
    fields = list(rows[0])
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def run_ratf_ablation(
    dataset_root: Path,
    layout: DatasetLayout,
    branch_guard: GitBranchGuard,
    method: str,
    args: argparse.Namespace,
) -> str:
    statuses = []
    for name, view_weighting, depth_consistency, hole_filling in RATF_ABLATIONS:
        statuses.append(
            run_bundlesdf_batch(
                dataset_root=dataset_root,
                layout=layout,
                branch_guard=branch_guard,
                branch=args.bundlesdf_gaijin_branch,
                output_root=layout.evaluate_root / args.ratf_ablation_bundlesdf_dir / name,
                evaluation_root=layout.evaluate_root / args.ratf_ablation_eval_dir / name,
                methods=[method],
                label=f"ratf_{name}",
                args=args,
                ratf_modules=(view_weighting, depth_consistency, hole_filling),
            )
        )
    write_ratf_ablation_summary(layout, method, args)
    return "dry_run" if all(status == "dry_run" for status in statuses) else "completed"


def write_reconstruction_comparison(layout: DatasetLayout, method: str) -> Optional[Path]:
    main_summary = layout.main_reconstruction_eval_root / "summary.csv"
    gaijin_summary = layout.gaijin_reconstruction_eval_root / "summary.csv"
    if not main_summary.is_file() or not gaijin_summary.is_file():
        return None

    def find_row(path: Path) -> Optional[Dict[str, str]]:
        for row in read_csv_dicts(path):
            if row.get("method") == method:
                return row
        return None

    main_row = find_row(main_summary)
    gaijin_row = find_row(gaijin_summary)
    if main_row is None or gaijin_row is None:
        return None

    output_path = layout.evaluate_root / "reconstruction_mesh_comparison.csv"
    fields = ["variant", "method", *[key for key in main_row.keys() if key != "method"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"variant": "main", **main_row})
        writer.writerow({"variant": "gaijin", **gaijin_row})
    return output_path


def run_foundationpose_experiments(
    dataset_root: Path,
    tracking_dataset_root: Optional[Path],
    layout: DatasetLayout,
    method: str,
    args: argparse.Namespace,
) -> str:
    if args.skip_foundationpose:
        return "skipped_by_option"

    tracking_intrinsics_path: Optional[Path] = None
    if args.foundationpose_depth_source == "tracking_dataset":
        if tracking_dataset_root is None:
            raise RuntimeError(
                "FoundationPose tracking uses --foundationpose_depth_source tracking_dataset, "
                "but no tracking dataset was resolved."
            )
        tracking_intrinsics_path = validate_tracking_dataset(tracking_dataset_root, args)

    failed = False
    for variant in MESH_VARIANTS:
        mesh_path = mesh_path_for(layout, variant, method)
        if not mesh_path.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Mesh for FoundationPose does not exist: {mesh_path}")

        if args.foundationpose_depth_source == "tracking_dataset":
            assert tracking_dataset_root is not None
            assert tracking_intrinsics_path is not None
            rgbd_root = tracking_dataset_root
            rgb_subdir = args.tracking_left_subdir
            mask_subdir = args.tracking_mask_subdir
            intrinsics_path = tracking_intrinsics_path
            depth_inputs = tracking_depth_inputs(tracking_dataset_root, args)
        else:
            rgbd_root = dataset_root
            rgb_subdir = args.left_subdir
            depth_inputs = (
                (
                    args.foundationpose_depth_source,
                    depth_dir_for_foundationpose(
                        layout,
                        method=method,
                        variant=variant,
                        source=args.foundationpose_depth_source,
                    ),
                ),
            )
            mask_subdir = args.mask_subdir
            intrinsics_path = dataset_root / "cam_K.txt"

        for depth_variant, depth_dir in depth_inputs:
            if not depth_dir.is_dir() and not args.dry_run:
                raise FileNotFoundError(f"FoundationPose depth dir does not exist: {depth_dir}")

            scene_source = args.foundationpose_depth_source
            if depth_variant == "confidence_filtered":
                scene_source += "__confidence_filtered"
            scene_name = (
                f"{method.replace('/', '__')}__{variant}__"
                f"{scene_source}__{rgbd_root.name}"
            )
            scene_dir = prepare_foundationpose_scene(
                scene_dir=layout.foundationpose_scene_root / scene_name,
                rgbd_root=rgbd_root,
                depth_dir=depth_dir,
                rgb_subdir=rgb_subdir,
                mask_subdir=mask_subdir,
                intrinsics_path=intrinsics_path,
            )

            for runner in FOUNDATIONPOSE_RUNNERS:
                output_dir = foundationpose_output_dir(layout, variant, runner, depth_variant)
                label = f"{variant}:{runner}:{depth_variant}"
                failure_start_frame = args.fp_failure_start_frame_map.get(
                    f"{variant}:{runner}", ""
                )
                if (
                    not args.overwrite
                    and not args.fp_overwrite_metrics
                    and foundationpose_complete(output_dir, runner, failure_start_frame)
                ):
                    print(
                        f"[FoundationPose:{label}] summary already exists: "
                        f"{foundationpose_summary_path(output_dir, runner)}",
                        flush=True,
                    )
                    continue

                if args.overwrite and not args.fp_overwrite_metrics and not args.dry_run:
                    debug_dir = foundationpose_debug_dir(output_dir, runner)
                    if debug_dir.exists():
                        shutil.rmtree(debug_dir)

                command = build_foundationpose_command(
                    scene_dir=scene_dir,
                    mesh_path=mesh_path,
                    output_dir=output_dir,
                    runner=runner,
                    args=args,
                    failure_start_frame=failure_start_frame,
                )
                log_suffix = f"_{depth_variant}" if depth_variant == "confidence_filtered" else ""
                return_code = run_logged(
                    command,
                    cwd=args.foundationpose_root,
                    log_path=layout.logs_root
                    / f"foundationpose_{variant}_{runner}{log_suffix}.log",
                    label=f"{dataset_root.name}:fp_{label}",
                    stall_timeout_seconds=args.fp_stall_timeout_minutes * 60.0,
                    dry_run=args.dry_run,
                )
                if return_code != 0:
                    failed = True
                    if args.stop_on_error:
                        raise RuntimeError(
                            f"FoundationPose {label} failed with exit code {return_code}"
                        )

    write_foundationpose_summary(layout, tracking_dataset_root, method, args)
    if args.dry_run:
        return "dry_run"
    return "failed" if failed else "completed"


def write_bundletrack_comparison(
    output_root: Path,
    tracking_dataset_root: Path,
    mesh_path: Path,
    depth_dir: Path,
    render_mode: str,
    foundationpose_summary_path: Path,
    bundletrack_summary_path: Path,
) -> Optional[Path]:
    if not foundationpose_summary_path.is_file() or not bundletrack_summary_path.is_file():
        return None

    foundationpose_summary = json.loads(foundationpose_summary_path.read_text(encoding="utf-8"))
    groups = (
        foundationpose_summary.get("results", {})
        .get("gaijin", {})
        .get("adaptive_filter", {})
    )
    references = [
        groups[name]
        for name in sorted(groups, key=lambda value: value != "raw")
        if isinstance(groups[name], Mapping)
    ]
    if not references:
        return None

    bundletrack = json.loads(bundletrack_summary_path.read_text(encoding="utf-8"))
    bundletrack.update(
        tracker="bundletrack",
        tracking_run_summary=str(output_root / "tracking" / "run_summary.yml"),
    )
    output_path = output_root / "comparison.json"
    write_json(
        output_path,
        {
            "dataset": str(tracking_dataset_root),
            "mesh": str(mesh_path),
            "depth": str(depth_dir),
            "render_mode": render_mode,
            "alignment": str(output_root / "tracking" / "pose_alignment.json"),
            "bundletrack": bundletrack,
            "foundationpose": references,
        },
    )
    print(f"[BundleTrack] comparison: {output_path}", flush=True)
    return output_path


def bundletrack_failed_frame_ids_file(
    output_root: Path,
    run_summary: Path,
    pipeline_log: Path,
) -> Path:
    output_path = output_root / "tracking" / "failed_frame_ids.txt"
    if output_path.is_file():
        return output_path

    summary_text = run_summary.read_text(encoding="utf-8")
    match = re.search(r"^num_failed_frames:\s*(\d+)\s*$", summary_text, flags=re.MULTILINE)
    expected = int(match.group(1)) if match else None
    if expected == 0:
        output_path.write_text("", encoding="utf-8")
        return output_path

    for log_path in (pipeline_log, output_root / "logs" / "bundletrack.log"):
        if not log_path.is_file():
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "\n===== " in log_text:
            log_text = "\n===== " + log_text.rsplit("\n===== ", 1)[1]
        failed_ids = re.findall(r"frame\s+(\S+)\s+finished:\s+status=FAIL", log_text)
        if expected is None or len(failed_ids) == expected:
            output_path.write_text(
                "".join(f"{frame_id}\n" for frame_id in failed_ids),
                encoding="utf-8",
            )
            return output_path

    raise RuntimeError(
        f"Could not recover {expected} BundleTrack failed frame ids for {output_root}"
    )


def bundletrack_evaluation_complete(summary_path: Path, failed_frame_ids_file: Path) -> bool:
    if not summary_path.is_file() or not failed_frame_ids_file.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    failed_count = sum(
        1
        for line in failed_frame_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return (
        summary.get("num_frames_excluded") == failed_count
        and summary.get("exclude_frame_ids_file") == str(failed_frame_ids_file)
    )


def run_bundletrack_experiment(
    dataset_root: Path,
    tracking_dataset_root: Optional[Path],
    layout: DatasetLayout,
    method: str,
    args: argparse.Namespace,
) -> str:
    if not args.run_bundletrack:
        return "skipped_by_option"
    if tracking_dataset_root is None:
        raise RuntimeError("BundleTrack requires a resolved RGBD tracking dataset")

    output_root = (
        resolve_under(layout.evaluate_root, args.bundletrack_output_dir)
        if args.bundletrack_output_dir
        else tracking_dataset_root / "evaluate" / "bundletrack"
    )
    tracking_output = output_root / "tracking"
    pose_dir = tracking_output / "ob_in_cam"
    aligned_pose_dir = tracking_output / "ob_in_cam_aligned"
    run_summary = tracking_output / "run_summary.yml"
    evaluation_output = (
        output_root / "evaluation" / f"raw_depth_gaijin_{args.fp_render_mode}"
    )
    evaluation_summary = evaluation_output / "foundationpose_debug" / "summary.json"
    mesh_path = mesh_path_for(layout, "gaijin", method)
    if args.bundletrack_reference_pose_dir:
        reference_pose_dir = expand_path(args.bundletrack_reference_pose_dir).resolve()
    else:
        reference_pose_dir = (
            foundationpose_debug_dir(
                foundationpose_output_dir(layout, "gaijin", "adaptive_filter", "raw"),
                "adaptive_filter",
            )
            / "ob_in_cam"
        )
    if not mesh_path.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Mesh for BundleTrack evaluation does not exist: {mesh_path}")

    preserve_predictions = args.overwrite and args.fp_overwrite_metrics
    if args.dry_run or not run_summary.is_file() or not pose_dir.is_dir() or (
        args.overwrite and not preserve_predictions
    ):
        tracking_command = [
            conda_executable(args.conda_prefix),
            "run",
            "--no-capture-output",
            "-n",
            args.bundlesdf_conda_env,
            "python",
            str(args.bundlesdf_root / "demo_bundletrack_jrgolden.py"),
            "--video_dir",
            str(tracking_dataset_root),
            "--out_dir",
            str(tracking_output),
            "--depth_scale",
            str(args.depth_scale),
            "--debug_level",
            str(args.bundle_debug_level),
            "--max_ba_frames",
            str(args.bundletrack_max_ba_frames),
            "--overwrite",
        ]
        return_code = run_logged(
            tracking_command,
            cwd=args.bundlesdf_root,
            log_path=layout.logs_root / "bundletrack_tracking.log",
            label=f"{dataset_root.name}:bundletrack",
            stall_timeout_seconds=args.bundletrack_stall_timeout_minutes * 60.0,
            dry_run=args.dry_run,
        )
        if return_code != 0:
            return "failed"
    else:
        print(f"[BundleTrack] existing poses found: {pose_dir}", flush=True)

    failed_frame_ids_file = tracking_output / "failed_frame_ids.txt"
    if not args.dry_run:
        failed_frame_ids_file = bundletrack_failed_frame_ids_file(
            output_root=output_root,
            run_summary=run_summary,
            pipeline_log=layout.logs_root / "bundletrack_tracking.log",
        )

    if (
        bundletrack_evaluation_complete(evaluation_summary, failed_frame_ids_file)
        and not args.overwrite
        and not args.fp_overwrite_metrics
    ):
        print(f"[BundleTrack] summary already exists: {evaluation_summary}", flush=True)
        write_bundletrack_comparison(
            output_root=output_root,
            tracking_dataset_root=tracking_dataset_root,
            mesh_path=mesh_path,
            depth_dir=tracking_dataset_root / args.tracking_depth_subdir,
            render_mode=args.fp_render_mode,
            foundationpose_summary_path=layout.evaluate_root
            / "foundationpose_tracking_summary.json",
            bundletrack_summary_path=evaluation_summary,
        )
        return "already_complete"

    if args.dry_run:
        print(
            f"[BundleTrack] would align {pose_dir} to first pose in {reference_pose_dir}",
            flush=True,
        )
    else:
        if not reference_pose_dir.is_dir():
            raise FileNotFoundError(
                "BundleTrack requires FoundationPose gaijin/adaptive_filter/raw poses "
                f"for first-frame alignment: {reference_pose_dir}"
            )
        alignment_path = align_bundletrack_poses(
            pose_dir=pose_dir,
            reference_pose_dir=reference_pose_dir,
            output_dir=aligned_pose_dir,
        )
        print(f"[BundleTrack] pose alignment: {alignment_path}", flush=True)

    evaluation_command = build_bundletrack_evaluation_command(
        tracking_dataset_root=tracking_dataset_root,
        mesh_path=mesh_path,
        pred_pose_dir=aligned_pose_dir,
        output_dir=evaluation_output,
        args=args,
        exclude_frame_ids_file=failed_frame_ids_file,
    )
    return_code = run_logged(
        evaluation_command,
        cwd=args.foundationpose_root,
        log_path=layout.logs_root / "bundletrack_evaluation.log",
        label=f"{dataset_root.name}:bundletrack_evaluation",
        stall_timeout_seconds=args.fp_stall_timeout_minutes * 60.0,
        dry_run=args.dry_run,
    )
    if return_code != 0:
        return "failed"
    if not args.dry_run:
        write_bundletrack_comparison(
            output_root=output_root,
            tracking_dataset_root=tracking_dataset_root,
            mesh_path=mesh_path,
            depth_dir=tracking_dataset_root / args.tracking_depth_subdir,
            render_mode=args.fp_render_mode,
            foundationpose_summary_path=layout.evaluate_root
            / "foundationpose_tracking_summary.json",
            bundletrack_summary_path=evaluation_summary,
        )
    return "dry_run" if args.dry_run else "completed"


def flatten_json(prefix: str, value: Any, output: Dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            flatten_json(f"{prefix}{key}_", nested, output)
        return
    output[prefix[:-1]] = value


def compact_foundationpose_metrics(row: Mapping[str, Any]) -> Dict[str, Any]:
    preferred = [
        "mesh_variant",
        "runner",
        "depth_variant",
        "depth_dir",
        "tracking_dataset",
        "num_frames_total",
        "num_frames_evaluated",
        "render_mode",
        "mask_iou_mean",
        "mask_iou_median",
        "depth_error_mean_m",
        "depth_error_median_m",
        "depth_error_mean_cm",
        "chamfer_distance_mean",
        "icp_residual_mean",
        "translation_increment_mean_m",
        "translation_increment_mean_cm",
        "rotation_increment_mean_deg",
        "mesh_path",
        "pred_pose",
        "summary_path",
        "per_frame_csv",
        "output_dir",
    ]
    return {key: row.get(key) for key in preferred if key in row}


def write_foundationpose_summary(
    layout: DatasetLayout,
    tracking_dataset_root: Optional[Path],
    method: str,
    args: argparse.Namespace,
) -> Optional[Path]:
    rows: List[Dict[str, Any]] = []
    results_by_variant: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    all_fields = {
        "mesh_variant",
        "runner",
        "depth_variant",
        "depth_dir",
        "tracking_dataset",
        "summary_path",
        "output_dir",
    }
    for variant in MESH_VARIANTS:
        results_by_variant[variant] = {runner: {} for runner in FOUNDATIONPOSE_RUNNERS}
        if tracking_dataset_root is not None:
            depth_inputs = tracking_depth_inputs(tracking_dataset_root, args)
        else:
            depth_inputs = (
                (
                    args.foundationpose_depth_source,
                    depth_dir_for_foundationpose(
                        layout,
                        method=method,
                        variant=variant,
                        source=args.foundationpose_depth_source,
                    ),
                ),
            )
        for runner in FOUNDATIONPOSE_RUNNERS:
            for depth_variant, depth_dir in depth_inputs:
                output_dir = foundationpose_output_dir(layout, variant, runner, depth_variant)
                summary_path = foundationpose_summary_path(output_dir, runner)
                if not summary_path.is_file():
                    continue
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                row: Dict[str, Any] = {
                    "mesh_variant": variant,
                    "runner": runner,
                    "depth_variant": depth_variant,
                    "depth_dir": str(depth_dir),
                    "tracking_dataset": str(tracking_dataset_root) if tracking_dataset_root else "",
                    "summary_path": str(summary_path),
                    "output_dir": str(output_dir),
                }
                flatten_json("", summary, row)
                all_fields.update(row.keys())
                rows.append(row)
                results_by_variant[variant][runner][depth_variant] = row

    if not rows:
        return None

    preferred = [
        "mesh_variant",
        "runner",
        "depth_variant",
        "depth_dir",
        "tracking_dataset",
        "num_frames_total",
        "num_frames_evaluated",
        "mask_iou_mean",
        "depth_error_mean_m",
        "chamfer_distance_mean",
        "icp_residual_mean",
        "translation_increment_mean_m",
        "rotation_increment_mean_deg",
        "summary_path",
        "output_dir",
    ]
    fields = [field for field in preferred if field in all_fields]
    fields.extend(sorted(all_fields - set(fields)))

    output_path = layout.evaluate_root / "foundationpose_tracking_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_json = {
        "dataset": str(layout.dataset_root),
        "evaluate_root": str(layout.evaluate_root),
        "tracking_dataset": str(tracking_dataset_root) if tracking_dataset_root else "",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": [compact_foundationpose_metrics(row) for row in rows],
        "results": results_by_variant,
        "results_list": rows,
        "csv_path": str(output_path),
    }
    json_output_path = layout.evaluate_root / "foundationpose_tracking_summary.json"
    write_json(json_output_path, summary_json)
    print(f"[FoundationPose] summary JSON: {json_output_path}", flush=True)
    print(f"[FoundationPose] summary CSV: {output_path}", flush=True)
    return json_output_path


def process_dataset(
    dataset_root: Path,
    branch_guard: GitBranchGuard,
    args: argparse.Namespace,
) -> DatasetResult:
    started = time.monotonic()
    layout = build_layout(dataset_root, args)
    result = DatasetResult(dataset=str(dataset_root), output_dir=str(layout.evaluate_root))
    status_path = layout.evaluate_root / (
        "pipeline_confidence_status.json" if args.run_confidence_reconstruction_comparison
        else "pipeline_status.json"
    )
    write_dataset_status(status_path, result)

    try:
        caminfo_path = validate_dataset(dataset_root, args)
        layout.evaluate_root.mkdir(parents=True, exist_ok=True)

        if args.run_confidence_reconstruction_comparison:
            result.depth_benchmark_status, result.main_reconstruction_status = (
                run_confidence_reconstruction_comparison(
                    dataset_root, caminfo_path, layout, branch_guard, args,
                )
            )
            write_dataset_status(status_path, result)
            return result

        tracking_dataset_root: Optional[Path] = None
        needs_tracking_dataset = args.run_bundletrack or (
            not args.skip_foundationpose and args.foundationpose_depth_source == "tracking_dataset"
        )
        if needs_tracking_dataset:
            tracking_dataset_root = resolve_tracking_dataset(dataset_root, args)
            validate_tracking_dataset(tracking_dataset_root, args)
            result.tracking_dataset = str(tracking_dataset_root)
            print(
                f"[{dataset_root.name}] pose tracking dataset: {tracking_dataset_root}",
                flush=True,
            )
            write_dataset_status(status_path, result)

        result.depth_benchmark_status = run_depth_benchmark(
            dataset_root,
            caminfo_path,
            layout,
            args,
        )
        write_dataset_status(status_path, result)

        result.main_reconstruction_status = run_bundlesdf_batch(
            dataset_root=dataset_root,
            layout=layout,
            branch_guard=branch_guard,
            branch=args.bundlesdf_main_branch,
            output_root=layout.main_bundle_root,
            evaluation_root=layout.main_reconstruction_eval_root,
            methods=None,
            label="main",
            args=args,
        )
        write_dataset_status(status_path, result)

        best_method = args.best_method
        selection_details: Dict[str, Any] = {}
        if not best_method:
            best_method, selection_details = select_best_method(
                layout.main_reconstruction_eval_root / "summary.csv",
                args.best_metric,
                args.best_metric_direction,
            )
        else:
            selection_details = {
                "selection_mode": "manual",
                "method": best_method,
                "summary_csv": str(layout.main_reconstruction_eval_root / "summary.csv"),
            }
        result.best_method = best_method
        write_json(
            layout.evaluate_root / "best_depth_method.json",
            {"best_method": best_method, **selection_details},
        )
        print(f"[{dataset_root.name}] best depth method: {best_method}", flush=True)
        write_dataset_status(status_path, result)

        if args.run_ratf_ablation:
            result.gaijin_reconstruction_status = run_ratf_ablation(
                dataset_root,
                layout,
                branch_guard,
                best_method,
                args,
            )
        else:
            result.gaijin_reconstruction_status = run_bundlesdf_batch(
                dataset_root=dataset_root,
                layout=layout,
                branch_guard=branch_guard,
                branch=args.bundlesdf_gaijin_branch,
                output_root=layout.gaijin_bundle_root,
                evaluation_root=layout.gaijin_reconstruction_eval_root,
                methods=[best_method],
                label="gaijin",
                args=args,
            )
            write_reconstruction_comparison(layout, best_method)
        write_dataset_status(status_path, result)

        result.foundationpose_status = run_foundationpose_experiments(
            dataset_root=dataset_root,
            tracking_dataset_root=tracking_dataset_root,
            layout=layout,
            method=best_method,
            args=args,
        )
        write_dataset_status(status_path, result)

        result.bundletrack_status = run_bundletrack_experiment(
            dataset_root=dataset_root,
            tracking_dataset_root=tracking_dataset_root,
            layout=layout,
            method=best_method,
            args=args,
        )
        write_dataset_status(status_path, result)
    except Exception as exc:
        result.message = str(exc)
        write_dataset_status(status_path, result)
        if args.stop_on_error:
            raise
        print(f"[{dataset_root.name}] ERROR: {exc}", flush=True)
    finally:
        result.elapsed_seconds = round(time.monotonic() - started, 3)
        write_dataset_status(status_path, result)

    return result


def looks_like_dataset(path: Path, args: argparse.Namespace) -> bool:
    return (
        (path / args.left_subdir).is_dir()
        and (path / args.right_subdir).is_dir()
        and (path / args.mask_subdir).is_dir()
        and (path / "cam_K.txt").is_file()
    )


def collect_datasets(args: argparse.Namespace) -> List[Path]:
    dataset_values = list(args.datasets)

    if args.dataset_list:
        list_path = expand_path(args.dataset_list)
        for line in list_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dataset_values.append(line)

    datasets = [expand_path(value).resolve() for value in dataset_values]

    for parent_value in args.dataset_parent:
        parent = expand_path(parent_value).resolve()
        for candidate in sorted(parent.glob(args.sequence_glob)):
            if candidate.is_dir() and looks_like_dataset(candidate, args):
                datasets.append(candidate.resolve())

    if not datasets:
        datasets = [Path(DEFAULT_DATASET).resolve()]

    unique: List[Path] = []
    seen = set()
    for dataset in datasets:
        if dataset not in seen:
            unique.append(dataset)
            seen.add(dataset)
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-run FFS depth, BundleSDF reconstruction, and FoundationPose evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset roots. If omitted, the jrnew-blue example path is used.",
    )
    parser.add_argument("--dataset_list", help="Text file with one dataset root per line.")
    parser.add_argument(
        "--dataset_parent",
        action="append",
        default=[],
        help="Parent directory containing multiple dataset roots. Can be passed more than once.",
    )
    parser.add_argument("--sequence_glob", default="*", help="Glob used with --dataset_parent.")
    parser.add_argument("--left_subdir", default="rgb")
    parser.add_argument("--right_subdir", default="camera2")
    parser.add_argument("--mask_subdir", default="masks")
    parser.add_argument(
        "--tracking_dataset_root",
        default="",
        help=(
            "FoundationPose RGBD tracking dataset root, or a parent directory "
            "containing per-reconstruction-dataset tracking folders."
        ),
    )
    parser.add_argument(
        "--tracking_dataset_map",
        default="",
        help=(
            "Optional text/CSV map: reconstruction_dataset_or_name,tracking_rgbd_dataset. "
            "Useful for batch runs where each object has a different tracking sequence."
        ),
    )
    parser.add_argument(
        "--tracking_dataset_subdir",
        default="",
        help="Optional RGBD tracking sequence subdirectory under each reconstruction dataset.",
    )
    parser.add_argument(
        "--tracking_dataset_suffix",
        default="",
        help="Suffix used when --tracking_dataset_root is a parent directory, e.g. _track.",
    )
    parser.add_argument("--tracking_left_subdir", default="rgb")
    parser.add_argument("--tracking_depth_subdir", default="depth")
    parser.add_argument(
        "--tracking_filtered_depth_subdir",
        default="depth_filter",
        help="Confidence-filtered tracking depth directory used for the second FP depth group.",
    )
    parser.add_argument(
        "--include_tracking_filtered_depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include confidence-filtered tracking depth, expanding FoundationPose from 4 to 8 groups.",
    )
    parser.add_argument("--tracking_mask_subdir", default="masks")
    parser.add_argument("--tracking_intrinsics_name", default="cam_K.txt")
    parser.add_argument(
        "--allow_reconstruction_dataset_for_tracking",
        action="store_true",
        help=(
            "Allow the reconstruction dataset root itself to be used as the "
            "FoundationPose RGBD tracking sequence when it also contains depth/."
        ),
    )
    parser.add_argument(
        "--caminfo_name",
        default="",
        help="Stereo calibration filename under each dataset. Empty means caminfo.txt then cam_info.txt.",
    )
    parser.add_argument(
        "--evaluate_dir",
        default="evaluate",
        help="Output root, relative to each dataset unless absolute.",
    )
    parser.add_argument(
        "--depth_benchmark_dir",
        default="",
        help=(
            "Depth benchmark root. Empty means <evaluate_dir>/depth_benchmark; "
            "explicit relative values are resolved under each dataset, and "
            "absolute values are used as-is."
        ),
    )
    parser.add_argument("--main_bundlesdf_dir", default="bundlesdf_main")
    parser.add_argument("--gaijin_bundlesdf_dir", default="bundlesdf_gaijin")
    parser.add_argument("--main_reconstruction_eval_dir", default="reconstruction_main")
    parser.add_argument("--gaijin_reconstruction_eval_dir", default="reconstruction_gaijin")
    parser.add_argument("--ratf_ablation_bundlesdf_dir", default="bundlesdf_ratf_ablation")
    parser.add_argument("--ratf_ablation_eval_dir", default="reconstruction_ratf_ablation")

    parser.add_argument("--bundlesdf_root", type=Path, default=Path(DEFAULT_BUNDLESDF_ROOT))
    parser.add_argument("--foundationpose_root", type=Path, default=Path(DEFAULT_FOUNDATIONPOSE_ROOT))
    parser.add_argument("--conda_prefix", default=DEFAULT_CONDA_PREFIX)
    parser.add_argument("--bundlesdf_conda_env", default="bundlesdf")
    parser.add_argument("--foundationpose_conda_env", default="fp_ros")
    parser.add_argument("--bundlesdf_main_branch", default="main")
    parser.add_argument("--bundlesdf_gaijin_branch", default="gaijin")
    parser.add_argument("--allow_dirty_bundlesdf", action="store_true")
    parser.add_argument(
        "--keep_bundlesdf_branch",
        action="store_true",
        help="Leave BundleSDF on the final branch instead of restoring the original branch.",
    )

    parser.add_argument("--skip_depth_benchmark", action="store_true")
    parser.add_argument(
        "--run_confidence_reconstruction_comparison", action="store_true",
        help="Generate FFS internal, LAF and ConfNet filtered depth, reconstruct on the main branch, and compare with existing results.",
    )
    parser.add_argument("--skip_laf_confidence", action="store_true", help="Omit LAF from confidence reconstruction comparisons.")
    parser.add_argument("--skip_main_reconstruction", action="store_true")
    parser.add_argument("--skip_gaijin_reconstruction", action="store_true")
    parser.add_argument("--skip_foundationpose", action="store_true")
    parser.add_argument(
        "--run_bundletrack",
        action="store_true",
        help="Run BundleTrack on raw tracking depth and evaluate it with the gaijin mesh.",
    )
    parser.add_argument(
        "--bundletrack_output_dir",
        default="",
        help=(
            "BundleTrack output directory. Empty uses <tracking_dataset>/evaluate/bundletrack; "
            "relative values use the reconstruction dataset evaluate directory."
        ),
    )
    parser.add_argument(
        "--bundletrack_reference_pose_dir",
        default="",
        help=(
            "Optional pose directory supplying only the first-frame mesh initialization. "
            "Empty uses the gaijin/adaptive_filter/raw FoundationPose output."
        ),
    )
    parser.add_argument("--bundletrack_stall_timeout_minutes", type=float, default=30.0)
    parser.add_argument(
        "--bundletrack_max_ba_frames",
        type=int,
        default=5,
        help="Maximum local BundleTrack BA frames; lower values reduce LoFTR GPU memory.",
    )
    parser.add_argument(
        "--run_ratf_ablation",
        action="store_true",
        help="Run the five reconstruction-only RATF ablation groups on the selected depth method.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--batch_status_csv", default="batch_pipeline_status.csv")

    parser.add_argument("--best_method", default="", help="Manually selected depth method.")
    parser.add_argument(
        "--best_metric",
        default="composite",
        help=(
            "Metric for selecting the best depth group. Use composite for average rank over "
            "PSNR/SSIM/IoU/coverage and LPIPS/RMSE/obs distance."
        ),
    )
    parser.add_argument("--best_metric_direction", choices=("auto", "min", "max"), default="auto")

    parser.add_argument("--depth_launch_package", default="fast_foundation_stereo")
    parser.add_argument("--depth_launch_file", default="offline_depth_benchmark.launch.py")
    parser.add_argument("--ros_setup", action="append", default=[], help="setup.bash to source before ros2 launch.")
    parser.add_argument("--engine_file_path", default=DEFAULT_ENGINE_FILE_PATH)
    parser.add_argument(
        "--confidence_engine_file_path",
        default='["/home/hc/model/ffs/23-36-37/feature_runner_fp16_5060.engine", '
                '"/home/hc/model/ffs/23-36-37/post_runner_with_prob_fp16_5060.engine"]',
    )
    parser.add_argument("--ffs_model_conf_threshold", type=float, default=0.5)
    parser.add_argument("--laf_conf_threshold", type=float, default=0.5)
    parser.add_argument("--laf_root", type=Path, default=Path("/home/hc/文档/LAF"))
    parser.add_argument("--laf_conda_env", default="laf")
    parser.add_argument("--confnet_root", type=Path, default=Path("/home/hc/文档/self-adapting-confidence"))
    parser.add_argument("--confnet_conda_env", default="self_adapting_confidence")
    parser.add_argument("--confnet_checkpoint", type=Path, default=Path("weights/census-sgm/model-25000"))
    parser.add_argument("--confnet_conf_threshold", type=float, default=0.5)
    parser.add_argument("--stereo_model_type", default="FAST_FOUNDATION_STEREO")
    parser.add_argument("--input_image_width", type=int, default=1920)
    parser.add_argument("--input_image_height", type=int, default=1080)
    parser.add_argument("--model_input_width", type=int, default=640)
    parser.add_argument("--model_input_height", type=int, default=448)
    parser.add_argument("--min_depth_meters", type=float, default=0.1)
    parser.add_argument("--max_depth_meters", type=float, default=100.0)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--save_input_resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_pairs", type=int, default=1000)
    parser.add_argument("--median_kernel_size", type=int, default=5)
    parser.add_argument("--bilateral_d", type=int, default=5)
    parser.add_argument("--bilateral_sigma_color", type=float, default=0.05)
    parser.add_argument("--bilateral_sigma_space", type=float, default=5.0)
    parser.add_argument("--use_temporal_confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--threshold_sweep", default="0.20,0.30,0.35,0.40,0.50")
    parser.add_argument(
        "--depth_launch_arg",
        action="append",
        default=[],
        help="Extra ros2 launch argument in name:=value form. Can be passed more than once.",
    )
    parser.add_argument("--min_depth_groups", type=int, default=7)
    parser.add_argument("--depth_stall_timeout_minutes", type=float, default=30.0)

    parser.add_argument("--bundle_device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bundle_lpips_net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--bundle_mesh_sample_points", type=int, default=200000)
    parser.add_argument("--bundle_distance_method", choices=("sampled", "exact"), default="sampled")
    parser.add_argument("--bundle_use_segmenter", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bundle_use_gui", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bundle_debug_level", type=int, default=2)
    parser.add_argument("--bundle_stride", type=int, default=1)
    parser.add_argument(
        "--bundle_shorter_side",
        "--bundlesdf_shorter_side",
        dest="bundle_shorter_side",
        type=int,
        default=720,
        help=(
            "BundleSDF input resize short-side length for both main and gaijin branches. "
            "Use <=0 to keep the original RGB/depth/mask resolution."
        ),
    )
    parser.add_argument("--bundle_stall_timeout_minutes", type=float, default=20.0)
    parser.add_argument("--bundle_outer_stall_timeout_minutes", type=float, default=0.0)
    parser.add_argument("--bundle_max_retries", type=int, default=1)

    parser.add_argument(
        "--foundationpose_depth_source",
        choices=(
            "tracking_dataset",
            "selected_depth",
            "main_filtered_depth",
            "variant_filtered_depth",
        ),
        default="tracking_dataset",
        help=(
            "Depth images used by FoundationPose and its no-GT evaluator. "
            "tracking_dataset uses the separate RGBD sequence requested by evaluate.md; "
            "the other values preserve the older reconstruction-sequence behavior."
        ),
    )
    parser.add_argument("--fp_num_points", type=int, default=2000)
    parser.add_argument("--fp_mask_threshold", type=float, default=0.0)
    parser.add_argument("--fp_depth_min", type=float, default=0.001)
    parser.add_argument("--fp_depth_max", type=float, default=0.0)
    parser.add_argument(
        "--fp_render_mode",
        choices=("faces", "points"),
        default="points",
        help=(
            "FoundationPose no-GT metric render strategy. points is much faster for long "
            "RGBD tracking sequences; use faces for the slower triangle-rasterized metric."
        ),
    )
    parser.add_argument("--fp_render_max_faces", type=int, default=0)
    parser.add_argument("--fp_render_point_radius", type=int, default=1)
    parser.add_argument("--fp_max_observed_points", type=int, default=20000)
    parser.add_argument("--fp_min_points", type=int, default=20)
    parser.add_argument("--fp_icp_max_iterations", type=int, default=20)
    parser.add_argument("--fp_icp_tolerance", type=float, default=1e-6)
    parser.add_argument("--fp_icp_max_correspondence_dist", type=float, default=0.05)
    parser.add_argument("--fp_use_open3d", action="store_true")
    parser.add_argument("--fp_verbose", dest="fp_verbose", action="store_true", default=True)
    parser.add_argument("--no_fp_verbose", dest="fp_verbose", action="store_false")
    parser.add_argument("--fp_progress_interval", type=int, default=20)
    parser.add_argument("--fp_max_frames", type=int, default=0)
    parser.add_argument(
        "--fp_overwrite_metrics",
        action="store_true",
        help=(
            "Recompute FoundationPose no-GT summary/per-frame metrics while preserving "
            "existing ob_in_cam predictions and passing --skip_existing_pred."
        ),
    )
    parser.add_argument("--fp_est_refine_iter", type=int, default=5)
    parser.add_argument("--fp_track_refine_iter", type=int, default=2)
    parser.add_argument("--fp_shorter_side", type=int, default=360)
    parser.add_argument("--fp_debug", type=int, default=0)
    parser.add_argument(
        "--fp_save_track_vis",
        action="store_true",
        help="Save FoundationPose track_vis overlays without enabling the GUI debug window.",
    )
    parser.add_argument(
        "--fp_track_only",
        action="store_true",
        help="Run FoundationPose tracking and skip no-GT metric evaluation.",
    )
    parser.add_argument(
        "--fp_failure_start_frame",
        action="append",
        default=[],
        metavar="GROUP=FRAME",
        help=(
            "Exclude FRAME and all later frames from one FoundationPose group's metrics. "
            "GROUP is main:standard, main:adaptive_filter, gaijin:standard, or "
            "gaijin:adaptive_filter. Repeat for multiple groups."
        ),
    )
    parser.add_argument("--fp_fps", type=float, default=30.0)
    parser.add_argument("--fp_q_pos", type=float, default=1e-4)
    parser.add_argument("--fp_q_rot", type=float, default=1e-4)
    parser.add_argument("--fp_q_vel", type=float, default=1e-2)
    parser.add_argument("--fp_q_omega", type=float, default=1e-2)
    parser.add_argument("--fp_sigma_pos_obs", type=float, default=0.005)
    parser.add_argument("--fp_sigma_rot_obs", type=float, default=0.02)
    parser.add_argument("--fp_sigma_pos_motion", type=float, default=0.05)
    parser.add_argument("--fp_sigma_rot_motion", type=float, default=0.35)
    parser.add_argument("--fp_min_quality", type=float, default=0.01)
    parser.add_argument("--fp_gate_threshold", type=float, default=100.0)
    parser.add_argument("--fp_reject_cov_increase", type=float, default=1e-4)
    parser.add_argument("--fp_disable_gating", action="store_true")
    parser.add_argument("--fp_disable_adaptive_R", action="store_true")
    parser.add_argument("--fp_stall_timeout_minutes", type=float, default=30.0)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.bundlesdf_root = args.bundlesdf_root.expanduser().resolve()
    args.foundationpose_root = args.foundationpose_root.expanduser().resolve()
    args.laf_root = args.laf_root.expanduser().resolve()
    args.confnet_root = args.confnet_root.expanduser().resolve()
    args.confnet_checkpoint = resolve_under(args.confnet_root, args.confnet_checkpoint)
    args.tracking_dataset_map_entries = load_tracking_dataset_map(args.tracking_dataset_map)
    args.fp_failure_start_frame_map = parse_foundationpose_failure_starts(
        args.fp_failure_start_frame
    )

    if args.ros_setup:
        args.ros_setup_files = [expand_path(path).resolve() for path in args.ros_setup]
    else:
        args.ros_setup_files = default_ros_setup_files()

    missing_setups = [path for path in args.ros_setup_files if not path.is_file()]
    if missing_setups:
        raise FileNotFoundError(
            "ROS setup file(s) do not exist: " + ", ".join(str(path) for path in missing_setups)
        )

    if not (args.bundlesdf_root / "batch_depth_benchmark.py").is_file():
        raise FileNotFoundError(args.bundlesdf_root / "batch_depth_benchmark.py")
    if not (args.foundationpose_root / "evaluate_pose_tracking_no_gt.py").is_file():
        raise FileNotFoundError(args.foundationpose_root / "evaluate_pose_tracking_no_gt.py")
    if args.run_bundletrack and not (args.bundlesdf_root / "demo_bundletrack_jrgolden.py").is_file():
        raise FileNotFoundError(args.bundlesdf_root / "demo_bundletrack_jrgolden.py")
    if args.bundle_max_retries < 0:
        raise ValueError("--bundle_max_retries must be non-negative")
    if args.bundletrack_max_ba_frames < 2:
        raise ValueError("--bundletrack_max_ba_frames must be at least 2")
    if args.run_ratf_ablation and args.skip_gaijin_reconstruction:
        raise ValueError("--run_ratf_ablation cannot be combined with --skip_gaijin_reconstruction")
    if args.run_ratf_ablation:
        args.skip_foundationpose = True
    if args.depth_scale <= 0:
        raise ValueError("--depth_scale must be positive")
    if not all(0 <= threshold <= 1 for threshold in (
        args.ffs_model_conf_threshold, args.laf_conf_threshold, args.confnet_conf_threshold,
    )):
        raise ValueError("Confidence thresholds must be in [0, 1]")
    if args.run_confidence_reconstruction_comparison and not args.skip_laf_confidence and not (
        args.laf_root / "saved_models/LAFNet_CVPR2019_networks.pth"
    ).is_file():
        raise FileNotFoundError("LAF pretrained weights are missing")
    if args.run_confidence_reconstruction_comparison and not args.confnet_checkpoint.with_suffix(".index").is_file():
        raise FileNotFoundError(f"ConfNet checkpoint is missing: {args.confnet_checkpoint}")
    return args


def main() -> int:
    args = normalize_args(build_arg_parser().parse_args())
    datasets = collect_datasets(args)
    print("Datasets:")
    for dataset in datasets:
        print(f"  {dataset}")
    print("ROS setup files:")
    for setup_file in args.ros_setup_files:
        print(f"  {setup_file}")

    batch_status_path = resolve_under(Path.cwd(), args.batch_status_csv)
    results: List[DatasetResult] = []
    branch_guard = GitBranchGuard(
        repo_root=args.bundlesdf_root,
        restore_original=not args.keep_bundlesdf_branch,
        allow_dirty=args.allow_dirty_bundlesdf,
        dry_run=args.dry_run,
    )

    try:
        for index, dataset_root in enumerate(datasets, 1):
            print(f"\n===== Dataset {index}/{len(datasets)}: {dataset_root} =====", flush=True)
            result = process_dataset(dataset_root, branch_guard, args)
            results.append(result)
            write_batch_status(batch_status_path, results)
            if result.message and args.stop_on_error:
                break
    finally:
        branch_guard.restore()
        if results:
            write_batch_status(batch_status_path, results)

    failed = [
        result
        for result in results
        if result.message
        or result.foundationpose_status == "failed"
        or result.bundletrack_status == "failed"
    ]
    print(f"\nBatch status: {batch_status_path}")
    if failed:
        print(f"Completed with {len(failed)} failed dataset(s).")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
