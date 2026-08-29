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
     baseline and improved meshes on the configured RGBD tracking sequence.

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


def foundationpose_complete(output_dir: Path, runner: str) -> bool:
    summary_path = foundationpose_summary_path(output_dir, runner)
    per_frame_path = summary_path.parent / "per_frame_metrics.csv"
    return summary_path.is_file() and per_frame_path.is_file()


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
    for relative in (
        args.tracking_left_subdir,
        args.tracking_depth_subdir,
        args.tracking_mask_subdir,
    ):
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
    return command


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
            depth_dir = tracking_dataset_root / args.tracking_depth_subdir
            mask_subdir = args.tracking_mask_subdir
            intrinsics_path = tracking_intrinsics_path
        else:
            rgbd_root = dataset_root
            rgb_subdir = args.left_subdir
            depth_dir = depth_dir_for_foundationpose(
                layout,
                method=method,
                variant=variant,
                source=args.foundationpose_depth_source,
            )
            mask_subdir = args.mask_subdir
            intrinsics_path = dataset_root / "cam_K.txt"

        if not depth_dir.is_dir() and not args.dry_run:
            raise FileNotFoundError(f"FoundationPose depth dir does not exist: {depth_dir}")

        scene_name = (
            f"{method.replace('/', '__')}__{variant}__"
            f"{args.foundationpose_depth_source}__{rgbd_root.name}"
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
            output_dir = layout.foundationpose_output_root / variant / runner
            if (
                not args.overwrite
                and not args.fp_overwrite_metrics
                and foundationpose_complete(output_dir, runner)
            ):
                print(
                    f"[FoundationPose:{variant}:{runner}] summary already exists: "
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
            )
            return_code = run_logged(
                command,
                cwd=args.foundationpose_root,
                log_path=layout.logs_root / f"foundationpose_{variant}_{runner}.log",
                label=f"{dataset_root.name}:fp_{variant}_{runner}",
                stall_timeout_seconds=args.fp_stall_timeout_minutes * 60.0,
                dry_run=args.dry_run,
            )
            if return_code != 0:
                failed = True
                if args.stop_on_error:
                    raise RuntimeError(
                        f"FoundationPose {variant}/{runner} failed with exit code {return_code}"
                    )

    write_foundationpose_summary(layout, tracking_dataset_root)
    if args.dry_run:
        return "dry_run"
    return "failed" if failed else "completed"


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
) -> Optional[Path]:
    rows: List[Dict[str, Any]] = []
    results_by_variant: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_fields = {"mesh_variant", "runner", "tracking_dataset", "summary_path", "output_dir"}
    for variant in MESH_VARIANTS:
        results_by_variant[variant] = {}
        for runner in FOUNDATIONPOSE_RUNNERS:
            output_dir = layout.foundationpose_output_root / variant / runner
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
                "tracking_dataset": str(tracking_dataset_root) if tracking_dataset_root else "",
                "summary_path": str(summary_path),
                "output_dir": str(output_dir),
            }
            flatten_json("", summary, row)
            all_fields.update(row.keys())
            rows.append(row)
            results_by_variant[variant][runner] = row

    if not rows:
        return None

    preferred = [
        "mesh_variant",
        "runner",
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
    status_path = layout.evaluate_root / "pipeline_status.json"
    write_dataset_status(status_path, result)

    try:
        caminfo_path = validate_dataset(dataset_root, args)
        layout.evaluate_root.mkdir(parents=True, exist_ok=True)

        tracking_dataset_root: Optional[Path] = None
        if not args.skip_foundationpose and args.foundationpose_depth_source == "tracking_dataset":
            tracking_dataset_root = resolve_tracking_dataset(dataset_root, args)
            validate_tracking_dataset(tracking_dataset_root, args)
            result.tracking_dataset = str(tracking_dataset_root)
            print(
                f"[{dataset_root.name}] FoundationPose tracking dataset: "
                f"{tracking_dataset_root}",
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
    parser.add_argument("--skip_main_reconstruction", action="store_true")
    parser.add_argument("--skip_gaijin_reconstruction", action="store_true")
    parser.add_argument("--skip_foundationpose", action="store_true")
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
    args.tracking_dataset_map_entries = load_tracking_dataset_map(args.tracking_dataset_map)

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
    if args.bundle_max_retries < 0:
        raise ValueError("--bundle_max_retries must be non-negative")
    if args.depth_scale <= 0:
        raise ValueError("--depth_scale must be positive")
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

    failed = [result for result in results if result.message or result.foundationpose_status == "failed"]
    print(f"\nBatch status: {batch_status_path}")
    if failed:
        print(f"Completed with {len(failed)} failed dataset(s).")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
