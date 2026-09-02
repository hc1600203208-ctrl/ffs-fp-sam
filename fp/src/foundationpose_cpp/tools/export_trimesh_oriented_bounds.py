#!/usr/bin/env python3
"""Export trimesh oriented-bounds sidecar files for C++ FoundationPose."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(mesh_file: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        meshes = [geometry for geometry in mesh.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No Trimesh geometry found in scene: {mesh_file}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type from trimesh.load: {type(mesh)!r}")
    return mesh


def compute_mesh_diameter(mesh: trimesh.Trimesh) -> float:
    """Compute a fast diameter estimate using the minimum enclosing sphere."""
    try:
        _, radius = trimesh.nsphere.minimum_nsphere(mesh)
        diameter = float(radius) * 2.0
        if np.isfinite(diameter) and diameter > 0.0:
            return diameter
    except Exception:
        pass

    # Conservative fallback if the nsphere solver fails for a degenerate mesh.
    return float(np.linalg.norm(mesh.bounding_box_oriented.extents))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export to_origin.txt, extents.txt, and diameter.txt for C++ FoundationPose."
        )
    )
    parser.add_argument("mesh_file", type=Path, help="Path to the mesh file loaded by FoundationPose.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for sidecar files. Defaults to the mesh file directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecar files.",
    )
    args = parser.parse_args()

    mesh_file = args.mesh_file.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None else mesh_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_origin_path = output_dir / "to_origin.txt"
    extents_path = output_dir / "extents.txt"
    diameter_path = output_dir / "diameter.txt"
    if not args.overwrite:
        existing = [path for path in (to_origin_path, extents_path, diameter_path) if path.exists()]
        if len(existing) == 3:
            print(f"Sidecar files already exist under {output_dir}, nothing to do.")
            return

    mesh = load_mesh(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    diameter = compute_mesh_diameter(mesh)

    if args.overwrite or not to_origin_path.exists():
        np.savetxt(to_origin_path, to_origin.reshape(4, 4), fmt="%.18e")
        print(f"Wrote {to_origin_path}")
    else:
        print(f"Kept existing {to_origin_path}")

    if args.overwrite or not extents_path.exists():
        np.savetxt(extents_path, np.asarray(extents).reshape(1, 3), fmt="%.18e")
        print(f"Wrote {extents_path}")
    else:
        print(f"Kept existing {extents_path}")

    if args.overwrite or not diameter_path.exists():
        np.savetxt(diameter_path, np.asarray([[diameter]]), fmt="%.18e")
        print(f"Wrote {diameter_path}")
    else:
        print(f"Kept existing {diameter_path}")


if __name__ == "__main__":
    main()
