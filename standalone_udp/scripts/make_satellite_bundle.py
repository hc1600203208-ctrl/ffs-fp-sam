#!/usr/bin/env python3
"""Collect an Ubuntu 20.04 renderer and its runtime assets into one offline tarball."""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependencies(binary):
    output = subprocess.check_output(["ldd", str(binary)], text=True)
    result = set()
    for line in output.splitlines():
        if "not found" in line:
            raise RuntimeError("Missing runtime library: " + line)
        match = re.search(r"=>\s+(/\S+)", line)
        if match:
            result.add(Path(match.group(1)))
        else:
            match = re.search(r"^\s*(/\S+)", line)
            if match:
                result.add(Path(match.group(1)))
    return result


def build(binary, config_file, output_dir):
    config = yaml.safe_load(config_file.read_text())
    params = config["foundationpose_stereo_tracker_multi_node"]["ros__parameters"]
    meshes = [Path(path) for path in params["tracked_mesh_paths"]]
    calibration = Path(params["caminfo_path"])
    if not calibration.is_file() or any(not path.is_file() for path in meshes):
        raise FileNotFoundError("tracker calibration or mesh is missing")
    if platform.machine() != "x86_64" or platform.libc_ver()[1] != "2.31":
        raise RuntimeError("Build this package inside the provided Ubuntu 20.04 x86_64 image (glibc 2.31)")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / "satellite_udp_ubuntu20_x86_64"
        (root / "bin").mkdir(parents=True)
        (root / "lib").mkdir()
        (root / "assets/meshes").mkdir(parents=True)
        (root / "licenses").mkdir()
        repository = Path(__file__).resolve().parents[2]
        for name in ("LICENSE", "NOTICE", "AUTHORS"):
            source = repository / name
            if source.is_file():
                shutil.copy2(source, root / "licenses" / name)
        shutil.copy2(binary, root / "bin/satellite_renderer_udp")
        shutil.copy2(calibration, root / "assets/calibration.txt")
        for index, mesh in enumerate(meshes):
            destination = root / "assets/meshes" / str(index)
            shutil.copytree(mesh.parent, destination, symlinks=False)
            if not (destination / mesh.name).is_file():
                raise RuntimeError("mesh copy failed: " + str(mesh))
        for library in dependencies(binary):
            shutil.copy2(library, root / "lib" / library.name)
        for package in ("libc6", "libstdc++6", "libgcc-s1", "libpng16-16", "zlib1g"):
            copyright_file = Path("/usr/share/doc") / package / "copyright"
            if copyright_file.is_file():
                shutil.copy2(copyright_file, root / "licenses" / f"{package}.copyright")
        (root / "run_satellite.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'root="$(cd "$(dirname "$0")" && pwd)"\n'
            'exec "$root/lib/ld-linux-x86-64.so.2" --library-path "$root/lib" '
            '"$root/bin/satellite_renderer_udp" "${1:-50000}" "$root/assets"\n')
        (root / "run_satellite.sh").chmod(0o755)
        files = {str(path.relative_to(root)): sha256(path) for path in root.rglob("*") if path.is_file()}
        manifest = {"target": "Ubuntu 20.04 x86_64", "glibc": platform.libc_ver()[1],
                    "objects": [f"assets/meshes/{i}/{mesh.name}" for i, mesh in enumerate(meshes)],
                    "files_sha256": files}
        (root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        (root / "SHA256SUMS").write_text("".join(
            f"{digest}  {name}\n" for name, digest in sorted(files.items())))
        archive = output_dir / "satellite_udp_ubuntu20_x86_64.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=root.name)
        print(archive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.binary, args.config, args.output)


if __name__ == "__main__":
    main()
