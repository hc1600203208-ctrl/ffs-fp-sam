#!/usr/bin/env python3
"""Collect a target-built renderer and its runtime assets into an offline tarball."""
import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml


TARGETS = {
    "ubuntu20-x86_64": {
        "archive": "satellite_udp_ubuntu20_x86_64",
        "system": "Ubuntu 20.04 x86_64",
        "machine": "x86_64",
        "glibc": "2.31",
        "loader": "ld-linux-x86-64.so.2",
        "loader_path": "/lib64/ld-linux-x86-64.so.2",
        "packages": ("libc6", "libstdc++6", "libgcc-s1", "libpng16-16", "zlib1g"),
    },
    "openeuler22sp3-aarch64": {
        "archive": "satellite_udp_openeuler22sp3_aarch64",
        "system": "openEuler 22.03 LTS SP3 aarch64",
        "machine": "aarch64",
        "glibc": "2.34",
        "loader": "ld-linux-aarch64.so.1",
        "loader_path": "/lib/ld-linux-aarch64.so.1",
    },
}


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


def collect_licenses(root, target, libraries):
    if target == "ubuntu20-x86_64":
        for package in TARGETS[target]["packages"]:
            copyright_file = Path("/usr/share/doc") / package / "copyright"
            if copyright_file.is_file():
                shutil.copy2(copyright_file, root / "licenses" / f"{package}.copyright")
        return
    packages = set()
    for library in libraries:
        query = subprocess.run(["rpm", "-qf", str(library), "--qf", "%{NAME}\n"],
                               capture_output=True, text=True, check=True)
        packages.add(query.stdout.strip())
    package_info = []
    for package in sorted(packages):
        metadata = subprocess.check_output(
            ["rpm", "-q", package, "--qf", "%{NAME}|%{VERSION}|%{RELEASE}|%{LICENSE}\n"],
            text=True).strip().split("|", 3)
        package_info.append(dict(zip(("name", "version", "release", "license"), metadata)))
        listing = subprocess.check_output(["rpm", "-ql", package], text=True).splitlines()
        for filename in listing:
            if not (filename.startswith("/usr/share/licenses/") or
                    filename.startswith(f"/usr/share/doc/{package}/")):
                continue
            source = Path(filename)
            if source.is_file():
                destination = root / "licenses" / package / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    (root / "licenses/runtime-packages.json").write_text(
        json.dumps(package_info, ensure_ascii=False, indent=2) + "\n")
    # openEuler's zlib runtime RPM omits its documentation files. The installed
    # development header carries the upstream notice for the bundled libz.
    header = Path("/usr/include/zlib.h").read_text()
    notice = header.split("*/", 1)[0] + "*/\n"
    (root / "licenses/zlib.LICENSE").write_text(notice)


def build(binary, config_file, output_dir, target):
    spec = TARGETS[target]
    config = yaml.safe_load(config_file.read_text())
    params = config["foundationpose_stereo_tracker_multi_node"]["ros__parameters"]
    meshes = [Path(path) for path in params["tracked_mesh_paths"]]
    calibration = Path(params["caminfo_path"])
    if not calibration.is_file() or any(not path.is_file() for path in meshes):
        raise FileNotFoundError("tracker calibration or mesh is missing")
    if platform.machine() != spec["machine"] or platform.libc_ver()[1] != spec["glibc"]:
        raise RuntimeError(f'Build inside {spec["system"]} with glibc {spec["glibc"]}')
    loader = Path(spec["loader_path"])
    if not loader.is_file():
        raise FileNotFoundError(f"Target dynamic loader is missing: {loader}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / spec["archive"]
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
        libraries = dependencies(binary) | {loader}
        if not any(library.name == spec["loader"] for library in libraries):
            raise RuntimeError("Dynamic loader was not collected")
        for library in libraries:
            shutil.copy2(library, root / "lib" / library.name)
        collect_licenses(root, target, libraries)
        (root / "run_satellite.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'root="$(cd "$(dirname "$0")" && pwd)"\n'
            f'exec "$root/lib/{spec["loader"]}" --library-path "$root/lib" '
            '"$root/bin/satellite_renderer_udp" "${1:-50000}" "$root/assets"\n')
        (root / "run_satellite.sh").chmod(0o755)
        (root / "DEPLOY.txt").write_text(
            f'Target: {spec["system"]}\n'
            '1. On the target, confirm: uname -m ; cat /etc/os-release\n'
            '2. Stop any old satellite_renderer_udp process using UDP port 50000.\n'
            '3. In this directory, verify: sha256sum -c SHA256SUMS\n'
            '4. Configure the satellite Ethernet address and check ground connectivity.\n'
            '5. Run: ./run_satellite.sh 50000\n'
            '6. Start the ground App and click Configure and start satellite.\n'
            'The renderer waits for the ground App; it sends no frames until started.\n'
            'Use the same object order, mesh geometry and calibration as the ground tracker.\n'
            'UDP 50000 must be reachable in both directions; ground tracker ports stay local.\n')
        files = {str(path.relative_to(root)): sha256(path) for path in root.rglob("*") if path.is_file()}
        manifest = {"target": spec["system"], "glibc": platform.libc_ver()[1],
                    "machine": platform.machine(), "loader": spec["loader"],
                    "objects": [f"assets/meshes/{i}/{mesh.name}" for i, mesh in enumerate(meshes)],
                    "files_sha256": files}
        (root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        (root / "SHA256SUMS").write_text("".join(
            f"{digest}  {name}\n" for name, digest in sorted(files.items())))
        archive = output_dir / f'{spec["archive"]}.tar.gz'
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=root.name)
        print(archive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, default="ubuntu20-x86_64")
    args = parser.parse_args()
    build(args.binary, args.config, args.output, args.target)


if __name__ == "__main__":
    main()
