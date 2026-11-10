#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
config="${1:-$repo/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml}"
output="$repo/standalone_udp/dist"
mkdir -p "$output"
docker build -f "$repo/standalone_udp/scripts/ubuntu20.Dockerfile" \
  -t satellite-udp-builder:ubuntu20 "$repo/standalone_udp"
docker run --rm --user "$(id -u):$(id -g)" \
  -v /home:/home:ro -v "$output":/out \
  satellite-udp-builder:ubuntu20 \
  bash -lc "cmake -S '$repo/standalone_udp' -B /tmp/satellite-udp-build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/satellite-udp-build -j\$(nproc) && \
    python3 '$repo/standalone_udp/scripts/make_satellite_bundle.py' \
      --binary /tmp/satellite-udp-build/satellite_renderer_udp --config '$config' --output /out"
