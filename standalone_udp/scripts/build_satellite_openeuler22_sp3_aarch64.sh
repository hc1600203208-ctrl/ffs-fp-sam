#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
config="${1:-$repo/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml}"
output="$repo/standalone_udp/dist"
if [[ "$config" != /home/* ]]; then
  echo "CONFIG must be an absolute path under /home: $config" >&2
  exit 2
fi
mkdir -p "$output"
docker build --platform linux/arm64 \
  -f "$repo/standalone_udp/scripts/openeuler22_sp3_aarch64.Dockerfile" \
  -t satellite-udp-builder:openeuler22sp3-aarch64 "$repo/standalone_udp"
docker run --rm --platform linux/arm64 --user "$(id -u):$(id -g)" \
  -v /home:/home:ro -v "$output":/out \
  -e SATELLITE_REPO="$repo" -e SATELLITE_CONFIG="$config" \
  satellite-udp-builder:openeuler22sp3-aarch64 \
  bash -c 'cmake -S "$SATELLITE_REPO/standalone_udp" -B /tmp/satellite-udp-build -DCMAKE_BUILD_TYPE=Release -DBUILD_GROUND_TRACKER=OFF &&
    cmake --build /tmp/satellite-udp-build -j"$(nproc)" &&
    python3 "$SATELLITE_REPO/standalone_udp/scripts/make_satellite_bundle.py" \
      --binary /tmp/satellite-udp-build/satellite_renderer_udp --config "$SATELLITE_CONFIG" \
      --output /out --target openeuler22sp3-aarch64'
