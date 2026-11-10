#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$root/.." && pwd)"
config="${1:-$repo/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml}"
port="${2:-50100}"
app_port="${3:-50101}"
if [[ ! -x "$root/build/ground_tracker_udp" ]]; then
  echo "Build the ground tracker first: $root/scripts/build_ground.sh" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$root/build/lib:$root/build:${LD_LIBRARY_PATH:-}"
exec "$root/build/ground_tracker_udp" "$config" "$port" "$app_port"
