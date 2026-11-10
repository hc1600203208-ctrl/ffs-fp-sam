#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$root/.." && pwd)"
config="${1:-$repo/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml}"
satellite_ip="${2:?usage: start_ground_app.sh [tracker_yaml] <satellite_ip> [satellite_port]}"
satellite_port="${3:-50000}"
exec /usr/bin/python3 "$root/app/satellite_stereo_multi_udp_app.py" \
  --config "$config" --satellite-ip "$satellite_ip" --satellite-port "$satellite_port"
