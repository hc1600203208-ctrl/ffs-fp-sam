#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$root" -B "$root/build" -DBUILD_GROUND_TRACKER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" --target ground_tracker_udp -j"${JOBS:-4}"
if ldd "$root/build/ground_tracker_udp" | grep -Eq 'librcl|libros|libcv_bridge'; then
  echo "Ground binary unexpectedly links against ROS" >&2
  exit 1
fi
