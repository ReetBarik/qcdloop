#!/usr/bin/env bash
set -euo pipefail

topologies=(tadpole bubble triangle box)
batch_sizes=(
  200000
  400000
  800000
  1600000
  3200000
  6400000
  12800000
  25600000
  51200000
  102400000
)

for topology in "${topologies[@]}"; do
  exe="./build/${topology}GPU_test"
  if [[ ! -x "$exe" ]]; then
    echo "Skipping ${topology}: executable not found or not executable at ${exe}" >&2
    continue
  fi

  for batch_size in "${batch_sizes[@]}"; do
    echo "Running: ${exe} 0 ${batch_size}"
    "$exe" 0 "$batch_size"
  done
done
