#!/bin/bash
# Rebuild the seven quad box executables whose kernels changed, copy them
# over the copies in build_quad/, and regenerate only those result files.
#
# Submit from a JLSE login node:
#   qsub -A pepper_hep -n 1 -t 360 -q gpu_b200 --mode script \
#        scripts/qsub_b200_affected_quad.sh
#
set -euo pipefail

REPO=/home/rbarik/qcdloop
cd "$REPO"

# Non-interactive job shells do not always define `module`.
if ! type module >/dev/null 2>&1; then
   # shellcheck disable=SC1091
   source /etc/profile
fi

# shellcheck disable=SC1091
source scripts/prepare_b200.sh

export QCDLOOP_TARGETS="boxGPU_test_quad_B11 boxGPU_test_quad_B12 boxGPU_test_quad_B13 boxGPU_test_quad_B14 boxGPU_test_quad_B15 boxGPU_test_quad_BIN2 boxGPU_test_quad_BIN4"

# shellcheck disable=SC1091
source scripts/build_QCDLoops_Kokkos_b200.sh "$REPO"

mkdir -p "$REPO/build_quad" "$REPO/result_quad"

for target in $QCDLOOP_TARGETS; do
   cp -f "$REPO/build_b200/$target" "$REPO/build_quad/$target"
   tag=${target#boxGPU_test_quad_}
   # Driver prints three preamble lines before the data rows. Keep only
   # the rows so the files match the existing result_quad format.
   "$REPO/build_quad/$target" 1 100000 \
      | grep "^${tag}," > "$REPO/result_quad/${tag}_val.txt"
   lines=$(wc -l < "$REPO/result_quad/${tag}_val.txt")
   echo "$tag lines=$lines"
   if [ "$lines" -ne 100000 ]; then
      echo "ERROR: $tag expected 100000 data rows, got $lines" >&2
      exit 1
   fi
done

echo "Replaced binaries in $REPO/build_quad and results in $REPO/result_quad"
