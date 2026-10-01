#!/bin/bash
# 100k accuracy dump from the already-built MI250 xpmath binaries.
# Writes result_xpmath/mi250/<backend>/<INTEGRAL>_val.txt.
#
#   qsub -A pepper_hep -n 1 -t 240 -q gpu_amd_mi250 --mode script \
#        --cwd /home/rbarik/qcdloop --jobname xpmath_mi250_100k \
#        -O /home/rbarik/qcdloop/build_logs/xpmath_mi250_100k \
#        scripts/qsub_mi250_xpmath_100k.sh
set -euo pipefail

REPO=/home/rbarik/qcdloop
cd "$REPO"

if ! type module >/dev/null 2>&1; then
   # shellcheck disable=SC1091
   source /etc/profile
fi

# shellcheck disable=SC1091
source scripts/prepare_mi250.sh

export LD_LIBRARY_PATH="/soft/compilers/gcc/13.3.0/x86_64-suse-linux/lib64:${REPO}/kokkos/install/kokkos-5.1.0-mi250/Release/lib64:${LD_LIBRARY_PATH:-}"
export BUILD_DIR="${REPO}/build_mi250_xp"

for be in dd ff qf tf; do
   scripts/run_xpmath_box_accuracy.sh "${be}" 100000 mi250
done

echo "MI250 100k dumps finished"
