#!/bin/bash
# Build Kokkos 5.1 HIP (MI250, gfx90a) if needed, then build and smoke-run
# the four xpmath box backends. quadmath is compiled with host g++.
#
#   qsub -A pepper_hep -n 1 -t 180 -q gpu_amd_mi250 --mode script \
#        --cwd /home/rbarik/qcdloop --jobname xpmath_mi250 \
#        -O /home/rbarik/qcdloop/build_logs/xpmath_mi250 \
#        scripts/qsub_mi250_xpmath.sh
set -euo pipefail

REPO=/home/rbarik/qcdloop
cd "$REPO"

if ! type module >/dev/null 2>&1; then
   # shellcheck disable=SC1091
   source /etc/profile
fi

# shellcheck disable=SC1091
source scripts/prepare_mi250.sh

GCC_BIN=/soft/compilers/gcc/13.3.0/x86_64-suse-linux/bin
export PATH="${GCC_BIN}:${PATH}"
export DD_HOST_CXX="${GCC_BIN}/g++"
export LD_LIBRARY_PATH="/soft/compilers/gcc/13.3.0/x86_64-suse-linux/lib64:${LD_LIBRARY_PATH:-}"

HIPCC=$(command -v hipcc)
GCC_TOOLCHAIN=/soft/compilers/gcc/13.3.0/x86_64-suse-linux

KOKKOS_SRC="$REPO/kokkos"
KOKKOS_BUILD="$KOKKOS_SRC/build/kokkos-5.1.0-mi250/Release"
KOKKOS_HOME="$KOKKOS_SRC/install/kokkos-5.1.0-mi250/Release"
XPMATH_PREFIX="$HOME/xpmath-kokkos-install"
BUILD_DIR="$REPO/build_mi250_xp"
LOGDIR="$REPO/build_logs"
mkdir -p "$LOGDIR"

if [[ ! -f "$KOKKOS_HOME/lib64/cmake/Kokkos/KokkosConfig.cmake" ]]; then
   echo "Configuring Kokkos 5.1 HIP gfx90a in $KOKKOS_BUILD"
   cmake -S "$KOKKOS_SRC" -B "$KOKKOS_BUILD" \
      -DCMAKE_INSTALL_PREFIX="$KOKKOS_HOME" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_CXX_COMPILER="$HIPCC" \
      -DCMAKE_CXX_FLAGS="--gcc-toolchain=${GCC_TOOLCHAIN}" \
      -DKokkos_ENABLE_HIP=ON \
      -DKokkos_ARCH_VEGA90A=ON
   cmake --build "$KOKKOS_BUILD" -j16 --target install
else
   echo "Reusing $KOKKOS_HOME"
fi

export LD_LIBRARY_PATH="$KOKKOS_HOME/lib64:${LD_LIBRARY_PATH}"
ulimit -s 131072

echo "Configuring QCDLoop in $BUILD_DIR"
cmake -S "$REPO" -B "$BUILD_DIR" \
   -DCMAKE_CXX_STANDARD=17 \
   -DCMAKE_C_COMPILER="$HIPCC" \
   -DCMAKE_CXX_COMPILER="$HIPCC" \
   -DCMAKE_CXX_FLAGS="--gcc-toolchain=${GCC_TOOLCHAIN}" \
   -DCMAKE_PREFIX_PATH="$KOKKOS_HOME;$XPMATH_PREFIX"

for be in dd ff qf tf; do
   echo "Building boxGPU_test_${be}"
   cmake --build "$BUILD_DIR" --target "boxGPU_test_${be}" -j8
done

for be in dd ff qf tf; do
   echo "=== boxGPU_test_${be} 1 1 ==="
   "$BUILD_DIR/boxGPU_test_${be}" 1 1 > "$LOGDIR/mi250_${be}_smoke.txt"
   echo "boxGPU_test_${be} wrote $(wc -l < "$LOGDIR/mi250_${be}_smoke.txt") lines"
done

echo "B1 id 1 from DoubleDouble:"
grep '^B1,1,' "$LOGDIR/mi250_dd_smoke.txt" || true
echo "MI250 xpmath smoke finished"
