################################################################
# QCDLoop + Kokkos build — Intel PVC (SYCL)
#
# Usage: source scripts/prepare_pvc.sh
#        source scripts/build_QCDLoops_Kokkos_pvc.sh <repo-root>
#
# <repo-root> must contain QCDLoop's CMakeLists.txt (the build is
# configured against it). Defaults to $PWD. Kokkos is cloned into
# <repo-root>/kokkos; its build/install trees and the QCDLoop build
# dir are all suffixed with "pvc" so several architectures can share
# one checkout without colliding.
#
# No SYCL-specific extras were ever defined; add them to EXTRA_FLAGS if needed.
################################################################

ARCH_TAG=pvc

export TARGET_DIR=$1
if [ "$#" -ne 1 ]; then
   export TARGET_DIR=$(pwd -LP)
fi
START_DIR=$(pwd -LP)
echo "Installing in path: $TARGET_DIR  (arch: $ARCH_TAG)"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || return 1

export LOGDIR=$TARGET_DIR/build_logs
mkdir -p "$LOGDIR"

##############
## SETTINGS ##
##############

# Compiler
CC=$(which icx)
CXX=$(which icpx)
if [ -z "$CC" ] || [ -z "$CXX" ]; then
   echo "ERROR: icx/icpx not on PATH — did you source scripts/prepare_pvc.sh?" >&2
   return 1
fi

# MPI Related settings
MPI_ENABLED=0
MPI_CC=NONE
MPI_CXX=NONE
if which mpicxx > /dev/null 2>&1; then
   echo Enabling MPI
   MPI_ENABLED=1
   MPI_CC=$(which mpicc)
   MPI_CXX=$(which mpicxx)
fi

# KOKKOS Related settings
KOKKOS_TAG=4.7.01
KOKKOS_BUILD=Release
KOKKOS_URL=https://github.com/kokkos/kokkos.git
KOKKOS_ENABLED=Kokkos_ENABLE_SYCL
KOKKOS_ARCH_FLAG=Kokkos_ARCH_INTEL_PVC

# Backend-specific extras. Kept as an array so values containing spaces or
# "=" survive; the old single-string form word-split incorrectly.
EXTRA_FLAGS=()

KOKKOS_SUBDIR=kokkos-$KOKKOS_TAG-$ARCH_TAG/$KOKKOS_BUILD

####################
## install Kokkos ##
####################
echo "Installing Kokkos BACKEND=$KOKKOS_ENABLED ARCH=$KOKKOS_ARCH_FLAG"

KOKKOS_CMAKE_ARGS=(
   -DCMAKE_INSTALL_PREFIX="install/$KOKKOS_SUBDIR"
   -DCMAKE_BUILD_TYPE="$KOKKOS_BUILD"
   -DCMAKE_CXX_STANDARD=17
   -D$KOKKOS_ENABLED=ON
)
if [ "$KOKKOS_ARCH_FLAG" != "NONE" ]; then
   KOKKOS_CMAKE_ARGS+=( -D$KOKKOS_ARCH_FLAG=ON )
fi
KOKKOS_CMAKE_ARGS+=( "${EXTRA_FLAGS[@]}" )

{
   { [ -d kokkos ] || git clone "$KOKKOS_URL" -b "$KOKKOS_TAG"; } &&
   cd kokkos &&
   cmake -S . -B "build/$KOKKOS_SUBDIR" "${KOKKOS_CMAKE_ARGS[@]}" &&
   make -C "build/$KOKKOS_SUBDIR" -j install
} 1> "$LOGDIR/kokkos.$ARCH_TAG.stdout.txt" 2> "$LOGDIR/kokkos.$ARCH_TAG.stderr.txt"
KOKKOS_RC=$?
if [ $KOKKOS_RC -ne 0 ]; then
   echo "ERROR: Kokkos build failed (exit $KOKKOS_RC)." >&2
   echo "       see $LOGDIR/kokkos.$ARCH_TAG.stderr.txt" >&2
   cd "$START_DIR"
   return $KOKKOS_RC
fi

cd "$TARGET_DIR/kokkos" || return 1
KOKKOS_HOME=$PWD/install/$KOKKOS_SUBDIR
{
   echo "export KOKKOS_HOME=$KOKKOS_HOME"
   echo 'export CMAKE_PREFIX_PATH=$KOKKOS_HOME/lib64/cmake/Kokkos:$CMAKE_PREFIX_PATH'
   echo 'export CPATH=$KOKKOS_HOME/include:$CPATH'
   echo 'export PATH=$KOKKOS_HOME/bin:$PATH'
   echo 'export LD_LIBRARY_PATH=$KOKKOS_HOME/lib64:$LD_LIBRARY_PATH'
} > setup.$ARCH_TAG.sh
source setup.$ARCH_TAG.sh

######################
## install QCDLoops ##
######################
cd "$TARGET_DIR" || return 1
ulimit -s 131072
export LD_LIBRARY_PATH=$TARGET_DIR/build_$ARCH_TAG:$LD_LIBRARY_PATH
mkdir -p "build_$ARCH_TAG"
cd "build_$ARCH_TAG" || return 1
cmake -DCMAKE_INSTALL_PREFIX="$TARGET_DIR" \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_CXX_FLAGS="-g" \
      .. &&
make -j32
QCDLOOP_RC=$?
cd "$TARGET_DIR"
if [ $QCDLOOP_RC -ne 0 ]; then
   echo "ERROR: QCDLoop build failed (exit $QCDLOOP_RC)." >&2
   return $QCDLOOP_RC
fi
echo "Done. Binaries in $TARGET_DIR/build_$ARCH_TAG"
