##############################################################
# Usage: bash scripts/build_QCDLoops_Kokkos.sh <install-dir> #
#   Ensure you have your environment setup with proper       #
#   compilers and drivers for your target architecture.      #
##############################################################
# Disclaimer: this is meant to give an example of how to     #
#   inject Kokkos in QCDLoops, but it is not failproof.      #
##############################################################



# install in provided path or in current path
export TARGET_DIR=$1
if [ "$#" -ne 1 ]; then
   export TARGET_DIR=$(pwd -LP)
fi
START_DIR=$(pwd -LP)
echo Installing in path: $TARGET_DIR

mkdir -p $TARGET_DIR

cd "$TARGET_DIR" || exit 1

export LOGDIR=$TARGET_DIR/build_logs
mkdir -p $LOGDIR

##############
## SETTINGS ##
##############


# Compiler settings (choose one):
# A) generic GCC
CC=$(which gcc)
CXX=$(which g++)
# B) possible alternative on AMD systems:
# CC=$(which hipcc)
# CXX=$(which hipcc)
# C) possible alternative on Cray systems:
# CC=$(which cc)
# CXX=$(which CC)
# D) possible alternative on Intel systems:
# CC=$(which icx)
# CXX=$(which icpx)


# MPI Related settings
MPI_ENABLED=0
MPI_CC=NONE
MPI_CXX=NONE
if which mpicxx > /dev/null 2>&1; then
   echo Enabling MPI
   MPI_ENABLED=1
   # Choose one:
   # A) Standard MPI install
   MPI_CC=$(which mpicc)
   MPI_CXX=$(which mpicxx)
   # B) possible alternative on Cray systems:
   # MPI_CC=$(which cc)
   # MPI_CXX=$(which CC)
fi

# KOKKOS Related settings
KOKKOS_TAG=4.5.00
KOKKOS_BUILD=Release
KOKKOS_URL=https://github.com/kokkos/kokkos.git

# Enable Sofware Framework (choose one):
# A) Enable CUDA
KOKKOS_ENABLED=Kokkos_ENABLE_CUDA
# B) Enable HIP
# KOKKOS_ENABLED=Kokkos_ENABLE_HIP
# C) Enable OpenMP
# KOKKOS_ENABLED=Kokkos_ENABLE_OPENMP
# more available on Kokkos website

# Enable Architecture (choose one):
# A) NVidia H100
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_HOPPER90
# B) NVidia A100
KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80
# C) NVidia V100
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_VOLTA70
# D) AMD MI250
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_VEGA90A
# E) AMD MI100
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_VEGA908
# F) Intel Skylake
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_SKX
# more available on Kokkos website

# Some need extra flags needed for some software frameworks
NO_EXTRA_FLAGS=
CUDA_EXTRA_FLAGS="-DKokkos_ENABLE_CUDA_LAMBDA=On \
                  -DKokkos_ENABLE_CUDA_CONSTEXPR=On"
HIP_EXTRA_FLAGS="-DCMAKE_CXX_COMPILER=$CXX \
                 -DCMAKE_CXX_FLAGS=\"--gcc-toolchain=/soft/compilers/gcc/12.2.0/x86_64-suse-linux\""
# Set this to HIP_EXTRA_FLAGS or CUDA_EXTRA_FLAGS 
#   or NO_EXTRA_FLAGS depending on your build
# EXTRA_FLAGS=$NO_EXTRA_FLAGS
EXTRA_FLAGS=$CUDA_EXTRA_FLAGS





####################
## install Kokkos ##
####################
echo Installing Kokkos ARCH=$KOKKOS_ARCH_FLAG
{
   git clone $KOKKOS_URL -b $KOKKOS_TAG
   check_exit_status "kokkos git clone"

   cd kokkos
   
   cmake -S . -B build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD \
   -DCMAKE_INSTALL_PREFIX=install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD \
   -DCMAKE_BUILD_TYPE=$KOKKOS_BUILD \
   -DCMAKE_CXX_STANDARD=17 \
   -D$KOKKOS_ARCH_FLAG=ON \
   -D$KOKKOS_ENABLED=ON \
   $EXTRA_FLAGS
   check_exit_status "kokkos cmake"
   
   make -C build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD -j install
   check_exit_status "kokkos make"

   echo "export KOKKOS_HOME=$PWD/install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" > setup.sh
   echo "export CMAKE_PREFIX_PATH=\$KOKKOS_HOME/lib64/cmake/Kokkos:\$CMAKE_PREFIX_PATH" >> setup.sh
   echo "export CPATH=\$KOKKOS_HOME/include:\$CPATH" >> setup.sh
   echo "export PATH=\$KOKKOS_HOME/bin:\$PATH" >> setup.sh
   echo "export LD_LIBRARY_PATH=\$KOKKOS_HOME/lib64:\$LD_LIBRARY_PATH" >> setup.sh
   source setup.sh
} 1> $LOGDIR/kokkos.stdout.txt 2> $LOGDIR/kokkos.stderr.txt


cd "$TARGET_DIR" || exit 1


######################
## install QCDLoops ##
######################

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$1 -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX ..
make && make install