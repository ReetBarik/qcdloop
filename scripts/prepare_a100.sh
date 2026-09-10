#!/bin/sh
################################################################
# Environment for the NVIDIA A100 (CUDA, AMPERE80) build of QCDLoop + Kokkos.
#
# Usage: source scripts/prepare_a100.sh
#        source scripts/build_QCDLoops_Kokkos_a100.sh <repo-root>
################################################################

module use /soft/modulefiles
module load gcc/13.3.0
module load cmake/3.28.3
module load cuda/12.9.1
module list
