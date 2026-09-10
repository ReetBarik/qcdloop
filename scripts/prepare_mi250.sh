#!/bin/sh
################################################################
# Environment for the AMD MI250 (HIP, VEGA90A) build of QCDLoop + Kokkos.
#
# Usage: source scripts/prepare_mi250.sh
#        source scripts/build_QCDLoops_Kokkos_mi250.sh <repo-root>
################################################################

module use /soft/modulefiles
module load gcc/13.3.0
module load cmake/3.28.3
module load rocm/7.0.2
module list
