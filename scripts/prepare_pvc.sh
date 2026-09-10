#!/bin/sh
################################################################
# Environment for the Intel PVC (SYCL, INTEL_PVC) build of QCDLoop + Kokkos.
#
# Usage: source scripts/prepare_pvc.sh
#        source scripts/build_QCDLoops_Kokkos_pvc.sh <repo-root>
################################################################

module use /soft/modulefiles
module load gcc/13.3.0
module load cmake/3.28.3
module load frameworks
module list
