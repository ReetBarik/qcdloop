#!/bin/sh
################################################################
# Environment for the CPU (Kokkos Serial, no target arch) build of QCDLoop + Kokkos.
#
# Usage: source scripts/prepare_cpu.sh
#        source scripts/build_QCDLoops_Kokkos_cpu.sh <repo-root>
################################################################

module use /soft/modulefiles
module load gcc/13.3.0
module load cmake/3.28.3

module list
