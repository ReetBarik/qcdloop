#!/bin/sh

module use /soft/modulefiles
module load gcc/12.2.0
module load cmake/3.28.3

# For NVIDIA
# module load cuda/12.3.0

# For AMD
# module load rocm/6.3.0 

# For Intel

echo "Modules loaded"