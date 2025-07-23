# Compiler settings (choose one):
# A) generic GCC
# CC=$(which gcc)
# CXX=$(which g++)
# B) possible alternative on AMD systems:
CC=$(which hipcc)
CXX=$(which hipcc)
# C) possible alternative on Cray systems:
# CC=$(which cc)
# CXX=$(which CC)
# D) possible alternative on Intel systems:
# CC=$(which icx)
# CXX=$(which icpx)
#
#
export LD_LIBRARY_PATH=$1/build/:$LD_LIBRARY_PATH
cd build
cmake -DCMAKE_INSTALL_PREFIX=$1 -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CXX_FLAGS="-g" ..
make VERBOSE=1
# make 
cd ..

######################
## only for AMD GPU ##
######################
module unload gcc/12.1.0
module load gcc/13.3.0