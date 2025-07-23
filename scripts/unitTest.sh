#!/bin/sh

./build/tadpoleGPU_test 1 1 > scripts/validate.txt
echo >> scripts/validate.txt
./build/bubbleGPU_test 1 1 >> scripts/validate.txt
echo >> scripts/validate.txt
./build/triangleGPU_test 1 1 >> scripts/validate.txt
echo >> scripts/validate.txt
./build/boxGPU_test 1 1 >> scripts/validate.txt
