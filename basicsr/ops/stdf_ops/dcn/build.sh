#!/usr/bin/env bash

# You may need to modify the following paths before compiling.
CUDA_HOME=/usr/local/cuda-12.2
CUDNN_INCLUDE_DIR=$CUDA_HOME/include
CUDNN_LIB_DIR=$CUDA_HOME/lib64

python setup.py build_ext --inplace

if [ -d "build" ]; then
    rm -r build
fi
