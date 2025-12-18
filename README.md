
To build
--------

export CUTENSOR_ROOT = /path/to/cutensor
export LAPACK_ROOT = /path/to/lapack/lib64

mkdir build; cd build; cmake ..; make

This is where these are on my system
#export CUTENSOR_ROOT=/home/jaina/software/libcutensor-linux-x86_64-2.3.1.0_cuda12-archive
#export LAPACK_ROOT=/home/jaina/tools/lapack/lib64/



To Execute:
-----------

On cpus

./tensor_product_volume --cpu

On gpus

./tensor_product_volume --gpu

Use openmp [performance is poor for kernels]

./tensor_product_volume --openmp

Some Documentation:
------------------

https://jsitaraman.substack.com/p/gpucpu-implementation-of-tensor-products

