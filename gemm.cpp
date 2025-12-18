/*
MIT License

Copyright (c) 2025 DGX3D developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// -*-c++-*-
#include <iostream>
#include <vector>
// TODO : fix this for portability
// for now, we will assume we have cuda
#ifdef DGX3D_HAS_CUDA
#include "cuda_runtime.h"
#include <cublas_v2.h>
#endif
extern "C" {
// this is a dgemm from standard lapack for now
// could use MKL to improve it
void dgemm_(char *, char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);
}

namespace dgx3d {
#ifdef DGX3D_HAS_CUDA
// full gemm
cublasHandle_t *handle = NULL;
#endif

void gemm(double *A, double *B, double *C, int M, int N, int K, int LDA,
          int LDB, int LDC, double alpha0, double beta0, bool use_gpu) {
  // this is so that we can put constants in input
  double alpha = alpha0;
  double beta = beta0;
#ifdef DGX3D_HAS_CUDA
  if (!handle) {
    handle = new cublasHandle_t[1];
    cublasCreate(handle);
  }
#endif

  if (use_gpu) {
#ifdef DGX3D_HAS_CUDA
    cublasDgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, // transa, transb
                M, N, K,                             // sizes
                &alpha, A, LDA, // A is m x k, leading dimension m
                B, LDB,         // B is k x n, leading dimension k
                &beta, C, LDC   // C is m x n, leading dimension m
    );
#else
    printf("Can't run on GPU \n");
#endif
  } else {
    char trans = 'N';
    dgemm_(&trans, &trans, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C,
           &LDC);
  }
}

void destroyGemmHandle() {
#ifdef DGX3D_HAS_CUDA
  cublasDestroy(handle[0]);
  delete[] handle;
#endif
}
} // namespace dgx3d
