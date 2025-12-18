// namespace dgx3d
//------------------------------------------------------------------------------
// Function, which performs a generic tensor multiplication to compute volume
// entities. The tensor product to be carried out is
// C = Ar*(As*(At*B)).
// The C tensor is a rank 4 tensor of dimension C[M][M][M][N] in column major
// order, i.e. the first index is the fastest changing. The first three
// dimensions, i.e. M*M*M, may be padded. The associated variable is LDC. B is
// also a rank 4 tensor of size B[K][K][K][N] in column major order. Also its
// first three dimensions, i.e. K*K*K, may be padded using the variable LDB. Ar,
// As and At are rank 2 tensors of dimension A[M][K] in column major order,
// where the first dimension is padded to MPad.
//
// Note that the values of M, N and K must be known at compile time in order to
// have the best possible performance.
void TensorProductVolume(const int M, const int MPad, const int N, const int K,
                         const int LDB, const int LDC, 
			 const double * __restrict__ Ar,
                         const double * __restrict__ As, 
			 const double * __restrict__ At, 
			 const double * __restrict__ B,
                         double * __restrict__ C, 
			 const bool addToC) {
#if TIMER
  Timer stopwatch;
  stopwatch.tick();
  for (int itry = 0; itry < NTRYS; itry++) {
#endif
    // Outer loop over N.
#pragma omp parallel for
    for (int l = 0; l < N; ++l) {
      // Define the variables to store the intermediate results.
      double tmpK[K * K * MPad];
      double tmpJ[M * K * MPad];
      double tmpI[M * M * MPad];

      // Set the pointers for the current data of the tensors B and C.
      auto b = B + l * LDB;
      auto c = C + l * LDC;

      // Tensor product in k-direction. This determines the data in the M points
      // in k-direction, while in i- and j-direction this is still in the
      // K-points. Note that for tmpK the fastest changing index is k to improve
      // performance.
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
          const int off = MPad * (j + i * K);
#pragma omp simd
          for (int k = 0; k < MPad; ++k)
            tmpK[off + k] = 0.0;
          for (int kk = 0; kk < K; ++kk) {
            const int offA = kk * MPad;
            const int indB = K * (j + kk * K) + i;
#pragma omp simd
            for (int k = 0; k < MPad; ++k)
              tmpK[off + k] += At[offA + k] * b[indB];
          }
        }
      }
      // Tensor product in j-direction. This determines the data in the M points
      // in j- and k-direction, while in i-direction this is still in the
      // K-points. Note that for tmpJ the fastest changing index is j to improve
      // performance.
      // Define the variables to store the intermediate results.
      for (int k = 0; k < M; ++k) {
        for (int i = 0; i < K; ++i) {
          const int off = MPad * (i + k * K);
#pragma omp simd
          for (int j = 0; j < MPad; ++j)
            tmpJ[off + j] = 0.0;
          for (int jj = 0; jj < K; ++jj) {
            const int offA = jj * MPad;
            const int indK = MPad * (jj + i * K) + k;
#pragma omp simd
            for (int j = 0; j < MPad; ++j)
              tmpJ[off + j] += As[offA + j] * tmpK[indK];
          }
        }
      }
      // Tensor product in i-direction. This determines the data in the M points
      // in all three direction and is the result of the tensor product. Note
      // that for tmpI the fastest changing index is i to improve performance.
      // Define the variables to store the intermediate results.
      for (int k = 0; k < M; ++k) {
        for (int j = 0; j < M; ++j) {
          const int off = MPad * (j + k * M);
#pragma omp simd
          for (int i = 0; i < MPad; ++i)
            tmpI[off + i] = 0.0;
          for (int ii = 0; ii < K; ++ii) {
            const int offA = ii * MPad;
            const int indJ = MPad * (ii + k * K) + j;
#pragma omp simd
            for (int i = 0; i < MPad; ++i)
              tmpI[off + i] += Ar[offA + i] * tmpJ[indJ];
          }
        }
      }
      // Either add or copy the result of tmpI into c.
      if (!addToC) {
        for (int k = 0; k < M; ++k) {
          for (int j = 0; j < M; ++j) {
            const int offC = M * (j + k * M);
            const int offI = MPad * (j + k * M);
#pragma omp simd safelen(8)
            for (int i = 0; i < M; ++i)
              c[offC + i] = tmpI[offI + i];
          }
        }
      } else {
        for (int k = 0; k < M; ++k) {
          for (int j = 0; j < M; ++j) {
            const int offC = M * (j + k * M);
            const int offI = MPad * (j + k * M);
#pragma omp simd safelen(8)
            for (int i = 0; i < M; ++i)
              c[offC + i] += tmpI[offI + i];
          }
        }
      }
    }
#if TIMER
  }
  printf("Original C++ Implementation time = %e\n", stopwatch.tock() / NTRYS);
#endif
}
