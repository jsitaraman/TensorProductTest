namespace dgx3d {
void gemm(double *A, double *B, double *C, int M, int N, int K, int LDA,
          int LDB, int LDC, double alpha0, double beta0, bool use_gpu);
void destroyGemmHandle();
} // namespace dgx3d
// Build the full 3D tensor product operator: T = Ar ⊗ As ⊗ At
// Each A* is (M x K), column-major
// Output T is (M^3 x K^3), also column-major
void BuildTensorProductOperator(int M, int K, const double *Ar,
                                const double *As, const double *At,
                                std::vector<double> &T) {
  const int MK3 = M * M * M;
  const int KK3 = K * K * K;
  T.assign(MK3 * KK3, 0.0);

  // column-major accessors
  auto A = [&](const double *A_, int m, int k, int strideM) {
    return A_[m + k * strideM];
  };

  for (int kt = 0; kt < K; ++kt)
    for (int ks = 0; ks < K; ++ks)
      for (int kr = 0; kr < K; ++kr) {
        int col = kr + ks * K + kt * K * K; // column index in T

        for (int mt = 0; mt < M; ++mt)
          for (int ms = 0; ms < M; ++ms)
            for (int mr = 0; mr < M; ++mr) {
              int row = mr + ms * M + mt * M * M; // row index in T

              double val =
                  A(Ar, mr, kr, M) * A(As, ms, ks, M) * A(At, mt, kt, M);

              T[row + col * MK3] = val;
            }
      }
}

void gemmWrap(occa::device &device, int M, int N, int K, int LDA, int LDB,
              int LDC, const std::vector<double> &A,
              const std::vector<double> &B, std::vector<double> &C,
              double alpha, double beta, bool use_gpu) {

  occa::memory d_A = device.malloc<double>(A.size(), A.data());
  occa::memory d_B = device.malloc<double>(B.size(), B.data());
  occa::memory d_C = device.malloc<double>(C.size(), C.data());

  double *Aptr = static_cast<double *>(d_A.ptr());
  double *Bptr = static_cast<double *>(d_B.ptr());
  double *Cptr = static_cast<double *>(d_C.ptr());
#if TIMER
  Timer stopwatch;
  for (int itry = 0; itry < NTRYS; itry++) {
  if (itry ==1 ) stopwatch.tick();
#endif
    gemm(Aptr, Bptr, Cptr, M, N, K, // M,N,K
         LDA, LDB, LDC, alpha, beta, use_gpu);
#if TIMER
  }
  double gemmTime = stopwatch.tock() / (NTRYS-1);
  printf("GEMM time (%s)= %e TFLOPS=%e\n", (use_gpu) ? "cublas" : "dgemm",
         gemmTime, (double)(2.0 * M * N * K) / gemmTime / 1e12);
#endif
  destroyGemmHandle();
  d_C.copyTo(C.data());
}
