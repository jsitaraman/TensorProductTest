// tensor_product_test.cpp
#include "Timer.hpp"
#include "cutensorProduct.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <occa.hpp>
#include <random>
#include <vector>
#define TIMER 1
#define NTRYS 10
#define COMPARE 0

using namespace dgx3d;
#include "TensorProductVolumeOccaKernels.hpp"
#include "TensorProductVolumeOrig.hpp"
#include "cutensorWrap.hpp"
#include "gemmWrap.hpp"
// =====================================================================
// Utility: fill array with reproducible random numbers
// =====================================================================
void fill_random(double *arr, size_t n, unsigned seed = 12345u) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < n; ++i)
    arr[i] = dist(rng);
}
// =====================================================================
// Validation / comparison test
// =====================================================================
void maxAbsDiff(const std::vector<double> &a, const std::vector<double> &b,
                std::string first, std::string second) {
  size_t idx_bad = (size_t)-1;
  double max_abs_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::abs(b[i] - a[i]);
    if (d > max_abs_diff) {
      max_abs_diff = d;
      idx_bad = i;
    }
  }
  const double tol = 1e-12;
  if (max_abs_diff <= tol) {
    std::cout << "✅ PASS: outputs between " << first << "  and " << second
              << " match within tol=" << tol << "\n\n";
  } else {
    std::cout << "❌ FAIL: max abs diff = " << max_abs_diff << " at index "
              << idx_bad << "\n";
    std::cout << "a[" << idx_bad << "] = " << a[idx_bad] << "  b[" << idx_bad
              << "] = " << b[idx_bad] << "\n";
  }
}

int main(int argc, char **argv) {
  occa::device device;
  bool use_gpu = false;
  // take in some arguments
  if (argc >= 2) {
    if (strstr(argv[1], "--gpu")) {
      device.setup({{"mode", "CUDA"}, {"device_id", 0}});
      occa::setDevice(device);
      use_gpu = true;
    } else if (strstr(argv[1], "--openmp")) {
      device.setup({{"mode", "OpenMP"}});
    } else {
      device.setup({{"mode", "Serial"}});
    }
  } else {
    printf("Example usage : ./tensor_product_test --gpu 8 4 \n");
    printf("--gpu, --openmp and --serial are supported and (M,K)=(8,4) in example above\n");
    exit(0);
  }
  std::cout << "-------     Device used -------- \n";
  std::cout << device.properties() << std::endl;
  std::cout << "---------------------------------\n";
  // small test sizes (easy to debug); adjust as needed
  const int M = std::stoi(argv[2]);
  const int MPad = M;
  const int K = std::stoi(argv[3]);
  const int N = 8192 * 8;

  const int LDB = K * K * K; // per-slice stride in elements (caller semantics)
  const int LDC = MPad * MPad * MPad;

  // allocate arrays: shapes:
  // Ar,As,At : [K][MPad] stored as kk*MPad + i
  std::vector<double> Ar((size_t)K * MPad);
  std::vector<double> As((size_t)K * MPad);
  std::vector<double> At((size_t)K * MPad);

  // B: [K][K][K][N] per-slice layout b[indB] where indB = K*(j + kk*K) + i (per
  // original)
  std::vector<double> B((size_t)N * LDB);

  // C buffers for original and three-stage
  std::vector<double> C_orig((size_t)N * LDC);
  std::vector<double> C_three((size_t)N * LDC);
  std::vector<double> C_occa0((size_t)N * LDC);
  std::vector<double> C_occa((size_t)N * LDC);
  std::vector<double> C_gemm((size_t)N * LDC);
  std::vector<double> C_cute((size_t)N * LDC);

  // fill with random data
  fill_random(Ar.data(), Ar.size(), 101);
  fill_random(As.data(), As.size(), 102);
  fill_random(At.data(), At.size(), 103);
  fill_random(B.data(), B.size(), 201);

  // init C
  std::fill(C_orig.begin(), C_orig.end(), 0.0);
  std::fill(C_three.begin(), C_three.end(), 0.0);

  bool addToC = false;

  printf("Begin calculations, M,K,N = (%d, %d, %d)\n\n", M, K, N);
  ///////////////////////////////////////////////////////////
  // Run GEMM by constructing the full matrix
  // as tensor product Ar x As x At
  ///////////////////////////////////////////////////////////
  std::vector<double> T;
  BuildTensorProductOperator(M, K, Ar.data(), As.data(), At.data(), T);

  int M3 = M * M * M;
  int K3 = K * K * K;
  gemmWrap(device, M3, N, K3, M3, LDB, LDC, T, B, C_gemm, 1.0, 0.0, use_gpu);
  printf("\n");
  ///////////////////////////////////////////////////////////
  // run original c++
  ///////////////////////////////////////////////////////////
  TensorProductVolume(M, MPad, N, K, LDB, LDC, Ar.data(), As.data(), At.data(),
                      B.data(), C_orig.data(), addToC);

  maxAbsDiff(C_orig, C_gemm, "original", "gemm");

  ///////////////////////////////////////////////////////////
  // run occa implementation
  ///////////////////////////////////////////////////////////

  TensorProductVolumeOCCA(device, M, MPad, N, K, LDB, LDC, Ar, As, At, B,
                                C_occa0, addToC);
  maxAbsDiff(C_gemm, C_occa0, "original", "occa");


  ///////////////////////////////////////////////////////////
  // run occa implementation
  ///////////////////////////////////////////////////////////
#if 0
  TensorProductVolumeSplit_OCCA(device, M, MPad, N, K, LDB, LDC, Ar, As, At, B,
                                C_occa, addToC);
  maxAbsDiff(C_gemm, C_occa, "original", "occa");
#endif

  // TODO does not work yet
  if (use_gpu) {
    ///////////////////////////////////////////////////////////
    // run cutensor implementation
    ///////////////////////////////////////////////////////////
    cutensorWrap(device, M, N, K, Ar, As, At, B, C_cute);

    maxAbsDiff(C_gemm, C_cute, "original", "cutensor");
  }

  // compare
#if COMPARE
  // print small slices for visual check, uncomment and recompile if fails above
  std::cout << "Sample outputs (first slice l=0, first 20 elements):\n";
  for (int i = 0; i < std::min<size_t>(20, C_orig.size()); ++i) {
    std::cout << i << ": orig=" << C_orig[i] << "  three=" << C_three[i]
              << "\n";
  }
#endif
  return 0;
}
