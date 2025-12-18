#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <cstdio>
#include <ctime>
#include <stack>
#include <string>
#ifdef DGX3D_HAS_CUDA
#include "cuda_runtime.h"
#endif

namespace dgx3d {

class Timer {

#ifdef DGX3D_HAS_CUDA
  cudaEvent_t cuda_t0, cuda_t1;
#else
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;
#endif

  double counter;

public:
  Timer();
  ~Timer(){};
  void tick();
  double tock();
  std::string timestring();
  double elapsed();
};

} // namespace dgx3d

#endif // TIMER_HPP
