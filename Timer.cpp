#include "Timer.hpp"

using namespace std::chrono;
using namespace std;

namespace dgx3d {

Timer::Timer() {
  counter = 0.0;
#ifdef DGX3D_HAS_CUDA
  cudaEventCreate(&cuda_t0);
  cudaEventCreate(&cuda_t1);
#endif
}

void Timer::tick() {
#ifdef DGX3D_HAS_CUDA
  cudaEventRecord(cuda_t0, 0);
  cudaEventSynchronize(cuda_t0);
#else
  this->t0 = high_resolution_clock::now();
#endif
}

double Timer::tock() {
#ifdef DGX3D_HAS_CUDA
  cudaEventRecord(cuda_t1, 0);
  cudaEventSynchronize(cuda_t1);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, cuda_t0, cuda_t1);
  elapsedTime *= 0.001; // seconds from ms
  counter += (double)elapsedTime;
  return (double)elapsedTime; // want in seconds
#else
  double recent = duration<double>(high_resolution_clock::now() - t0).count();
  counter += recent;
  return recent;
#endif
}

string Timer::timestring() {
  std::time_t now = system_clock::to_time_t(std::chrono::system_clock::now());
  std::string s(30, '\0');
  std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  return s;
}

double Timer::elapsed() { return counter; }

} // namespace dgx3d
