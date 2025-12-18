cmake_minimum_required(VERSION 3.16)
project(TensorProductVolumeOCCA LANGUAGES CXX)

find_package(CUDAToolkit QUIET)
# Check if CUDA was found
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    set(USE_CUDA TRUE)
else()
    message(WARNING "CUDA compiler not found. Building without CUDA support.")
    set(USE_CUDA FALSE)
endif()

find_package(OpenMP REQUIRED)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ENABLE_EXPORTS TRUE)
set(CMAKE_BUILD_TYPE Release)

# --- Find OCCA ---
# If OCCA is not installed system-wide, you can point to it:
# cmake -DOCCA_DIR=/path/to/occa ..
find_package(OCCA REQUIRED)

if (USE_CUDA) 
 # --- cuTENSOR setup ---
 # edit this to reflect your cutensor root
 #set(CUTENSOR_ROOT "/home/jaina/software/libcutensor-linux-x86_64-2.3.1.0_cuda12-archive")
 set(CUTENSOR_ROOT $ENV{CUTENSOR_ROOT})
 include_directories(${CUTENSOR_ROOT}/include)
 link_directories(${CUTENSOR_ROOT}/lib)
endif()

# --- Optional message for clarity ---
message(STATUS "OCCA include dirs: ${OCCA_INCLUDE_DIRS}")
message(STATUS "OCCA libraries: ${OCCA_LIBRARIES}")

set(LAPACK_DIR_INSTALL $ENV{LAPACK_ROOT})
message(STATUS "LAPACK_DIR_INSTALL ${LAPACK_DIR_INSTALL}")
find_library(LAPACK_LIBRARY lapack PATHS "${LAPACK_DIR_INSTALL}" NO_DEFAULT_PATH)
find_library(BLAS_LIBRARY blas PATHS "${LAPACK_DIR_INSTALL}" NO_DEFAULT_PATH)
get_filename_component(DGX3D_OKL_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/okl"  ABSOLUTE)


# --- Executable ---
add_executable(tensor_product_volume main.cpp Timer.cpp gemm.cpp )
target_compile_definitions( tensor_product_volume PRIVATE DGX3D_OKL_DIR="${DGX3D_OKL_DIR}")

# --- Link Stuff ---
if (USE_CUDA) 
 target_compile_definitions( tensor_product_volume PRIVATE DGX3D_HAS_CUDA)
 target_link_libraries(tensor_product_volume PUBLIC ${LAPACKE_LIBRARY} ${LAPACK_LIBRARY} ${BLAS_LIBRARY} OCCA::libocca CUDA::cudart CUDA::cublas cutensor OpenMP::OpenMP_CXX gfortran)
else()
 target_link_libraries(tensor_product_volume PUBLIC ${LAPACK_LIBRARY} ${BLAS_LIBRARY} OCCA::libocca gfortran)
endif()

# --- Compiler warnings ---
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tensor_product_volume PRIVATE -Wall -Wextra -O3)
endif()
