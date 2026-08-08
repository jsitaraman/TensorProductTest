#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>
#ifdef DGX3D_HAS_CUDA
#include <cutensor.h>
#endif

namespace dgx3d {
// this does not compile yet
#ifdef DGX3D_HAS_CUDA

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      printf("Error: %s\n", cutensorGetErrorString(err));                      \
      exit(-1);                                                                \
    }                                                                          \
  };

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("Error: %s\n", cudaGetErrorString(err));                          \
      exit(-1);                                                                \
    }                                                                          \
  };

class cutensorProduct {
private:
  void *work_, *work2_;
  void *E_d_; // Intermediate tensor
  cutensorPlan_t plan_, plan2_;
  cutensorHandle_t handle_;
  cutensorTensorDescriptor_t descAt_, descAr_, descAs_, descB_, descE_, descD_;
  cutensorOperationDescriptor_t desc_, desc2_;
  uint64_t actualWorkspaceSize_, actualWorkspaceSize2_;
  const cutensorComputeDescriptor_t descCompute_ = CUTENSOR_COMPUTE_DESC_64F;
  cutensorPlanPreference_t planPref_, planPref_2;
  //
  // TODO: I need to add a "_" to these variables
  // For now trying to make this work
  std::vector<int64_t> extentB, extentE, extentD, extentAr, extentAs, extentAt;
  std::unordered_map<int, int64_t> extent;
  std::vector<int> modeE;
  std::vector<int> modeAr;
  std::vector<int> modeAs;
  std::vector<int> modeAt;
  std::vector<int> modeB;
  std::vector<int> modeD;
  cudaStream_t stream;

public:
  cutensorProduct(int M, int K, int N) {
    const uint32_t kAlignment = 128;

    cutensorDataType_t typeAr = CUTENSOR_R_64F;
    cutensorDataType_t typeAs = CUTENSOR_R_64F;
    cutensorDataType_t typeAt = CUTENSOR_R_64F;
    cutensorDataType_t typeB = CUTENSOR_R_64F;
    cutensorDataType_t typeE = CUTENSOR_R_64F;
    cutensorDataType_t typeD = CUTENSOR_R_64F;

    // Step 1: Trinary outer product
    // Step 2: Contraction E * B -> D
    // D_{m,s,t,n} = E[s,q,m,k,t,r] * B[k,q,r,n]  (contracts k, q, r)
    // ----------------------
    modeE = {'s', 'q', 'm', 'k', 't', 'r'};
    modeAr = {'m', 'k'};
    modeAs = {'s', 'q'};
    modeAt = {'t', 'r'};
    modeB = {'k', 'q', 'r', 'n'};
    modeD = {'m', 's', 't', 'n'};

    int nmodeAr = modeAr.size();
    int nmodeAs = modeAs.size();
    int nmodeAt = modeAt.size();
    int nmodeB = modeB.size();
    int nmodeE = modeE.size();
    int nmodeD = modeD.size();

    extent['m'] = M;
    extent['s'] = M;
    extent['t'] = M;
    extent['n'] = N;
    extent['k'] = K;
    extent['q'] = K;
    extent['r'] = K;

    for (auto mode : modeE)
      extentE.push_back(extent[mode]);
    for (auto mode : modeD)
      extentD.push_back(extent[mode]);
    for (auto mode : modeAr)
      extentAr.push_back(extent[mode]);
    for (auto mode : modeAs)
      extentAs.push_back(extent[mode]);
    for (auto mode : modeAt)
      extentAt.push_back(extent[mode]);
    for (auto mode : modeB)
      extentB.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsAr = 1;
    for (auto mode : modeAr)
      elementsAr *= extent[mode];
    size_t elementsAs = 1;
    for (auto mode : modeAs)
      elementsAs *= extent[mode];
    size_t elementsAt = 1;
    for (auto mode : modeAt)
      elementsAt *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
      elementsB *= extent[mode];
    size_t elementsE = 1;
    for (auto mode : modeE)
      elementsE *= extent[mode];
    size_t elementsD = 1;
    for (auto mode : modeD)
      elementsD *= extent[mode];

    HANDLE_CUDA_ERROR(cudaMalloc(&E_d_, elementsE * sizeof(double)));

    /*************************
     * cuTENSOR
     *************************/

    HANDLE_ERROR(cutensorCreate(&handle_));

    /**********************
     * Create Tensor Descriptors
     **********************/

    HANDLE_ERROR(cutensorCreateTensorDescriptor(
        handle_, &descAt_, nmodeAt, extentAt.data(), NULL, /*stride*/
        typeAt, kAlignment));

    HANDLE_ERROR(cutensorCreateTensorDescriptor(
        handle_, &descAr_, nmodeAr, extentAr.data(), NULL, /*stride*/
        typeAr, kAlignment));

    HANDLE_ERROR(cutensorCreateTensorDescriptor(
        handle_, &descAs_, nmodeAs, extentAs.data(), NULL, /*stride*/
        typeAs, kAlignment));

    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle_, &descB_, nmodeB,
                                                extentB.data(), NULL, /*stride*/
                                                typeB, kAlignment));

    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle_, &descE_, nmodeE,
                                                extentE.data(), NULL, /*stride*/
                                                typeE, kAlignment));

    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle_, &descD_, nmodeD,
                                                extentD.data(), NULL, /*stride*/
                                                typeD, kAlignment));

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    // Step 1: As ⊗ Ar ⊗ At -> E 
    HANDLE_ERROR(cutensorCreateContractionTrinary(
        handle_, &desc_, descAs_, modeAs.data(),
        /* unary operator A*/ CUTENSOR_OP_IDENTITY, descAr_, modeAr.data(),
        /* unary operator B*/ CUTENSOR_OP_IDENTITY, descAt_, modeAt.data(),
        /* unary operator C*/ CUTENSOR_OP_IDENTITY, descE_, modeE.data(),
        /* unary operator D*/ CUTENSOR_OP_IDENTITY, descE_, modeE.data(),
        descCompute_));

    // Step 2: E * B -> D 
    HANDLE_ERROR(cutensorCreateContraction(
        handle_, &desc2_, descE_, modeE.data(),
        /* unary operator A*/ CUTENSOR_OP_IDENTITY, descB_, modeB.data(),
        /* unary operator C*/ CUTENSOR_OP_IDENTITY, descD_, modeD.data(),
        /* unary operator D*/ CUTENSOR_OP_IDENTITY, descD_, modeD.data(),
        descCompute_));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(
        handle_, desc_, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
        (void *)&scalarType, sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_64F);

    /**************************
     * Set the algorithm to use
     ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    HANDLE_ERROR(cutensorCreatePlanPreference(handle_, &planPref_, algo,
                                              CUTENSOR_JIT_MODE_DEFAULT));
    // CUTENSOR_JIT_MODE_NONE));

    HANDLE_ERROR(cutensorCreatePlanPreference(handle_, &planPref_2, algo,
                                              CUTENSOR_JIT_MODE_DEFAULT));
    // CUTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref =
        CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(
        handle_, desc_, planPref_, workspacePref, &workspaceSizeEstimate));

    uint64_t workspaceSizeEstimate2 = 0;
    const cutensorWorksizePreference_t workspacePref2 =
        CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(
        handle_, desc2_, planPref_2, workspacePref2, &workspaceSizeEstimate2));

    /**************************
     * Create Contraction Plan
     **************************/

    HANDLE_ERROR(cutensorCreatePlan(handle_, &plan_, desc_, planPref_,
                                    workspaceSizeEstimate));

    HANDLE_ERROR(cutensorCreatePlan(handle_, &plan2_, desc2_, planPref_2,
                                    workspaceSizeEstimate2));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    actualWorkspaceSize_ = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(
        handle_, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize_,
        sizeof(actualWorkspaceSize_)));

    // At this point the user knows exactly how much memory is need by the
    // operation and only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    if (actualWorkspaceSize_ > 0) {
      HANDLE_CUDA_ERROR(cudaMalloc(&work_, actualWorkspaceSize_));
      assert(uintptr_t(work_) % 128 ==
             0); // workspace must be aligned to 128 byte-boundary
    }
    // query actually used workspace
    actualWorkspaceSize2_ = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(
        handle_, plan2_, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize_, sizeof(actualWorkspaceSize_)));

    // At this point the user knows exactly how much memory is need by the
    // operation and only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize2 <= workspaceSizeEstimate2);

    if (actualWorkspaceSize2_ > 0) {
      HANDLE_CUDA_ERROR(cudaMalloc(&work2_, actualWorkspaceSize2_));
      assert(uintptr_t(work2_) % 128 ==
             0); // workspace must be aligned to 128 byte-boundary
    }
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
  }
  void contract(const double *Ar_d, const double *As_d, const double *At_d,
                const double *B_d, double *D_d) {
    [[maybe_unused]] const uint32_t kAlignment = 128;

    assert(uintptr_t(Ar_d) % kAlignment == 0);
    assert(uintptr_t(As_d) % kAlignment == 0);
    assert(uintptr_t(At_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(D_d) % kAlignment == 0);

    typedef double floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta = (floatTypeCompute)0.f;
    // Step 1: As ⊗ Ar ⊗ At -> E (outer product)
    HANDLE_ERROR(cutensorContractTrinary(handle_, plan_, (void *)&alpha, As_d,
                                         Ar_d, At_d, (void *)&beta, E_d_, E_d_,
                                         work_, actualWorkspaceSize_, stream));
    // Step 2: E * B -> D (contracts k, q, r)
    HANDLE_ERROR(cutensorContract(handle_, plan2_, (void *)&alpha, E_d_, B_d,
                                  (void *)&beta, D_d, D_d, work2_,
                                  actualWorkspaceSize2_, stream));
  };
  ~cutensorProduct() {
    /*************************/
    HANDLE_ERROR(cutensorDestroy(handle_));
    HANDLE_ERROR(cutensorDestroyPlan(plan_));
    HANDLE_ERROR(cutensorDestroyPlan(plan2_));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc_));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc2_));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descAr_));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descAs_));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descAt_));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB_));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descE_));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descD_));
    if (work_)
      cudaFree(work_);
    if (work2_)
      cudaFree(work2_);
    if (E_d_)
      cudaFree(E_d_);
  };
};
#endif
} // namespace dgx3d
