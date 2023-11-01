#include "module_hamilt_pw/hamilt_pwdft/kernels/ekinetic_op.h"

#include <complex>

#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace hamilt {
#define THREADS_PER_BLOCK 256

template <typename FPTYPE>
__global__ void ekinetic_pw(
    const int npw,
    const int max_npw,
    const FPTYPE tpiba2,
    const FPTYPE* gk2,
    thrust::complex<FPTYPE>* hpsi,
    const thrust::complex<FPTYPE>* psi)
{
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;
  for (int ii = thread_idx; ii < npw; ii+= blockDim.x) {
    hpsi[block_idx * max_npw + ii] 
      += gk2[ii] * tpiba2 * psi[block_idx * max_npw + ii];
  }
}

template <typename T>
struct ekinetic_pw_op<T, psi::DEVICE_GPU> {
    using Real = typename GetTypeReal<T>::type;

    void operator()(
        const int &nband,
        const int &npw,
        const Real &tpiba2,
        const Real *gk2,
        const T *psi,
        T *hpsi) {
        // denghui implement 20221019
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // ekinetic_pw<Real><<<nband, THREADS_PER_BLOCK>>>(
        //   npw, max_npw, tpiba2, // control params
        //   gk2_ik, // array of data
        //   reinterpret_cast<thrust::complex<FPTYPE>*>(tmhpsi), // array of data
        //   reinterpret_cast<const thrust::complex<FPTYPE>*>(tmpsi_in)); // array of data
    }
};

template struct ekinetic_pw_op<std::complex<float>, psi::DEVICE_GPU>;
template struct ekinetic_pw_op<std::complex<double>, psi::DEVICE_GPU>;

}  // namespace hamilt