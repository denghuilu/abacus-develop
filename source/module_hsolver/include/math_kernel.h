#ifndef MODULE_HSOLVER_MATH_KERNEL_H
#define MODULE_HSOLVER_MATH_KERNEL_H

#include "module_base/blas_connector.h"
#include "module_psi/psi.h"
#include "src_parallel/parallel_reduce.h"

namespace hsolver {

template <typename FPTYPE, typename Device> 
struct zdot_real_op {
  FPTYPE operator() (
      const Device* d, 
      const int& dim, 
      const std::complex<FPTYPE>* psi_L, 
      const std::complex<FPTYPE>* psi_R, 
      const bool reduce = true);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
// Partially specialize functor for psi::GpuDevice.
template <typename FPTYPE> 
struct zdot_real_op<FPTYPE, psi::DEVICE_GPU> {
  FPTYPE operator()(
      const psi::DEVICE_GPU* d, 
      const int& dim, 
      const std::complex<FPTYPE>* psi_L, 
      const std::complex<FPTYPE>* psi_R, 
      const bool reduce = true);
};
#endif // __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
} // namespace hsolver

#endif // MODULE_HSOLVER_MATH_KERNEL_H