#ifndef HSOLVER_MATH_KERNEL_H
#define HSOLVER_MATH_KERNEL_H

#include <complex.h>
#include "module_psi/psi.h"
#include "module_base/blas_connector.h"
#include "src_parallel/parallel_reduce.h"

namespace hsolver {
template<typename FPTYPE>
FPTYPE zdot_real(const int &dim, const std::complex<FPTYPE>* psi_L, const std::complex<FPTYPE>* psi_R, const psi::AbacusDevice_t device = psi::CpuDevice, const bool reduce = true);

#if __CUDA || __UT_USE_CUDA
template<typename FPTYPE>
FPTYPE zdot_real_gpu_cuda(const int &dim, const std::complex<FPTYPE>* psi_L, const std::complex<FPTYPE>* psi_R, const psi::AbacusDevice_t device = psi::GpuDevice, const bool reduce = true);
#elif __ROCM || __UT_USE_ROCM
template<typename FPTYPE>
FPTYPE zdot_real_gpu_rocm(const int &dim, const std::complex<FPTYPE>* psi_L, const std::complex<FPTYPE>* psi_R, const psi::AbacusDevice_t device = psi::GpuDevice, const bool reduce = true);
#endif // 
}

#endif 