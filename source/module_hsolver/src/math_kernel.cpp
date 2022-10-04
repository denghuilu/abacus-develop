#include "module_hsolver/include/math_kernel.h"

using namespace hsolver;
template<typename FPTYPE>
FPTYPE hsolver::zdot_real(
    const int &dim, 
    const std::complex<FPTYPE>* psi_L, 
    const std::complex<FPTYPE>* psi_R, 
    const psi::AbacusDevice_t device,
    const bool reduce)
{
    if (device == psi::CpuDevice) {
      //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      //qianrui modify 2021-3-14
      //Note that  ddot_(2*dim,a,1,b,1) = REAL( zdotc_(dim,a,1,b,1) )
      const FPTYPE* pL = reinterpret_cast<const FPTYPE*>(psi_L);
      const FPTYPE* pR = reinterpret_cast<const FPTYPE*>(psi_R);
      FPTYPE result = BlasConnector::dot(2 * dim, pL, 1, pR, 1);
      if(reduce)  Parallel_Reduce::reduce_double_pool( result );
      return result;
    }
    else if (device == psi::GpuDevice){
      #if __CUDA
      zdot_real_gpu_cuda(dim, psi_L, psi_R, device, reduce);
      #elif __ROCM
      zdot_real_gpu_rocm(dim, psi_L, psi_R, device, reduce);
      #endif 
    }
}

namespace hsolver {
    template double zdot_real<double>(const int &dim, const std::complex<double>* psi_L, const std::complex<double>* psi_R, const psi::AbacusDevice_t device, const bool reduce);
}