#include "module_hsolver/include/math_kernel.h"

#include <iomanip>
#include <iostream>

using namespace hsolver;

// CPU specialization of actual computation.
template <typename FPTYPE> 
struct zdot_real_op<FPTYPE, psi::DEVICE_CPU> {
  FPTYPE operator() (
      const psi::DEVICE_CPU* d,
      const int& dim,
      const std::complex<FPTYPE>* psi_L,
      const std::complex<FPTYPE>* psi_R,
      const bool reduce) 
  {
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // qianrui modify 2021-3-14
    // Note that  ddot_(2*dim,a,1,b,1) = REAL( zdotc_(dim,a,1,b,1) )
    const FPTYPE* pL = reinterpret_cast<const FPTYPE*>(psi_L);
    const FPTYPE* pR = reinterpret_cast<const FPTYPE*>(psi_R);
    FPTYPE result = BlasConnector::dot(2 * dim, pL, 1, pR, 1);
    if (reduce) {
      Parallel_Reduce::reduce_double_pool(result);
    }
    return result;
  }
};

namespace hsolver {
// Explicitly instantiate functors for the types of functor registered.
template struct zdot_real_op<double, psi::DEVICE_CPU>;
}