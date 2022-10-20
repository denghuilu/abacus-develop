#include "module_hamilt/include/nonlocal.h"
#include <complex>

using namespace hamilt; 

template <typename FPTYPE> 
void hamilt::nonlocal_pw_op<FPTYPE, psi::DEVICE_GPU>::operator() (
      const psi::DEVICE_GPU* dev,
      const int& l1,
      const int& l2,
      const int& l3,
      int& sum,
      int& iat,
      const int& current_spin,
      const int& nkb,
      const int& deeq_x,
      const int& deeq_y,
      const int& deeq_z,
      std::complex<FPTYPE>* ps,
      const std::complex<FPTYPE>* becp,
      const FPTYPE* deeq)
{
  
}

namespace hamilt{
template struct nonlocal_pw_op<double, psi::DEVICE_GPU>;
}