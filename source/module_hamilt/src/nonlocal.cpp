#include "module_hamilt/include/nonlocal.h"

#include <iomanip>
#include <iostream>

using namespace hamilt; 

template <typename FPTYPE> 
struct nonlocal_pw_op<FPTYPE, psi::DEVICE_CPU> {
  void operator() (
      const psi::DEVICE_CPU* dev,
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
    for (int ii = 0; ii < l1; ii++) {
      // each atom has nproj, means this is with structure factor;
      // each projector (each atom) must multiply coefficient
      // with all the other projectors.
      for (int jj = 0; jj < l2; ++jj) 
        for (int kk = 0; kk < l3; kk++) 
          for (int xx = 0; xx < l3; xx++) 
            ps[(sum + kk) * l2 + jj]
                += deeq[((current_spin * deeq_x + iat) * deeq_y + xx) * deeq_z + kk] 
                *  becp[jj * nkb + sum + xx];
      sum += l3;
      ++iat;
    }
  }
};

namespace hamilt{
template struct nonlocal_pw_op<double, psi::DEVICE_CPU>;
}