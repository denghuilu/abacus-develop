#include "module_hamilt_pw/hamilt_pwdft/kernels/ekinetic_op.h"

namespace hamilt {

template <typename T>
struct ekinetic_pw_op<T, psi::DEVICE_CPU> {
  using Real = typename GetTypeReal<T>::type;
  void operator() (
      const int& nband,
      const int& npw,
      const Real& tpiba2,
      const Real* gk2,
      const T* psi,
      T* hpsi)
  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 4096/sizeof(FPTYPE))
#endif
    for (int ib = 0; ib < nband; ++ib) {
      for (int ig = 0; ig < npw; ++ig) {
        hpsi[ib * npw + ig] += gk2[ig] * tpiba2 * psi[ib * npw + ig];
      }
    }
  }
};

template struct ekinetic_pw_op<float, psi::DEVICE_CPU>;
template struct ekinetic_pw_op<double, psi::DEVICE_CPU>;

}  // namespace hamilt

