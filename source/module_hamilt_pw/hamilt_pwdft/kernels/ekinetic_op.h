#ifndef MODULE_HAMILT_PW_HAMILT_PWDFT_EKINETIC_H_
#define MODULE_HAMILT_PW_HAMILT_PWDFT_EKINETIC_H_

#include <complex>

#include "module_psi/psi.h"

#include <module_base/macros.h>

namespace hamilt {

template <typename T, typename Device>
struct ekinetic_pw_op {
  /// @brief Compute the ekinetic potential of hPsi
  ///
  /// Input Parameters
  /// \param nband : nbands
  /// \param npw : number of planewaves of current k point
  /// \param max_npw : max number of planewaves of all k points
  /// \param tpiba2 : GlobalC::ucell.tpiba2
  /// \param spin : current spin
  /// \param gk2_ik : GlobalC::wfcpw->gk2
  /// \param tmpsi_in : intermediate array
  ///
  /// Output Parameters
  /// \param tmhpsi : output array
  using Real = typename GetTypeReal<T>::type;
  void operator() (
      const int& nband,
      const int& npw,
      const Real& tpiba2,
      const Real* gk2,
      const T* psi,
      T* hpsi);
};
} // namespace hamilt
#endif //MODULE_HAMILT_PW_HAMILT_PWDFT_EKINETIC_H_