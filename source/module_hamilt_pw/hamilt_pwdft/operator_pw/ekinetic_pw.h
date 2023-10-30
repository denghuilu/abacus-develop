#ifndef OPERATOR_PW_EKINETICPW_H_
#define OPERATOR_PW_EKINETICPW_H_

#include "operator_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/kernels/ekinetic_op.h"

#include <module_base/macros.h>

namespace hamilt {

// Not needed anymore
#ifndef EKINETICTEMPLATE
#define EKINETICTEMPLATE

template<class T> class Ekinetic : public T {};
// template<typename R, typename Device = psi::DEVICE_CPU>
// class Ekinetic : public OperatorPW<T, Device> {};

#endif // EKINETICTEMPLATE

// template<typename R, typename Device = psi::DEVICE_CPU>
// class Ekinetic : public OperatorPW<T, Device>
template<typename T, typename Device>
class Ekinetic<OperatorPW<T, Device>> : public OperatorPW<T, Device>
{
  private: 
    using Real = typename GetTypeReal<T>::type;
  public:
    Ekinetic(Real tpiba2, const ct::Tensor& gk2);

    void act(const ct::Tensor& psi, ct::Tensor& hpsi) const override;

  private:
    Real tpiba2_ = {};
    ct::Tensor gk2_ = {};
};

} // namespace hamilt
#endif // OPERATOR_PW_EKINETICPW_H_