#ifndef OPERATOR_PW_VEFF_H_
#define OPERATOR_PW_VEFF_H_

#include "operator_pw.h"
#include "module_base/matrix.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_hamilt_pw/hamilt_pwdft/kernels/veff_op.h"

#include <module_base/macros.h>

namespace hamilt {

#ifndef VEFFTEMPLATE
#define VEFFTEMPLATE

template<class T> class Veff : public T {};
// template<typename Real, typename Device = psi::DEVICE_CPU>
// class Veff : public OperatorPW<T, Device> {};

#endif // VEFFTEMPLATE

template<typename T, typename Device>
class Veff<OperatorPW<T, Device>> : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;
  public:
    Veff(const ct::Tensor& isk, const ct::Tensor& veff, const ModulePW::PW_Basis_K* wfcpw_in);

    void act(const ct::Tensor& psi, ct::Tensor& hpsi)const override;

  private:

    Device* ctx_ = {};
    ct::Tensor isk_ = {}, veff_ = {};
    ct::Tensor porter_ = {}, porter1_ = {};

    const ModulePW::PW_Basis_K* wfcpw_ = nullptr;

};

} // namespace hamilt

#endif // OPERATOR_PW_VEFF_H_