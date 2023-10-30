#include "ekinetic_pw.h"

#include "module_base/timer.h"
#include "module_base/tool_quit.h"
#include "module_psi/kernels/device.h"


namespace hamilt {

template<typename T, typename Device>
Ekinetic<OperatorPW<T, Device>>::Ekinetic(
    Real tpiba2,
    const ct::Tensor& gk2)
{
    this->gk2_.CopyFrom(gk2);
    this->tpiba2_ = tpiba2;
    this->type_ = pw_ekinetic;
    REQUIRES_OK(tpiba2_ >= 1e-10 && this->gk2_,
        "EkineticPW::Ekinetic, Constuctor of Operator::EkineticPW is failed, please check your code!")
}

template<typename T, typename Device>
void Ekinetic<OperatorPW<T, Device>>::act(
    const ct::Tensor& psi,
    ct::Tensor& hpsi) const
{
    ModuleBase::timer::tick("Operator", "EkineticPW");
    REQUIRES_OK(psi.shape() == hpsi.shape(),
                "EkineticPW::act, shape of psi and hpsi are not equal")
    const auto *gk2 = gk2_[this->ik_].template data<Real>();
    ekinetic_pw_op<T, Device>()(
        psi.shape().dim_size(0), psi.shape().dim_size(1),
        tpiba2_, gk2, hpsi.data<T>(), psi.data<T>());

    ModuleBase::timer::tick("Operator", "EkineticPW");
}

template class Ekinetic<OperatorPW<std::complex<float>, psi::DEVICE_CPU>>;
template class Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_CPU>>;
#if ((defined __CUDA) || (defined __ROCM))
template class Ekinetic<OperatorPW<std::complex<float>, psi::DEVICE_GPU>>;
template class Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_GPU>>;
#endif
} // namespace hamilt