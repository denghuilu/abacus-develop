#include "ekinetic_pw.h"

#include "module_base/timer.h"
#include "module_base/tool_quit.h"
#include "module_psi/kernels/device.h"


namespace hamilt {

template<typename T, typename Device>
Ekinetic<OperatorPW<T, Device>>::Ekinetic(
    Real tpiba2_in,
    const Real* gk2_in,
    const int gk2_row,
    const int gk2_col)
{
  this->classname = "Ekinetic";
  this->cal_type = pw_ekinetic;
  this->tpiba2 = tpiba2_in;
  this->gk2 = gk2_in;
  this->gk2_row = gk2_row;
  this->gk2_col = gk2_col;
  this->device = psi::device::get_device_type<Device>(this->ctx);
  if( this->tpiba2 < 1e-10 || this->gk2 == nullptr) {
      ModuleBase::WARNING_QUIT("EkineticPW", "Constuctor of Operator::EkineticPW is failed, please check your code!");
  }
}

template<typename T, typename Device>
Ekinetic<OperatorPW<T, Device>>::~Ekinetic() = default;

template<typename T, typename Device>
void Ekinetic<OperatorPW<T, Device>>::act(
    const int64_t nbands,
    const int64_t nbasis,
    const int npol,
    const ct::Tensor& psi_in,
    ct::Tensor& hpsi,
    const int ngk_ik) const
{
    ModuleBase::timer::tick("Operator", "EkineticPW");

    auto max_npw = nbasis / npol;
    const Real *gk2_ik = &(this->gk2[this->ik * this->gk2_col]);
    ekinetic_op()(this->ctx, nbands, ngk_ik, max_npw, tpiba2, gk2_ik, hpsi.data<T>(), psi_in.data<T>());

    ModuleBase::timer::tick("Operator", "EkineticPW");
}

template class Ekinetic<OperatorPW<std::complex<float>, psi::DEVICE_CPU>>;
template class Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_CPU>>;
// template Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_CPU>>::Ekinetic(const Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_CPU>> *ekinetic);
#if ((defined __CUDA) || (defined __ROCM))
template class Ekinetic<OperatorPW<std::complex<float>, psi::DEVICE_GPU>>;
template class Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_GPU>>;
// template Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_CPU>>::Ekinetic(const Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_GPU>> *ekinetic);
// template Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_GPU>>::Ekinetic(const Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_CPU>> *ekinetic);
// template Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_GPU>>::Ekinetic(const Ekinetic<OperatorPW<std::complex<double>, psi::DEVICE_GPU>> *ekinetic);
#endif
} // namespace hamilt