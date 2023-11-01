#include "veff_pw.h"

#include "module_base/timer.h"
#include "module_base/tool_quit.h"
#include "module_psi/kernels/device.h"

namespace hamilt {

template<typename T, typename Device>
Veff<OperatorPW<T, Device>>::Veff(
    const ct::Tensor& isk,
    const ct::Tensor& veff,
    const ModulePW::PW_Basis_K* wfcpw)
{
    this->type_ = pw_veff;
    isk_.CopyFrom(isk); // reference to the isk data
    veff_.CopyFrom(veff); // reference to the veff data
    //note: "veff = nullptr" means that this core does not treat potential but still treats wf. 
    wfcpw_ = wfcpw;
    porter_ = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {wfcpw_->nmaxgr});
    porter1_ = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {wfcpw_->nmaxgr});
    REQUIRES_OK(isk_ && wfcpw_ != nullptr,
        "VeffPW::Veff, Constuctor of Operator::VeffPW is failed, please check your code!")
}

template<typename T, typename Device>
void Veff<OperatorPW<T, Device>>::act(
    const ct::Tensor& psi,
    ct::Tensor& hpsi) const
{
    ModuleBase::timer::tick("Operator", "VeffPW");

    REQUIRES_OK(psi.shape() == hpsi.shape(),
        "VeffPW::act, shape of psi and hpsi are not equal")

    int max_npw = psi.shape().dim_size(1) / this->npol_;
    const int current_spin = isk_[this->ik_].template data<int>()[0];
    const int nband = static_cast<int>(psi.shape().dim_size(0));
    
    // T *porter = new T[wfcpw->nmaxgr];
    // TODO: Use a batched fft to replace the following loop
    for (int ib = 0; ib < nband; ib += this->npol_)
    {
        if (this->npol_ == 1)
        {
            // wfcpw->recip2real(tmpsi_in, porter, this->ik);
            wfcpw_->recip_to_real(this->ctx_, psi[ib].data<T>(), porter_.data<T>(), this->ik_);
            // NOTICE: when MPI threads are larger than number of Z grids
            // veff would contain nothing, and nothing should be done in real space
            // but the 3DFFT can not be skipped, it will cause hanging
            if (veff_) {
                veff_pw_op<T, Device>()(veff_.shape().dim_size(1), porter_.data<T>(), veff_[current_spin].data<Real>());
            }
            // wfcpw->real2recip(porter, tmhpsi, this->ik, true);
            wfcpw_->real_to_recip(this->ctx_, porter_.data<T>(), hpsi[ib].data<T>(), true);
        }
        else
        {
            // T *porter1 = new T[wfcpw->nmaxgr];
            // fft to real space and doing things.
            wfcpw_->recip_to_real(this->ctx_, psi[ib].data<T>(), porter_.data<T>(), this->ik_);
            wfcpw_->recip_to_real(this->ctx_, psi[ib][max_npw].data<T>(), porter1_.data<T>(), this->ik_);
            if (veff_) {
                /// denghui added at 20221109
                const Real* current_veff[4];
                for(int is = 0; is < 4; is++) {
                    current_veff[is] = veff_[is].data<Real>(); // for CPU device
                }
                veff_pw_op<T, Device>()(veff_.shape().dim_size(1), porter_.data<T>(), porter1_.data<T>(), current_veff);
            }
            // (3) fft back to G space.
            wfcpw_->real_to_recip(this->ctx_, porter_.data<T>(),  hpsi[ib].data<T>(), this->ik_, true);
            wfcpw_->real_to_recip(this->ctx_, porter1_.data<T>(), hpsi[ib][max_npw].data<T>(), this->ik_, true);
        }
    }
    ModuleBase::timer::tick("Operator", "VeffPW");
}

template class Veff<OperatorPW<std::complex<float>, psi::DEVICE_CPU>>;
template class Veff<OperatorPW<std::complex<double>, psi::DEVICE_CPU>>;
#if ((defined __CUDA) || (defined __ROCM))
template class Veff<OperatorPW<std::complex<float>, psi::DEVICE_GPU>>;
template class Veff<OperatorPW<std::complex<double>, psi::DEVICE_GPU>>;
#endif
} // namespace hamilt