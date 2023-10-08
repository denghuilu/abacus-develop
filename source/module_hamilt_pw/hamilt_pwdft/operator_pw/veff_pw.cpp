#include "veff_pw.h"

#include "module_base/timer.h"
#include "module_base/tool_quit.h"
#include "module_psi/kernels/device.h"

namespace hamilt {

template<typename T, typename Device>
Veff<OperatorPW<T, Device>>::Veff(const int* isk_in,
                                       const Real* veff_in,
                                       const int veff_row,
                                       const int veff_col,
                                       const ModulePW::PW_Basis_K* wfcpw_in)
{
    this->classname = "Veff";
    this->cal_type = pw_veff;
    this->isk = isk_in;
    this->veff = veff_in;
    //note: "veff = nullptr" means that this core does not treat potential but still treats wf. 
    this->veff_row = veff_row;
    this->veff_col = veff_col;
    this->wfcpw = wfcpw_in;
    resmem_complex_op()(this->ctx, this->porter, this->wfcpw->nmaxgr, "Veff<PW>::porter");
    resmem_complex_op()(this->ctx, this->porter1, this->wfcpw->nmaxgr, "Veff<PW>::porter1");
    if (this->isk == nullptr || this->wfcpw == nullptr) {
        ModuleBase::WARNING_QUIT("VeffPW", "Constuctor of Operator::VeffPW is failed, please check your code!");
    }
}

template<typename T, typename Device>
Veff<OperatorPW<T, Device>>::~Veff()
{
    delmem_complex_op()(this->ctx, this->porter);
    delmem_complex_op()(this->ctx, this->porter1);
}

template<typename T, typename Device>
void Veff<OperatorPW<T, Device>>::act(
    const int64_t nbands,
    const int64_t nbasis,
    const int npol,
    const ct::Tensor* tmpsi_in,
    ct::Tensor* tmhpsi,
    const int ngk_ik)const
{
    ModuleBase::timer::tick("Operator", "VeffPW");

    int max_npw = nbasis / npol;
    const int current_spin = this->isk[this->ik];
    
    // T *porter = new T[wfcpw->nmaxgr];
    // TODO: Use a batched fft to replace the following loop
    for (int ib = 0; ib < nbands; ib += npol)
    {
        int tmhpsi_bias = 0;
        int tmpsi_in_bias = 0;
        if (npol == 1)
        {
            // wfcpw->recip2real(tmpsi_in, porter, this->ik);
            wfcpw->recip_to_real(this->ctx, tmpsi_in->data<T>() + tmpsi_in_bias, this->porter, this->ik);
            // NOTICE: when MPI threads are larger than number of Z grids
            // veff would contain nothing, and nothing should be done in real space
            // but the 3DFFT can not be skipped, it will cause hanging
            if(this->veff_col != 0)
            {
                veff_op()(this->ctx, this->veff_col, this->porter, this->veff + current_spin * this->veff_col);
                // const Real* current_veff = &(this->veff[0](current_spin, 0));
                // for (int ir = 0; ir < this->veff->nc; ++ir)
                // {
                //     porter[ir] *= current_veff[ir];
                // }
            }
            // wfcpw->real2recip(porter, tmhpsi, this->ik, true);
            wfcpw->real_to_recip(this->ctx, this->porter, tmhpsi->data<T>() + tmhpsi_bias, this->ik, true);
        }
        else
        {
            // T *porter1 = new T[wfcpw->nmaxgr];
            // fft to real space and doing things.
            wfcpw->recip_to_real(this->ctx, tmpsi_in->data<T>() + tmpsi_in_bias, this->porter, this->ik);
            wfcpw->recip_to_real(this->ctx, tmpsi_in->data<T>() + tmpsi_in_bias + max_npw, this->porter1, this->ik);
            if(this->veff_col != 0)
            {
                /// denghui added at 20221109
                const Real* current_veff[4];
                for(int is = 0; is < 4; is++) {
                    current_veff[is] = this->veff + is * this->veff_col ; // for CPU device
                }
                veff_op()(this->ctx, this->veff_col, this->porter, this->porter1, current_veff);
            }
            // (3) fft back to G space.
            wfcpw->real_to_recip(this->ctx, this->porter,  tmhpsi->data<T>() + tmhpsi_bias, this->ik, true);
            wfcpw->real_to_recip(this->ctx, this->porter1, tmhpsi->data<T>() + tmhpsi_bias + max_npw, this->ik, true);
        }
        tmhpsi += max_npw * npol;
        tmpsi_in += max_npw * npol;
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