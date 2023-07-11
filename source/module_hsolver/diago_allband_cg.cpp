#include "module_hsolver/diago_allband_cg.h"

#include "diago_iter_assist.h"
#include "module_base/blas_connector.h"
#include "module_base/global_function.h"
#include "module_hsolver/kernels/math_kernel_op.h"

namespace hsolver {

template<typename FPTYPE, typename Device>
DiagoAllBandCG<FPTYPE, Device>::DiagoAllBandCG(const FPTYPE* precondition_in)
{
    this->fp_type   = container::DataTypeToEnum<FPTYPE>::value;
    this->cx_type   = container::DataTypeToEnum<std::complex<FPTYPE>>::value;
    this->device_type    = container::DeviceTypeToEnum<Device>::value;

    this->one     = new container::Tensor(cx_type, container::DeviceType::CpuDevice, {1});
    this->zero    = new container::Tensor(cx_type, container::DeviceType::CpuDevice, {1});
    this->neg_one = new container::Tensor(cx_type, container::DeviceType::CpuDevice, {1});
    this->one->template data<std::complex<FPTYPE>>()[0]     = std::complex<FPTYPE>(1.0, 0.0);
    this->zero->template data<std::complex<FPTYPE>>()[0]    = std::complex<FPTYPE>(0.0, 0.0);
    this->neg_one->template data<std::complex<FPTYPE>>()[0] = std::complex<FPTYPE>(-1.0, 0.0);

    this->device  = psi::device::get_device_type<Device>(this->ctx);
    this->h_prec  = new container::TensorMap((void *) precondition_in, fp_type, device_type, {this->n_basis_max});
}

template<typename FPTYPE, typename Device>
DiagoAllBandCG<FPTYPE, Device>::~DiagoAllBandCG() {
    // Note, we do not need to free the h_prec and psi pointer as they are refs to the outside data
    delete this->beta;
    delete this->eigen;
    delete this->err_st;

    delete this->hsub;

    delete this->hpsi;
    delete this->work;

    delete this->grad;
    delete this->hgrad;
    delete this->grad_old;

    delete this->prec;
    delete this->h_prec;

    delete this->one;
    delete this->zero;
    delete this->neg_one;
    delete this->grad_wrapper;
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::init_iter(const psi::Psi<std::complex<FPTYPE>, Device> &psi_in) {
    // Specify the problem size n_basis, n_band, while lda is n_basis_max
    this->n_band        = psi_in.get_nbands();
    this->n_basis_max   = psi_in.get_nbasis();
    this->n_basis       = psi_in.get_current_nbas();

    // All column major tensors

    this->beta          = new container::Tensor(fp_type, device_type, {this->n_band});
    this->eigen         = new container::Tensor(fp_type, device_type, {this->n_band});
    this->err_st        = new container::Tensor(fp_type, device_type, {this->n_band});

    this->hsub          = new container::Tensor(cx_type, device_type, {this->n_band, this->n_band});

    this->hpsi          = new container::Tensor(cx_type, device_type, {this->n_band, this->n_basis_max});
    this->work          = new container::Tensor(cx_type, device_type, {this->n_band, this->n_basis_max});
    this->hgrad         = new container::Tensor(cx_type, device_type, {this->n_band, this->n_basis_max});
    this->grad_old      = new container::Tensor(cx_type, device_type, {this->n_band, this->n_basis_max});

    this->prec          = new container::Tensor(fp_type, device_type, {this->n_basis_max});

    //TODO: Remove class Psi, using container::Tensor instead!
    this->grad_wrapper  = new psi::Psi<std::complex<FPTYPE>, Device>(1, this->n_band, this->n_basis_max, psi_in.get_ngk_pointer());
    this->grad          = new container::TensorMap(grad_wrapper->get_pointer(), cx_type, device_type, {this->n_band, this->n_basis_max});
}

template<typename FPTYPE, typename Device>
bool DiagoAllBandCG<FPTYPE, Device>::test_error(const container::Tensor * err_in, FPTYPE thr_in)
{
    const FPTYPE * _err_st = err_in->data<FPTYPE>();
    std::vector<FPTYPE> h_err_st;
    if (this->device == psi::GpuDevice) {
        h_err_st.resize(this->n_band);
        syncmem_var_d2h_op()(this->cpu_ctx, this->ctx, h_err_st.data(), _err_st, this->n_band);
        _err_st = h_err_st.data();
    }
    for (int ii = 0; ii < this->n_band; ii++) {
        if (_err_st[ii] > thr_in) {
            return true;
        }
    }
    return false;
}

// Finally, the last one!
template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::line_minimize(
        container::Tensor * grad_in,
        container::Tensor * hgrad_in,
        container::Tensor * psi_out,
        container::Tensor * hpsi_out)
{
    line_minimize_all_band_op()(this->ctx, grad_in->data<std::complex<FPTYPE>>(), hgrad_in->data<std::complex<FPTYPE>>(), psi_out->data<std::complex<FPTYPE>>(), hpsi_out->data<std::complex<FPTYPE>>(), this->n_basis, this->n_basis_max, this->n_band);
}

// Finally, the last two!
template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::orth_cholesky(container::Tensor * workspace_in, container::Tensor * psi_out, container::Tensor * hpsi_out, container::Tensor * hsub_out)
{
    // hsub_out = transc(psi_out) * psi_out
    // TODO: rename operator
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'C',
        'N',
        this->n_band,
        this->n_band,
        this->n_basis,
        this->one->template data<std::complex<FPTYPE>>(),
        psi_out->data<std::complex<FPTYPE>>(),
        this->n_basis_max,
        psi_out->data<std::complex<FPTYPE>>(),
        this->n_basis_max,
        this->zero->template data<std::complex<FPTYPE>>(),
        hsub_out->data<std::complex<FPTYPE>>(),
        this->n_band
    );

    // set hsub matrix to lower format;
    set_matrix_op()(this->ctx, 'L', hsub_out->data<std::complex<FPTYPE>>(), this->n_band);

    zpotrf_op()(this->ctx, hsub_out->data<std::complex<FPTYPE>>(), this->n_band);
    ztrtri_op()(this->ctx, hsub_out->data<std::complex<FPTYPE>>(), this->n_band);

    this->rotate_wf(hsub_out, psi_out, workspace_in);
    this->rotate_wf(hsub_out, hpsi_out, workspace_in);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_grad_all_band(
        const container::Tensor * prec_in,
        container::Tensor * err_out,
        container::Tensor * beta_out,
        container::Tensor * psi_in,
        container::Tensor * hpsi_in,
        container::Tensor * grad_out,
        container::Tensor * grad_old_out)
{
    calc_grad_all_band_op()(this->ctx, prec_in->data<FPTYPE>(), err_out->data<FPTYPE>(), beta_out->data<FPTYPE>(), psi_in->data<std::complex<FPTYPE>>(), hpsi_in->data<std::complex<FPTYPE>>(), grad_out->data<std::complex<FPTYPE>>(), grad_old_out->data<std::complex<FPTYPE>>(), this->n_basis, this->n_basis_max, this->n_band);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_prec()
{
    syncmem_var_h2d_op()(
        this->ctx,
        this->cpu_ctx,
        this->prec->template data<FPTYPE>(),
        this->h_prec->template data<FPTYPE>(),
        this->n_basis_max);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::orth_projection(
        const container::Tensor * psi_in,
        container::Tensor * hsub_in,
        container::Tensor * grad_out)
{
    // hsub_in = transc(psi_in) x grad_out
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'C',
        'N',
        this->n_band,
        this->n_band,
        this->n_basis,
        this->one->template data<std::complex<FPTYPE>>(),
        psi_in->data<std::complex<FPTYPE>>(),
        this->n_basis_max,
        grad_out->data<std::complex<FPTYPE>>(),
        this->n_basis_max,
        this->zero->template data<std::complex<FPTYPE>>(),
        hsub_in->data<std::complex<FPTYPE>>(),
        this->n_band
    );

    // workspace_in = - psi_in x hsub_in
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'N',
        'N',
        this->n_basis,
        this->n_band,
        this->n_band,
        this->neg_one->template data<std::complex<FPTYPE>>(),
        psi_in->data<std::complex<FPTYPE>>(),
        this->n_basis_max,
        hsub_in->data<std::complex<FPTYPE>>(),
        this->n_band,
        this->one->template data<std::complex<FPTYPE>>(),
        grad_out->data<std::complex<FPTYPE>>(),
        this->n_basis_max
    );

    // grad_out += workspace_in
    // mat_add_inplace_op()(this->ctx, grad_out, workspace_in, this->n_basis, this->n_band, this->n_basis_max);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::rotate_wf(
        const container::Tensor * hsub_in,
        container::Tensor * psi_out,
        container::Tensor * workspace_in)
{
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'N',
        'N',
        this->n_basis,
        this->n_band,
        this->n_band,
        this->one->template data<std::complex<FPTYPE>>(),
        psi_out->data<std::complex<FPTYPE>>(), // dmin * nstart
        this->n_basis_max,
        hsub_in->data<std::complex<FPTYPE>>(),  // nstart * n_band
        this->n_band,
        this->zero->template data<std::complex<FPTYPE>>(),
        workspace_in->data<std::complex<FPTYPE>>(),
        this->n_basis
    );

    syncmem_complex_op()(this->ctx, this->ctx, psi_out->data<std::complex<FPTYPE>>(), workspace_in->data<std::complex<FPTYPE>>(), this->n_basis * this->n_band);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_hpsi_all_band(
        hamilt::Hamilt<FPTYPE, Device> *hamilt_in,
        const psi::Psi<std::complex<FPTYPE>, Device> &psi_in,
        container::Tensor * hpsi_out)
{
    // calculate all-band hpsi
    psi::Range all_bands_range(1, psi_in.get_current_k(), 0, psi_in.get_nbands() - 1);
    hpsi_info info(&psi_in, all_bands_range, hpsi_out->data<std::complex<FPTYPE>>());
    hamilt_in->ops->hPsi(info);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::diag_hsub(
        const container::Tensor * psi_in,
        const container::Tensor * hpsi_in,
        container::Tensor * hsub_out,
        container::Tensor * eigenvalue_out)
{
    // calculate all-band hsub
    // Note: ctx is nothing but the devices used in this class (Device * ctx = nullptr;),
    // it controls the ops to use the corresponding device to calculate results
    gemm_op<FPTYPE, Device>()(
        this->ctx,
        'C',
        'N',
        this->n_band,
        this->n_band,
        this->n_basis,
        this->one->template data<std::complex<FPTYPE>>(),
        psi_in->data<std::complex<FPTYPE>>(),
        this->n_basis_max,
        hpsi_in->data<std::complex<FPTYPE>>(),
        this->n_basis_max,
        this->zero->template data<std::complex<FPTYPE>>(),
        hsub_out->data<std::complex<FPTYPE>>(),
        this->n_band
    );

     dnevd_op()(this->ctx, hsub_out->data<std::complex<FPTYPE>>(), this->n_band, eigenvalue_out->data<FPTYPE>());
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_hsub_all_band(
        hamilt::Hamilt<FPTYPE, Device> *hamilt_in,
        const psi::Psi<std::complex<FPTYPE>, Device> &psi_in,
        container::Tensor * psi_out,
        container::Tensor * hpsi_out,
        container::Tensor * hsub_out,
        container::Tensor * workspace_in,
        container::Tensor * eigenvalue_out)
{
    // Apply the H operator to psi and obtain the hpsi matrix.
    this->calc_hpsi_all_band(hamilt_in, psi_in, hpsi_out);

    // Diagonalization of the subspace matrix.
    this->diag_hsub(psi_out,hpsi_out, hsub_out, eigenvalue_out);

    // inplace matmul to get the initial guessed wavefunction psi.
    // psi_out[n_basis, n_band] = psi_out[n_basis, n_band] x hsub_out[n_band, n_band]
    // hpsi_out[n_basis, n_band] = psi_out[n_basis, n_band] x hsub_out[n_band, n_band]
    this->rotate_wf(hsub_out, psi_out, workspace_in);
    this->rotate_wf(hsub_out, hpsi_out, workspace_in);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::diag(
        hamilt::Hamilt<FPTYPE, Device> *hamilt_in,
        psi::Psi<std::complex<FPTYPE>, Device> &psi_in,
        FPTYPE *eigenvalue_in)
{
    // Get the pointer of the input psi
    this->psi = new container::TensorMap(psi_in.get_pointer(), cx_type, device_type, {this->n_band, this->n_basis_max});
    // Update the precondition array
    this->calc_prec();

    // Improving the initial guess of the wave function psi through a subspace diagonalization.
    this->calc_hsub_all_band(hamilt_in, psi_in, this->psi, this->hpsi, this->hsub, this->work, this->eigen);

    setmem_complex_op()(this->ctx, this->grad_old->template data<std::complex<FPTYPE>>(), 0, this->n_basis_max * this->n_band);
    setmem_var_op()(this->ctx, this->beta->template data<FPTYPE>(), 1E+40, this->n_band);
    int ntry = 0;
    int max_iter = hsolver::DiagoIterAssist<FPTYPE, Device>::SCF_ITER > 1 ?
                   this->nline :
                   this->nline * 8;
    do
    {
        ++ntry;
        // Be careful here ! dangerous zone!
        // 1. normalize psi
        // 2. calculate the epsilo
        // 3. calculate the gradient by hpsi - epsilo * psi
        // 4. gradient mix with the previous gradient
        // 5. Do precondition
        this->calc_grad_all_band(this->prec, this->err_st, this->beta,
                                 this->psi, this->hpsi, this->grad, this->grad_old);

        // Orthogonalize column vectors g_i in matrix grad to column vectors p_j in matrix psi
        // for all 'j less or equal to i'.
        // Note: hsub and work are only used to store intermediate variables of gemm operator.
        this->orth_projection(this->psi, this->hsub, this->grad);

        // this->grad_old = this->grad;
        syncmem_complex_op()(this->ctx, this->ctx, this->grad_old->template data<std::complex<FPTYPE>>(), this->grad->template data<std::complex<FPTYPE>>(), n_basis_max * n_band);

        // Calculate H|grad> matrix
        this->calc_hpsi_all_band(hamilt_in, this->grad_wrapper[0], this->hgrad);

        // optimize psi as well as the hpsi
        // 1. normalize grad
        // 2. calculate theta
        // 3. update psi as well as hpsi
        this->line_minimize(this->grad, this->hgrad, this->psi, this->hpsi);

        // orthogonal psi by cholesky method
        this->orth_cholesky(this->work, this->psi, this->hpsi, this->hsub);

    } while (ntry < max_iter && this->test_error(this->err_st, this->all_band_cg_thr));
    this->calc_hsub_all_band(hamilt_in, psi_in, this->psi, this->hpsi, this->hsub, this->work, this->eigen);
    syncmem_var_d2h_op()(this->cpu_ctx, this->ctx, eigenvalue_in, this->eigen->template data<FPTYPE>(), this->n_band);

    delete (container::TensorMap*)this->psi;
}

template class DiagoAllBandCG<float, psi::DEVICE_CPU>;
template class DiagoAllBandCG<double, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoAllBandCG<float, psi::DEVICE_GPU>;
template class DiagoAllBandCG<double, psi::DEVICE_GPU>;
#endif

} // namespace hsolver