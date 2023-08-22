#include "module_hsolver/diago_allband_cg.h"

#include "diago_iter_assist.h"
#include "module_base/blas_connector.h"
#include "module_base/global_function.h"
#include "module_hsolver/kernels/math_kernel_op.h"

#include <ATen/kernels/blas_op.h>
#include <ATen/kernels/lapack_op.h>
#include <ATen/kernels/einsum_op.h>

namespace hsolver {

template<typename FPTYPE, typename Device>
DiagoAllBandCG<FPTYPE, Device>::DiagoAllBandCG(const FPTYPE* precondition_in)
{
    this->fp_type   = ct::DataTypeToEnum<FPTYPE>::value;
    this->cx_type   = ct::DataTypeToEnum<std::complex<FPTYPE>>::value;
    this->device_type    = ct::DeviceTypeToEnum<Device>::value;

    this->device  = psi::device::get_device_type<Device>(this->ctx);
    this->h_prec  = std::move(ct::TensorMap((void *) precondition_in, fp_type, device_type, {this->n_basis}));
}

template<typename FPTYPE, typename Device>
DiagoAllBandCG<FPTYPE, Device>::~DiagoAllBandCG() {
    // Note, we do not need to free the h_prec and psi pointer as they are refs to the outside data
    delete this->grad_wrapper;
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::init_iter(const psi::Psi<std::complex<FPTYPE>, Device> &psi_in) {
    // Specify the problem size n_basis, n_band, while lda is n_basis
    this->n_band        = psi_in.get_nbands();
    this->n_basis       = psi_in.get_nbasis();

    // All column major tensors

    this->beta          = std::move(ct::Tensor(fp_type, device_type, {this->n_band}));
    this->eigen         = std::move(ct::Tensor(fp_type, device_type, {this->n_band}));
    this->err_st        = std::move(ct::Tensor(fp_type, device_type, {this->n_band}));

    this->hsub          = std::move(ct::Tensor(cx_type, device_type, {this->n_band, this->n_band}));

    this->hpsi          = std::move(ct::Tensor(cx_type, device_type, {this->n_band, this->n_basis}));
    this->work          = std::move(ct::Tensor(cx_type, device_type, {this->n_band, this->n_basis}));
    this->hgrad         = std::move(ct::Tensor(cx_type, device_type, {this->n_band, this->n_basis}));
    this->grad_old      = std::move(ct::Tensor(cx_type, device_type, {this->n_band, this->n_basis}));

    this->prec          = std::move(ct::Tensor(fp_type, device_type, {this->n_basis}));

    //TODO: Remove class Psi, using ct::Tensor instead!
    this->grad_wrapper  = new psi::Psi<std::complex<FPTYPE>, Device>(1, this->n_band, this->n_basis, psi_in.get_ngk_pointer());
    this->grad          = std::move(ct::TensorMap(grad_wrapper->get_pointer(), cx_type, device_type, {this->n_band, this->n_basis}));
}

template<typename FPTYPE, typename Device>
bool DiagoAllBandCG<FPTYPE, Device>::test_error(const ct::Tensor& err_in, FPTYPE thr_in)
{
    const FPTYPE * _err_st = err_in.data<FPTYPE>();
    if (this->device == psi::GpuDevice) {
        ct::Tensor h_err_in = err_in.to_device<ct::DEVICE_CPU>();
        _err_st = h_err_in.data<FPTYPE>();
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
    ct::Tensor& grad_in,
    ct::Tensor& hgrad_in,
    ct::Tensor& psi_out,
    ct::Tensor& hpsi_out)
{
    line_minimize_all_band_op()(this->ctx, grad_in.data<std::complex<FPTYPE>>(), hgrad_in.data<std::complex<FPTYPE>>(), psi_out.data<std::complex<FPTYPE>>(), hpsi_out.data<std::complex<FPTYPE>>(), this->n_basis, this->n_basis, this->n_band);
}

// Finally, the last two!
template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::orth_cholesky(ct::Tensor& workspace_in, ct::Tensor& psi_out, ct::Tensor& hpsi_out, ct::Tensor& hsub_out)
{
    // hsub_out = psi_out * transc(psi_out)
    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&hsub_out);
    hsub_out = ct::op::einsum("ij,kj->ik", psi_out, psi_out, option);

    // set hsub matrix to lower format;
    ct::op::set_matrix<std::complex<FPTYPE>, ct::DEVICE_GPU>()(
        'L', hsub_out.data<std::complex<FPTYPE>>(), this->n_band);

    zpotrf_op()(this->ctx, hsub_out.data<std::complex<FPTYPE>>(), this->n_band);
    ztrtri_op()(this->ctx, hsub_out.data<std::complex<FPTYPE>>(), this->n_band);

    this->rotate_wf(hsub_out, psi_out, workspace_in);
    this->rotate_wf(hsub_out, hpsi_out, workspace_in);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_grad_all_band(
        const ct::Tensor& prec_in,
        ct::Tensor& err_out,
        ct::Tensor& beta_out,
        ct::Tensor& psi_in,
        ct::Tensor& hpsi_in,
        ct::Tensor& grad_out,
        ct::Tensor& grad_old_out)
{
    calc_grad_all_band_op()(this->ctx, prec_in.data<FPTYPE>(), err_out.data<FPTYPE>(), beta_out.data<FPTYPE>(), psi_in.data<std::complex<FPTYPE>>(), hpsi_in.data<std::complex<FPTYPE>>(), grad_out.data<std::complex<FPTYPE>>(), grad_old_out.data<std::complex<FPTYPE>>(), this->n_basis, this->n_basis, this->n_band);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_prec()
{
    syncmem_var_h2d_op()(
        this->ctx,
        this->cpu_ctx,
        this->prec.data<FPTYPE>(),
        this->h_prec.data<FPTYPE>(),
        this->n_basis);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::orth_projection(
        const ct::Tensor& psi_in,
        ct::Tensor& hsub_in,
        ct::Tensor& grad_out)
{
    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&hsub_in);
    hsub_in = ct::op::einsum("ij,kj->ik", grad_out, psi_in, option);

    // set_matrix_op()(this->ctx, 'L', hsub_in->data<std::complex<FPTYPE>>(), this->n_band);
    option = ct::EinsumOption(
        /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/-1.0, /*beta=*/1.0, /*Tensor out=*/&grad_out);
    grad_out = ct::op::einsum("ij,jk->ik", hsub_in, psi_in, option);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::rotate_wf(
        const ct::Tensor& hsub_in,
        ct::Tensor& psi_out,
        ct::Tensor& workspace_in)
{
    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&workspace_in);
    workspace_in = ct::op::einsum("ij,jk->ik", hsub_in, psi_out, option);

    syncmem_complex_op()(this->ctx, this->ctx, psi_out.data<std::complex<FPTYPE>>(), workspace_in.data<std::complex<FPTYPE>>(), this->n_band * this->n_basis);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_hpsi_all_band(
        hamilt::Hamilt<FPTYPE, Device>* hamilt_in,
        const psi::Psi<std::complex<FPTYPE>, Device>& psi_in,
        ct::Tensor& hpsi_out)
{
    // calculate all-band hpsi
    psi::Range all_bands_range(1, psi_in.get_current_k(), 0, psi_in.get_nbands() - 1);
    hpsi_info info(&psi_in, all_bands_range, hpsi_out.data<std::complex<FPTYPE>>());
    hamilt_in->ops->hPsi(info);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::diag_hsub(
        const ct::Tensor& psi_in,
        const ct::Tensor& hpsi_in,
        ct::Tensor& hsub_out,
        ct::Tensor& eigenvalue_out)
{
    // calculate all-band hsub
    // Note: ctx is nothing but the devices used in this class (Device * ctx = nullptr;),
    // it controls the ops to use the corresponding device to calculate results
    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&hsub_out);
    hsub_out = ct::op::einsum("ij,kj->ik", psi_in, hpsi_in, option);

    dnevd_op()(this->ctx, hsub_out.data<std::complex<FPTYPE>>(), this->n_band, eigenvalue_out.data<FPTYPE>());
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::calc_hsub_all_band(
        hamilt::Hamilt<FPTYPE, Device> *hamilt_in,
        const psi::Psi<std::complex<FPTYPE>, Device> &psi_in,
        ct::Tensor& psi_out,
        ct::Tensor& hpsi_out,
        ct::Tensor& hsub_out,
        ct::Tensor& workspace_in,
        ct::Tensor& eigenvalue_out)
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
void DiagoAllBandCG<FPTYPE, Device>::calc_hsub_all_band_exit(
        ct::Tensor& psi_out, 
        ct::Tensor& hpsi_out,
        ct::Tensor& hsub_out, 
        ct::Tensor& workspace_in,
        ct::Tensor& eigenvalue_out)
{
    // Diagonalization of the subspace matrix.
    this->diag_hsub(psi_out, hpsi_out, hsub_out, eigenvalue_out);

    // inplace matmul to get the initial guessed wavefunction psi.
    // psi_out[n_basis, n_band] = psi_out[n_basis, n_band] x hsub_out[n_band, n_band]
    this->rotate_wf(hsub_out, psi_out, workspace_in);
}

template<typename FPTYPE, typename Device>
void DiagoAllBandCG<FPTYPE, Device>::diag(
        hamilt::Hamilt<FPTYPE, Device>* hamilt_in,
        psi::Psi<std::complex<FPTYPE>, Device>& psi_in,
        FPTYPE* eigenvalue_in)
{
    const int current_scf_iter = hsolver::DiagoIterAssist<FPTYPE, Device>::SCF_ITER;
    // Get the pointer of the input psi
    this->psi = std::move(ct::TensorMap(psi_in.get_pointer(), cx_type, device_type, {this->n_band, this->n_basis}));
    // Update the precondition array
    this->calc_prec();

    // Improving the initial guess of the wave function psi through a subspace diagonalization.
    this->calc_hsub_all_band(hamilt_in, psi_in, this->psi, this->hpsi, this->hsub, this->work, this->eigen);

    setmem_complex_op()(this->ctx, this->grad_old.data<std::complex<FPTYPE>>(), 0, this->n_basis * this->n_band);
    setmem_var_op()(this->ctx, this->beta.data<FPTYPE>(), 1E+40, this->n_band);
    int ntry = 0;
    int max_iter = current_scf_iter > 1 ?
                   this->nline :
                   this->nline * 6;
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
        syncmem_complex_op()(this->ctx, this->ctx, this->grad_old.data<std::complex<FPTYPE>>(), this->grad.data<std::complex<FPTYPE>>(), n_basis * n_band);

        // Calculate H|grad> matrix
        this->calc_hpsi_all_band(hamilt_in, this->grad_wrapper[0], this->hgrad);

        // optimize psi as well as the hpsi
        // 1. normalize grad
        // 2. calculate theta
        // 3. update psi as well as hpsi
        this->line_minimize(this->grad, this->hgrad, this->psi, this->hpsi);

        // orthogonal psi by cholesky method
        this->orth_cholesky(this->work, this->psi, this->hpsi, this->hsub);

        if (current_scf_iter == 1 && ntry % this->nline == 0) {
            this->calc_hsub_all_band(hamilt_in, psi_in, this->psi, this->hpsi, this->hsub, this->work, this->eigen);
        }
    } while (ntry < max_iter && this->test_error(this->err_st, this->all_band_cg_thr));
    this->calc_hsub_all_band_exit(this->psi, this->hpsi, this->hsub, this->work, this->eigen);
    syncmem_var_d2h_op()(this->cpu_ctx, this->ctx, eigenvalue_in, this->eigen.data<FPTYPE>(), this->n_band);
}

template class DiagoAllBandCG<float, psi::DEVICE_CPU>;
template class DiagoAllBandCG<double, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoAllBandCG<float, psi::DEVICE_GPU>;
template class DiagoAllBandCG<double, psi::DEVICE_GPU>;
#endif

} // namespace hsolver