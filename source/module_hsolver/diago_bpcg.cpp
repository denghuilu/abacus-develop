#include "module_hsolver/diago_bpcg.h"

#include <ATen/kernels/blas.h>
#include <ATen/kernels/lapack.h>

#include <ATen/ops/einsum_op.h>

#include "diago_iter_assist.h"
#include "module_base/blas_connector.h"
#include "module_base/global_function.h"
#include "module_hsolver/kernels/math_kernel_op.h"

namespace hsolver {

template<typename T, typename Device>
DiagoBPCG<T, Device>::DiagoBPCG(int n_band, int n_basis)
{
    this->n_band_ = n_band;
    this->n_basis_x_ = n_basis;
    this->c_type_ = ct::DataTypeToEnum<T>::value;
    this->r_type_ = ct::DataTypeToEnum<Real>::value;
    this->device_type_ = ct::DeviceTypeToEnum<Device>::value;
    this->work_space_ = ct::Tensor(c_type_, device_type_, {5, n_band, n_basis});

    this->beta_       = std::move(ct::Tensor(r_type_, device_type_, {this->n_band_}));
    this->eigen_      = std::move(ct::Tensor(r_type_, device_type_, {this->n_band_}));
    this->err_st_     = std::move(ct::Tensor(r_type_, device_type_, {this->n_band_}));
    this->hsub_       = std::move(ct::Tensor(c_type_, device_type_, {this->n_band_, this->n_band_}));
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::init_iter(const ct::Tensor& psi) {
    // Specify the problem size n_basis_, n_band_, while lda is n_basis_
    this->n_basis_    = psi.shape().dim_size(1);
    REQUIRES_OK(n_basis_ <= n_basis_x_, "n_basis_ should be less than n_basis_x_!");
    
    // All column major tensors
    auto work_pack = work_space_.accessor<T, 3>();
    this->hpsi_       = std::move(ct::TensorMap(&work_pack[0][0][0], c_type_, device_type_, {this->n_band_, this->n_basis_}));
    this->work_       = std::move(ct::TensorMap(&work_pack[1][0][0], c_type_, device_type_, {this->n_band_, this->n_basis_}));
    this->grad_       = std::move(ct::TensorMap(&work_pack[2][0][0], c_type_, device_type_, {this->n_band_, this->n_basis_}));
    this->hgrad_      = std::move(ct::TensorMap(&work_pack[3][0][0], c_type_, device_type_, {this->n_band_, this->n_basis_}));
    this->grad_old_   = std::move(ct::TensorMap(&work_pack[4][0][0], c_type_, device_type_, {this->n_band_, this->n_basis_}));
}

template<typename T, typename Device>
bool DiagoBPCG<T, Device>::test_error(const ct::Tensor& err_in, Real thr_in)
{
    const Real * _err_st = err_in.data<Real>();
    if (err_in.device_type() == ct::DeviceType::GpuDevice) {
        ct::Tensor h_err_in = err_in.to_device<ct::DEVICE_CPU>();
        _err_st = h_err_in.data<Real>();
    }
    for (int ii = 0; ii < this->n_band_; ii++) {
        if (_err_st[ii] > thr_in) {
            return true;
        }
    }
    return false;
}

// Finally, the last one!
template<typename T, typename Device>
void DiagoBPCG<T, Device>::line_minimize(
    ct::Tensor& grad_in,
    ct::Tensor& hgrad_in,
    ct::Tensor& psi_out,
    ct::Tensor& hpsi_out)
{
    line_minimize_with_block_op()(grad_in.data<T>(), hgrad_in.data<T>(), psi_out.data<T>(), hpsi_out.data<T>(), this->n_basis_, this->n_basis_, this->n_band_);
}

// Finally, the last two!
template<typename T, typename Device>
void DiagoBPCG<T, Device>::orth_cholesky(ct::Tensor& workspace_in, ct::Tensor& psi_out, ct::Tensor& hpsi_out, ct::Tensor& hsub_out)
{
    // hsub_out = psi_out * transc(psi_out)
    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&hsub_out);
    hsub_out = ct::op::einsum("ij,kj->ik", psi_out, psi_out, option);

    // set hsub matrix to lower format;
    ct::kernels::set_matrix<T, ct_Device>()(
        'L', hsub_out.data<T>(), this->n_band_);

    ct::kernels::lapack_potrf<T, ct_Device>()(
        'U', this->n_band_, hsub_out.data<T>(), this->n_band_);
    ct::kernels::lapack_trtri<T, ct_Device>()(
        'U', 'N', this->n_band_, hsub_out.data<T>(), this->n_band_);

    this->rotate_wf(hsub_out, psi_out, workspace_in);
    this->rotate_wf(hsub_out, hpsi_out, workspace_in);
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::calc_grad_with_block(
        const ct::Tensor& prec_in,
        ct::Tensor& err_out,
        ct::Tensor& beta_out,
        ct::Tensor& psi_in,
        ct::Tensor& hpsi_in,
        ct::Tensor& grad_out,
        ct::Tensor& grad_old_out)
{
    calc_grad_with_block_op()(prec_in.data<Real>(), err_out.data<Real>(), beta_out.data<Real>(), psi_in.data<T>(), hpsi_in.data<T>(), grad_out.data<T>(), grad_old_out.data<T>(), this->n_basis_, this->n_basis_, this->n_band_);
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::orth_projection(
        const ct::Tensor& psi_in,
        ct::Tensor& hsub_in,
        ct::Tensor& grad_out)
{
    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&hsub_in);
    hsub_in = ct::op::einsum("ij,kj->ik", grad_out, psi_in, option);

    // set_matrix_op()('L', hsub_in->data<T>(), this->n_band_);
    option = ct::EinsumOption(
        /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/-1.0, /*beta=*/1.0, /*Tensor out=*/&grad_out);
    grad_out = ct::op::einsum("ij,jk->ik", hsub_in, psi_in, option);
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::rotate_wf(
        const ct::Tensor& hsub_in,
        ct::Tensor& psi_out,
        ct::Tensor& workspace_in)
{
    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&workspace_in);
    workspace_in = ct::op::einsum("ij,jk->ik", hsub_in, psi_out, option);

    syncmem_complex_op()(psi_out.template data<T>(), workspace_in.template data<T>(), this->n_band_ * this->n_basis_);
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::diag_hsub(
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

    ct::kernels::lapack_dnevd<T, ct_Device>()('V', 'U', hsub_out.data<T>(), this->n_band_, eigenvalue_out.data<Real>());
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::calc_hsub_with_block(
        const ct::Tensor& psi_in,
        ct::Tensor& psi_out,
        ct::Tensor& hpsi_out,
        ct::Tensor& hsub_out,
        ct::Tensor& workspace_in,
        ct::Tensor& eigenvalue_out)
{
    // Apply the H operator to psi and obtain the hpsi matrix.
    this->hpsi_func_(psi_in, hpsi_out);

    // Diagonalization of the subspace matrix.
    this->diag_hsub(psi_out, hpsi_out, hsub_out, eigenvalue_out);

    // inplace matmul to get the initial guessed wavefunction psi.
    // psi_out[n_basis_, n_band_] = psi_out[n_basis_, n_band_] x hsub_out[n_band_, n_band_]
    // hpsi_out[n_basis_, n_band_] = psi_out[n_basis_, n_band_] x hsub_out[n_band_, n_band_]
    this->rotate_wf(hsub_out, psi_out, workspace_in);
    this->rotate_wf(hsub_out, hpsi_out, workspace_in);
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::calc_hsub_with_block_exit(
        ct::Tensor& psi_out, 
        ct::Tensor& hpsi_out,
        ct::Tensor& hsub_out, 
        ct::Tensor& workspace_in,
        ct::Tensor& eigenvalue_out)
{
    // Diagonalization of the subspace matrix.
    this->diag_hsub(psi_out, hpsi_out, hsub_out, eigenvalue_out);

    // inplace matmul to get the initial guessed wavefunction psi.
    // psi_out[n_basis_, n_band_] = psi_out[n_basis_, n_band_] x hsub_out[n_band_, n_band_]
    this->rotate_wf(hsub_out, psi_out, workspace_in);
}

template<typename T, typename Device>
void DiagoBPCG<T, Device>::diag(
    const Func& hpsi_func, 
    const Func& spsi_func, 
    ct::Tensor& psi,
    ct::Tensor& eigen,
    const ct::Tensor& prec)
{
    this->init_iter(psi);
    this->hpsi_func_ = hpsi_func;
    this->spsi_func_ = spsi_func;
    this->eigen_ = eigen.to_device<ct_Device>();
    const int current_scf_iter = hsolver::DiagoIterAssist<T, Device>::SCF_ITER;
    // Improving the initial guess of the wave function psi through a subspace diagonalization.
    this->calc_hsub_with_block(psi, psi, hpsi_, hsub_, work_, eigen_);
    setmem_complex_op()(this->grad_old_.template data<T>(), 0, this->n_basis_ * this->n_band_);
    setmem_var_op()(this->beta_.template data<Real>(), 1E+40, this->n_band_);
    int ntry = 0;
    int max_iter = current_scf_iter > 1 ?
                   this->nline_ :
                   this->nline_ * 6;
    do
    {
        ++ntry;
        // Be careful here ! dangerous zone!
        // 1. normalize psi
        // 2. calculate the epsilo
        // 3. calculate the gradient by hpsi - epsilo * psi
        // 4. gradient mix with the previous gradient
        // 5. Do precondition
        this->calc_grad_with_block(prec, this->err_st_, this->beta_,
                                   psi, this->hpsi_, this->grad_, this->grad_old_);

        // Orthogonalize column vectors g_i in matrix grad to column vectors p_j in matrix psi
        // for all 'j less or equal to i'.
        // Note: hsub and work are only used to store intermediate variables of gemm operator.
        this->orth_projection(psi, this->hsub_, this->grad_);

        // this->grad_old = this->grad;
        syncmem_complex_op()(this->grad_old_.template data<T>(), this->grad_.template data<T>(), n_basis_ * n_band_);

        // Calculate H|grad> matrix
        this->hpsi_func_(this->grad_, this->hgrad_);

        // optimize psi as well as the hpsi
        // 1. normalize grad
        // 2. calculate theta
        // 3. update psi as well as hpsi
        this->line_minimize(this->grad_, this->hgrad_, psi, this->hpsi_);

        // orthogonal psi by cholesky method
        this->orth_cholesky(this->work_, psi, this->hpsi_, this->hsub_);

        if (current_scf_iter == 1 && ntry % this->nline_ == 0) {
            this->calc_hsub_with_block(psi, psi, this->hpsi_, this->hsub_, this->work_, this->eigen_);
        }
    } while (ntry < max_iter && this->test_error(this->err_st_, this->all_band_cg_thr_));
    this->calc_hsub_with_block_exit(psi, this->hpsi_, this->hsub_, this->work_, this->eigen_);
    eigen = this->eigen_.to_device<ct::DEVICE_CPU>();
}

template class DiagoBPCG<std::complex<float>, psi::DEVICE_CPU>;
template class DiagoBPCG<std::complex<double>, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoBPCG<std::complex<float>, psi::DEVICE_GPU>;
template class DiagoBPCG<std::complex<double>, psi::DEVICE_GPU>;
#endif

} // namespace hsolver