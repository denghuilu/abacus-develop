#include "module_hsolver/diago_cg.h"

#include "diago_iter_assist.h"
#include "module_base/blas_connector.h"
#include "module_base/constants.h"
#include "module_base/global_function.h"
#include "module_base/timer.h"
#include "module_base/parallel_reduce.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_hamilt_pw/hamilt_pwdft/hamilt_pw.h"
#include "module_base/memory.h"

#include <ATen/core/tensor_map.h>

using namespace hsolver;

template<typename T, typename Device>
DiagoCG<T, Device>::DiagoCG(const ct::Tensor& prec_in, const ct::Tensor& n_basis_in)
{
    // reference to prec_in and n_basis_in
    this->h_prec_.CopyFrom(prec_in);
    this->n_basis_.CopyFrom(n_basis_in);
    
    this->test_cg_ = 0;
    this->reorder_ = false;
    this->one_ = ct::Tensor({static_cast<T>(1.0)});
    this->zero_ = ct::Tensor({static_cast<T>(0.0)});
    this->neg_one_ = ct::Tensor({static_cast<T>(-1.0)});
}

template<typename T, typename Device>
void DiagoCG<T, Device>::diag_mock(hamilt::Hamilt* phm_in, ct::Tensor& psi, ct::Tensor& eigen_in)
{
    ModuleBase::TITLE("DiagoCG", "diag_once");
    ModuleBase::timer::tick("DiagoCG", "diag_once");
    
    /// out : record for states of convergence
    this->notconv_ = 0;

    /// initialize variables
    this->n_band_ = psi.shape().dim_size(1);
    this->n_basis_max_ = psi.shape().dim_size(2);
    // Note: this->ik is a member value of basis class DiagH, 
    // and this value was initialized by a function call within the HsolverPW's solve function
    this->current_n_basis_ = this->n_basis_.data<int>()[this->ik_];
    // Reference to the eigen_in
    this->eigen_.CopyFrom(eigen_in);

    this->eigen_.zero();

    /// record for how many loops in cg convergence
    int avg = 0;

    //-------------------------------------------------------------------
    // "poor man" iterative diagonalization of a complex hermitian matrix
    // through preconditioned conjugate gradient algorithm
    // Band-by-band algorithm with minimal use of memory
    // Calls hPhi and sPhi to calculate H|phi> and S|phi>
    // Works for generalized eigenvalue problem (US pseudopotentials) as well
    //-------------------------------------------------------------------
    // this->phi_m = new psi::Psi<T, Device>(phi, 1, 1);
    this->phi_m_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    // this->hphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    this->hphi_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    this->hphi_.zero();
    // this->sphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    this->sphi_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    this->sphi_.zero();

    // this->cg = new psi::Psi<T, Device>(phi, 1, 1);
    this->cg_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    // this->scg.resize(this->n_basis_max_, ModuleBase::ZERO);
    this->scg_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    this->scg_.zero();
    // this->pphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    this->pphi_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    this->pphi_.zero();

    // this->gradient.resize(this->n_basis_max_, ModuleBase::ZERO);
    this->gradient_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    this->gradient_.zero();
    // this->g0.resize(this->n_basis_max_, ModuleBase::ZERO);
    this->g0_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->current_n_basis_}));
    this->g0_.zero();
    // this->lagrange.resize(this->n_band, ModuleBase::ZERO);
    this->lagrange_ = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_band_}));
    this->lagrange_.zero();

    ModuleBase::Memory::record("DiagoCG", this->current_n_basis_ * 8);

    auto psi_pack = psi.accessor<T, 3>();
    auto eigen_pack = this->eigen_.accessor<Real, 1>();
    for (int m = 0; m < this->n_band_; m++)
    {
        if (test_cg_ > 2)
            GlobalV::ofs_running << "Diagonal Band : " << m << std::endl;
        //copy psi_in into internal psi, m=0 has been done in Constructor
        if (m > 0)
        {
            // ModuleBase::GlobalFunc::COPYARRAY(psi_m_in, pphi_m, this->current_n_basis_);
            phi_m_.sync(
                ct::TensorMap(&psi_pack[this->ik_][m][0], psi.data_type(), psi.device_type(), {this->current_n_basis_}));
        }
        phm_in->sPsi(this->phi_m_, this->sphi_); // sphi = S|psi(m)>
        this->schmit_orth(m, psi);
        phm_in->sPsi(this->phi_m_, this->sphi_); // sphi = S|psi(m)>

        //do hPsi, actually the result of hpsi stored in Operator,
        //the necessary of copying operation should be checked later
        phm_in->ops->hPsi(this->phi_m_, this->hphi_);

        // TODO: Remove zdot_real_op and use Tensor operators instead.
        eigen_pack[m] = 
            zdot_real_op()(
                this->ctx_, 
                this->current_n_basis_, 
                this->phi_m_.data<T>(), 
                this->hphi_.data<T>());

        int iter = 0;
        Real gg_last = 0.0;
        Real cg_norm = 0.0;
        Real theta = 0.0;
        bool converged = false;
        for (iter = 0; iter < DiagoIterAssist<T, Device>::PW_DIAG_NMAX; iter++)
        {
            this->calculate_gradient();
            this->orthogonal_gradient(phm_in, psi, m);
            this->calculate_gamma_cg(iter, gg_last, cg_norm, theta);
            
            phm_in->ops->hPsi(this->cg_, this->pphi_);

            phm_in->sPsi(this->cg_, this->scg_);
            converged = this->update_psi(cg_norm, theta, eigen_pack[m]);

            if (converged) {
                break;
            }
        } // end iter

        // ModuleBase::GlobalFunc::COPYARRAY(this->phi_m->get_pointer(), psi_temp, this->current_n_basis_);
        auto psi_temp = ct::TensorMap(
            &psi_pack[this->ik_][m][0], psi.data_type(), psi.device_type(), {this->current_n_basis_});
        psi_temp.sync(this->phi_m_);

        if (!converged)
        {
            ++this->notconv_;
        }
        avg += iter + 1;

        // reorder eigenvalues if they are not in the right order
        // (this CAN and WILL happen in not-so-special cases)

        if (m > 0 && this->reorder_)
        {
            ModuleBase::GlobalFunc::NOTE("reorder bands!");
            if (eigen_pack[m] - eigen_pack[m - 1] < -2.0 * DiagoIterAssist<T, Device>::PW_DIAG_THR)
            {
                // if the last calculated eigen_pack is not the largest...
                int i = 0;
                for (i = m - 2; i >= 0; i--)
                {
                    if (eigen_pack[m] - eigen_pack[i] > 2.0 * DiagoIterAssist<T, Device>::PW_DIAG_THR)
                        break;
                }
                i++;

                // last calculated eigen_pack should be in the i-th position: reorder
                Real e0 = eigen_pack[m];
                // ModuleBase::GlobalFunc::COPYARRAY(psi_temp, pphi, this->current_n_basis_);
                this->pphi_.sync(psi_temp);

                for (int j = m; j >= i + 1; j--)
                {
                    eigen_pack[j] = eigen_pack[j - 1];
                    auto phi_j = ct::TensorMap(
                        &psi_pack[this->ik_][j][0], psi.data_type(), psi.device_type(), {this->current_n_basis_});
                    auto phi_j1 = ct::TensorMap(
                        &psi_pack[this->ik_][j - 1][0], psi.data_type(), psi.device_type(), {this->current_n_basis_});
                    // ModuleBase::GlobalFunc::COPYARRAY(phi_j1, phi_j, this->current_n_basis_);
                    phi_j.sync(phi_j1);
                }

                eigen_pack[i] = e0;
                // ModuleBase::GlobalFunc::COPYARRAY(pphi, phi_pointer, this->current_n_basis_);
                psi_temp = ct::TensorMap(
                    &psi_pack[this->ik_][i][0], psi.data_type(), psi.device_type(), {this->current_n_basis_});
                psi_temp.sync(this->pphi_);

            } // endif
        } // end reorder
    } // end m

    avg /= this->n_band_;
    DiagoIterAssist<T, Device>::avg_iter += avg;

    ModuleBase::timer::tick("DiagoCG", "diag_once");
} // end subroutine ccgdiagg

template<typename T, typename Device>
void DiagoCG<T, Device>::calculate_gradient()
{
    if (this->test_cg_ == 1) {
        ModuleBase::TITLE("DiagoCG", "calculate_gradient");
    }
    // for (int i = 0; i < this->current_n_basis_; i++)
    // {
    //     //(2) PH|psi>
    //     this->gradient[i] = this->hphi[i] / this->precondition[i];
    //     //(3) PS|psi>
    //     this->pphi[i] = this->sphi[i] / this->precondition[i];
    // }
    // denghui replace this at 20221106
    // TODO: use GPU precondition to initialize CG class
    // TODO: Use tensor operators
    vector_div_vector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->gradient_.data<T>(), this->hphi_.data<T>(), this->prec_.data<Real>());
    vector_div_vector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->pphi_.data<T>(), this->sphi_.data<T>(), this->prec_.data<Real>());

    // Update lambda !
    // (4) <psi|SPH|psi >
    const Real eh = hsolver::zdot_real_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->sphi_.data<T>(), this->gradient_.data<T>());
    // (5) <psi|SPS|psi >
    const Real es = hsolver::zdot_real_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->sphi_.data<T>(), this->pphi_.data<T>());
    const Real lambda = eh / es;

    // Update g!
    // for (int i = 0; i < this->current_n_basis_; i++)
    // {
    //     //               <psi|SPH|psi>
    //     // (6) PH|psi> - ------------- * PS |psi>
    //     //               <psi|SPS|psi>
    //     //
    //     // So here we get the gradient.
    //     this->gradient[i] -= lambda * this->pphi[i];
    // }
    // haozhihan replace this 2022-10-6
    constantvector_addORsub_constantVector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->gradient_.data<T>(), this->gradient_.data<T>(), 1.0, this->pphi_.data<T>(), (-lambda));
}

template<typename T, typename Device>
void DiagoCG<T, Device>::orthogonal_gradient(hamilt::Hamilt* phm_in, const ct::Tensor& eigenfunction, const int& m)
{
    auto psi_pack = eigenfunction.accessor<T, 3>();
    if (this->test_cg_ == 1) {
        ModuleBase::TITLE("DiagoCG", "orthogonal_gradient");
    }
    // ModuleBase::timer::tick("DiagoCG","orth_grad");

    phm_in->sPsi(this->gradient_, this->scg_);
    // int inc = 1;
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022-10-07
    // TODO: Use Tensor operators
    gemv_op<Real, Device>()(
        this->ctx_,
        'C',
        this->current_n_basis_,
        m,
        this->one_.data<T>(),
        &psi_pack[this->ik_][0][0],
        this->n_basis_max_,
        this->scg_.data<T>(),
        1,
        this->zero_.data<T>(),
        this->lagrange_.data<T>(),
        1);

    // TODO: add a wrapper for parallel reduce
    Parallel_Reduce::reduce_complex_double_pool(this->lagrange_.data<T>(), m);

    // (3) orthogonal |g> and |scg> to all states (0~m-1)
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022-10-07
    gemv_op<Real, Device>()(
        this->ctx_,
        'N',
        this->current_n_basis_,
        m,
        this->neg_one_.data<T>(),
        &psi_pack[this->ik_][0][0],
        this->n_basis_max_,
        this->lagrange_.data<T>(),
        1,
        this->one_.data<T>(),
        this->gradient_.data<T>(),
        1);

    gemv_op<Real, Device>()(
        this->ctx_,
        'N',
        this->current_n_basis_,
        m,
        this->neg_one_.data<T>(),
        &psi_pack[this->ik_][0][0],
        this->n_basis_max_,
        this->lagrange_.data<T>(),
        1,
        this->one_.data<T>(),
        this->scg_.data<T>(),
        1);
}

template<typename T, typename Device>
void DiagoCG<T, Device>::calculate_gamma_cg(const int& iter, Real& gg_last, const Real& cg_norm, const Real& theta)
{
    if (this->test_cg_ == 1) {
        ModuleBase::TITLE("DiagoCG", "calculate_gamma_cg");
    }
    Real gg_inter;
    if (iter > 0)
    {
        // (1) Update gg_inter!
        // gg_inter = <g|g0>
        // Attention : the 'g' in g0 is getted last time
        gg_inter
            = hsolver::zdot_real_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->gradient_.data<T>(), this->g0_.data<T>()); // b means before
    }

    // (2) Update for g0!
    // two usage:
    // firstly, for now, calculate: gg_now
    // secondly, prepare for the next iteration: gg_inter
    // |g0> = P | scg >
    // for (int i = 0; i < this->current_n_basis_; i++)
    // {
    //     this->g0[i] = this->precondition[i] * this->scg[i];
    // }
    // denghui replace this 20221106
    // TODO: use GPU precondition instead
    vector_mul_vector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->g0_.data<T>(), this->scg_.data<T>(), this->prec_.data<Real>());

    // (3) Update gg_now!
    // gg_now = < g|P|scg > = < g|g0 >
    const Real gg_now = hsolver::zdot_real_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->gradient_.data<T>(), this->g0_.data<T>());

    if (iter == 0)
    {
        // (40) gg_last first value : equal gg_now
        gg_last = gg_now;
        // (50) cg direction first value : |g>
        // |cg> = |g>
        // ModuleBase::GlobalFunc::COPYARRAY(this->gradient, pcg, this->current_n_basis_);
        this->cg_.sync(this->gradient_);
    }
    else
    {
        // (4) Update gamma !
        assert(gg_last != 0.0);
        const Real gamma = (gg_now - gg_inter) / gg_last;

        // (5) Update gg_last !
        gg_last = gg_now;

        // (6) Update cg direction !(need gamma and |go> ):
        // for (int i = 0; i < this->current_n_basis_; i++)
        // {
        //     pcg[i] = gamma * pcg[i] + this->gradient[i];
        // }
        // haozhihan replace this 2022-10-6
        constantvector_addORsub_constantVector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->cg_.data<T>(), this->cg_.data<T>(), gamma, this->gradient_.data<T>(), 1.0);

        const Real norma = gamma * cg_norm * sin(theta);
        T znorma(norma * -1, 0.0);

        // haozhihan replace this 2022-10-6
        // const int one = 1;
        // zaxpy_(&this->current_n_basis_, &znorma, pphi_m, &one, pcg, &one);
        /*for (int i = 0; i < this->current_n_basis_; i++)
        {
            pcg[i] -= norma * pphi_m[i];
        }*/
        axpy_op<Real, Device>()(this->ctx_, this->current_n_basis_, &znorma, this->phi_m_.data<T>(), 1, this->cg_.data<T>(), 1);
    }
}

template<typename T, typename Device>
bool DiagoCG<T, Device>::update_psi(Real &cg_norm, Real &theta, Real &eigenvalue)
{
    if (this->test_cg_ == 1)
        ModuleBase::TITLE("DiagoCG", "update_psi");
    cg_norm = sqrt(hsolver::zdot_real_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->cg_.data<T>(), this->scg_.data<T>()));

    if (cg_norm < 1.0e-10)
        return 1;

    const Real a0
        = hsolver::zdot_real_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->phi_m_.data<T>(), this->pphi_.data<T>()) * 2.0 / cg_norm;
    const Real b0
        = hsolver::zdot_real_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->cg_.data<T>(), this->pphi_.data<T>()) / (cg_norm * cg_norm);

    const Real e0 = eigenvalue;
    theta = atan(a0 / (e0 - b0)) / 2.0;

    const Real new_e = (e0 - b0) * cos(2.0 * theta) + a0 * sin(2.0 * theta);

    const Real e1 = (e0 + b0 + new_e) / 2.0;
    const Real e2 = (e0 + b0 - new_e) / 2.0;

    if (e1 > e2)
    {
        theta += ModuleBase::PI_HALF;
    }

    eigenvalue = std::min(e1, e2);
    //	OUT("eigenvalue",eigenvalue);

    const Real cost = cos(theta);
    const Real sint_norm = sin(theta) / cg_norm;

    // for (int i = 0; i < this->current_n_basis_; i++)
    // {
    //     phi_m_pointer[i] = phi_m_pointer[i] * cost + sint_norm * pcg[i];
    // }
    
    // haozhihan replace this 2022-10-6
    constantvector_addORsub_constantVector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->phi_m_.data<T>(), this->phi_m_.data<T>(), cost, this->cg_.data<T>(), sint_norm);


    //	std::cout << "\n overlap2 = "  << this->ddot(dim, phi_m, phi_m);

    if (std::abs(eigenvalue - e0) < DiagoIterAssist<T, Device>::PW_DIAG_THR)
    {
        // ModuleBase::timer::tick("DiagoCG","update");
        return 1;
    }
    else
    {
        // for (int i = 0; i < this->current_n_basis_; i++)
        // {
        //     this->sphi[i] = this->sphi[i] * cost + sint_norm * this->scg[i];
        //     this->hphi[i] = this->hphi[i] * cost + sint_norm * this->pphi[i];
        // }

        // haozhihan replace this 2022-10-6
        constantvector_addORsub_constantVector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->sphi_.data<T>(), this->sphi_.data<T>(), cost, this->scg_.data<T>(), sint_norm);
        constantvector_addORsub_constantVector_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->hphi_.data<T>(), this->hphi_.data<T>(), cost, this->pphi_.data<T>(), sint_norm);
        return 0;
    }
}

template<typename T, typename Device>
void DiagoCG<T, Device>::schmit_orth(
                          const int& m, // end
                          const ct::Tensor& psi)
{
    //	ModuleBase::TITLE("DiagoCG","schmit_orth");
    // ModuleBase::timer::tick("DiagoCG","schmit_orth");
    // orthogonalize starting eigenfunction to those already calculated
    // phi_m orthogonalize to psi(start) ~ psi(m-1)
    // Attention, the orthogonalize here read as
    // psi(m) -> psi(m) - \sum_{i < m} < psi(i) | S | psi(m) > psi(i)
    // so the orthogonalize is performed about S.
    REQUIRES_OK(m >= 0);
    REQUIRES_OK(this->n_band_ >= m);

    ct::Tensor lagrange_so = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {m + 1});

    auto psi_pack = psi.accessor<T, 3>();
    int inc = 1;
    // TODO: Use Tensor operators
    gemv_op<Real, Device>()(
        this->ctx_,
        'C',
        this->current_n_basis_,
        m + 1,
        this->one_.data<T>(),
        &psi_pack[this->ik_][0][0],
        this->n_basis_max_,
        this->sphi_.data<T>(),
        inc,
        this->zero_.data<T>(),
        lagrange_so.data<T>(),
        inc);

    // be careful , here reduce m+1
    Parallel_Reduce::reduce_complex_double_pool(lagrange_so.data<T>(), m + 1);

    T var(0, 0);
    syncmem_complex_d2h_op()(this->cpu_ctx_, this->ctx_, &var, lagrange_so.data<T>() + m, 1);
    Real psi_norm = var.real();

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022-10-6
    gemv_op<Real, Device>()(
        this->ctx_,
        'N',
        this->current_n_basis_,
        m,
        this->neg_one_.data<T>(),
        &psi_pack[this->ik_][0][0],
        this->n_basis_max_,
        lagrange_so.data<T>(),
        inc,
        this->one_.data<T>(),
        this->phi_m_.data<T>(),
        inc);

    //======================================================================
    /*for (int j = 0; j < m; j++)
    {
        for (int ig =0; ig < dim; ig++)
        {
            phi_m[ig] -= lagrange[j] * psi(j, ig);
        }
        psi_norm -= ( conj(lagrange[j]) * lagrange[j] ).real();
    }*/
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    psi_norm -= hsolver::zdot_real_op<Real, Device>()(this->ctx_, m, lagrange_so.data<T>(), lagrange_so.data<T>(), false);

    if (psi_norm <= 0.0)
    {
        std::cout << " m = " << m << std::endl;
        for (int j = 0; j <= m; ++j)
        {
            std::cout << "j = " << j << " lagrange norm = " << (conj(lagrange_so.data<T>()[j]) * lagrange_so.data<T>()[j]).real()
                      << std::endl;
        }
        std::cout << " in DiagoCG, psi norm = " << psi_norm << std::endl;
        std::cout << " If you use GNU compiler, it may due to the zdotc is unavailable." << std::endl;
        ModuleBase::WARNING_QUIT("schmit_orth", "psi_norm <= 0.0");
    }

    psi_norm = sqrt(psi_norm);

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022-10-6
    // scal_op<Real, Device>()(this->ctx_, this->current_n_basis_, &psi_norm, pphi_m, 1);
    //======================================================================
    // for (int ig = 0; ig < this->current_n_basis_; ig++)
    // {
    //     pphi_m[ig] /= psi_norm;
    // }
    vector_div_constant_op<Real, Device>()(this->ctx_, this->current_n_basis_, this->phi_m_.data<T>(), this->phi_m_.data<T>(), psi_norm);

    // ModuleBase::timer::tick("DiagoCG","schmit_orth");
}

template<typename T, typename Device>
void DiagoCG<T, Device>::diag(hamilt::Hamilt* phm_in, ct::Tensor& psi, ct::Tensor& eigenvalue_in)
{
    /// record the times of trying iterative diagonalization
    int ntry = 0;
    this->notconv_ = 0;
    this->prec_.sync(this->h_prec_);
    do
    {
        if(DiagoIterAssist<T, Device>::need_subspace || ntry > 0)
        {
            DiagoIterAssist<T, Device>::diagH_subspace(phm_in, psi, psi, eigenvalue_in, this->ik_, this->current_n_basis_);
        }

        DiagoIterAssist<T, Device>::avg_iter += 1.0;
        this->reorder_ = true;

        this->diag_mock(phm_in, psi, eigenvalue_in);

        ++ntry;
    } while (DiagoIterAssist<T, Device>::test_exit_cond(ntry, this->notconv_));

    if (notconv_ > std::max(5, this->n_band_ / 4)) {
        std::cout << "\n notconv = " << this->notconv_;
        std::cout << "\n DiagoCG::diag', too many bands are not converged! \n";
    }
}

namespace hsolver {
template class DiagoCG<std::complex<float>, psi::DEVICE_CPU>;
template class DiagoCG<std::complex<double>, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoCG<std::complex<float>, psi::DEVICE_GPU>;
template class DiagoCG<std::complex<double>, psi::DEVICE_GPU>;
#endif 
} // namespace hsolver