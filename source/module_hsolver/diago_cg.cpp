#include "module_hsolver/diago_cg.h"

#include "diago_iter_assist.h"
#include "module_base/blas_connector.h"
#include "module_base/constants.h"
#include "module_base/global_function.h"
#include "module_base/memory.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_hamilt_pw/hamilt_pwdft/hamilt_pw.h"
#include "module_hsolver/kernels/math_kernel_op.h"

#include <ATen/core/tensor_map.h>
#include <ATen/core/tensor_utils.h>
#include <ATen/ops/einsum_op.h>

#include "diago_cg.h"

namespace hsolver {

template <typename T, typename Device>
DiagoCG<T, Device>::DiagoCG(
    const std::function<void(const ct::Tensor&, ct::Tensor&)>& hpsi_func,
    const std::function<void(const ct::Tensor&, ct::Tensor&)>& spsi_func)
{
    this->hpsi_func_ = hpsi_func;
    this->spsi_func_ = spsi_func;
}

template<typename T, typename Device>
void DiagoCG<T, Device>::diag_mock(const ct::Tensor& prec, ct::Tensor& psi, ct::Tensor& eigen)
{
    ModuleBase::timer::tick("DiagoCG", "diag_once");
    /// out : record for states of convergence
    this->notconv_ = 0;
    /// initialize variables
    this->n_band_ = psi.shape().dim_size(0);
    this->n_basis_ = psi.shape().dim_size(1);

    /// record for how many loops in cg convergence
    int avg = 0;

    //-------------------------------------------------------------------
    // "poor man" iterative diagonalization of a complex hermitian matrix
    // through preconditioned conjugate grad algorithm
    // Band-by-band algorithm with minimal use of memory
    // Calls hPhi and sPhi to calculate H|phi> and S|phi>
    // Works for generalized eigenvalue problem (US pseudopotentials) as well
    //-------------------------------------------------------------------
    // phi_m = new psi::Psi<T, Device>(phi, 1, 1);
    auto phi_m = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));
    // hphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto hphi = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));
    // sphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto sphi = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));
    // pphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto pphi = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));

    // cg = new psi::Psi<T, Device>(phi, 1, 1);
    auto cg = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));
    // scg.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto scg = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));

    // grad.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto grad = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));
    // g0.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto g0 = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_basis_}));
    // lagrange.resize(this->n_band, ModuleBase::ZERO);
    auto lagrange = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {this->n_band_}));

    ModuleBase::Memory::record("DiagoCG", this->n_basis_ * 9);

    auto eigen_pack = eigen.accessor<Real, 1>();

    for (int m = 0; m < this->n_band_; m++) {
        //copy psi_in into internal psi, m=0 has been done in Constructor
        if (m > 0) {
            phi_m.sync(psi[m]);
        }
        this->spsi_func_(phi_m, sphi); // sphi = S|psi(m)>
        this->schmit_orth(m, psi, sphi, phi_m);
        this->spsi_func_(phi_m, sphi); // sphi = S|psi(m)>
        this->hpsi_func_(phi_m, hphi);
        eigen_pack[m] = ct::extract<Real>(
            ct::op::einsum("i,i->", phi_m, hphi));

        int  iter      = 0;
        Real gg_last   = 0.0;
        Real cg_norm   = 0.0;
        Real theta     = 0.0;
        bool converged = false;
        // start the iterative diagonalization
        do
        {
            this->calc_grad(prec, grad, hphi, sphi, pphi);
            this->orth_grad(psi, m, grad, scg, lagrange);
            this->calc_gamma_cg(
                iter,  // const int&
                cg_norm, theta, // const Real&
                prec, scg, grad, phi_m, // const Tensor&
                gg_last, // Real&
                g0, cg); // Tensor&

            this->hpsi_func_(cg, pphi);
            this->spsi_func_(cg, scg);

            converged = this->update_psi(
                pphi, cg, scg, // const Tensor&
                cg_norm, theta, eigen_pack[m], // Real&
                phi_m, sphi, hphi); // Tensor&

        } while (!converged || ++iter < DiagoIterAssist<T, Device>::PW_DIAG_NMAX);

        // ModuleBase::GlobalFunc::COPYARRAY(phi_m->get_pointer(), psi_temp, this->n_basis_);
        psi[m].sync(phi_m);

        if (!converged) {
            ++this->notconv_;
        }
        avg += iter + 1;

        // reorder eigenvalues if they are not in the right order
        // (this CAN and WILL happen in not-so-special cases)
        if (m > 0) {
            if (eigen_pack[m] - eigen_pack[m - 1] < -2.0 * DiagoIterAssist<T, Device>::PW_DIAG_THR) {
                // if the last calculated eigen_pack is not the largest...
                int ii = 0;
                for (ii = m - 2; ii >= 0; ii--) {
                    if (eigen_pack[m] - eigen_pack[ii] > 2.0 * DiagoIterAssist<T, Device>::PW_DIAG_THR)
                        break;
                }
                ii++;

                // last calculated eigen_pack should be in the i-th position: reorder
                Real e0 = eigen_pack[m];
                pphi.sync(psi[m]);
                for (int jj = m; jj >= ii + 1; jj--)
                {
                    eigen_pack[jj] = eigen_pack[jj - 1];
                    psi[jj].sync(psi[jj - 1]);
                }
                eigen_pack[ii] = e0;
                psi[ii].sync(pphi);
            } // endif
        } // end reorder
    } // end m

    avg /= this->n_band_;
    DiagoIterAssist<T, Device>::avg_iter += avg;
    ModuleBase::timer::tick("DiagoCG", "diag_once");
} // end subroutine ccgdiagg

template <typename T, typename Device>
void DiagoCG<T, Device>::calc_grad(
    const ct::Tensor& prec,
    ct::Tensor& grad,
    ct::Tensor& hphi,
    ct::Tensor& sphi,
    ct::Tensor& pphi)
{
    // precondition |g> = P |scg>
    grad = hphi / prec;
    pphi = sphi / prec;
    // Update lambda !
    // (4) <psi|SPH|psi >
    auto eh = ct::extract<Real>(ct::op::einsum("i,i->", sphi, grad));
    // (5) <psi|SPS|psi >
    auto es = ct::extract<Real>(ct::op::einsum("i,i->", sphi, pphi));
    auto lambda = eh / es;
    // Update g!
    // for (int i = 0; i < this->n_basis_; i++)
    // {
    //     //               <psi|SPH|psi>
    //     // (6) PH|psi> - ------------- * PS |psi>
    //     //               <psi|SPS|psi>
    //     //
    //     // So here we get the grad.
    //     grad[i] -= lambda * this->pphi[i];
    // }
    // TODO: add some possible lazy evaluation ? 
    grad -= lambda * pphi;
}

template<typename T, typename Device>
void DiagoCG<T, Device>::orth_grad(
    const ct::Tensor& psi, 
    const int& m, 
    ct::Tensor& grad, 
    ct::Tensor& scg,
    ct::Tensor& lagrange)
{
    this->spsi_func_(grad, scg); // scg = S|grad>
    ct::EinsumOption option(
        /*conj_x=*/true, /*conj_y=*/false, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&lagrange);
    lagrange = ct::op::einsum("ij,i->j", psi, scg, option);
    // TODO: add Tensor operators for communications
    Parallel_Reduce::reduce_complex_double_pool(lagrange.data<T>(), m);

    // (3) orthogonal |g> and |scg> to all states (0~m-1)
    option = ct::EinsumOption(
        /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/-1.0, /*beta=*/1.0, /*Tensor out=*/&grad);
    grad = ct::op::einsum("ij,j->i", psi, lagrange, option);

    option = ct::EinsumOption(
        /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/-1.0, /*beta=*/1.0, /*Tensor out=*/&scg);
    scg = ct::op::einsum("ij,j->i", psi, lagrange, option);
}

template<typename T, typename Device>
void DiagoCG<T, Device>::calc_gamma_cg(
    const int& iter,
    const Real& cg_norm, 
    const Real& theta,
    const ct::Tensor& prec,
    const ct::Tensor& scg,
    const ct::Tensor& grad,
    const ct::Tensor& phi_m,
    Real& gg_last,
    ct::Tensor& g0,
    ct::Tensor& cg)
{
    // (2) Update for g0!
    // two usage:
    // firstly, for now, calculate: gg_now
    // secondly, prepare for the next iteration: gg_inter
    // |g0> = P | scg >
    // for (int i = 0; i < this->n_basis_; i++)
    // {
    //     g0[i] = this->precondition[i] * scg[i];
    // }
    g0 = scg * prec;    

    // (3) Update gg_now!
    // gg_now = < g|P|scg > = < g|g0 >
    auto gg_now = ct::extract<Real>(ct::op::einsum("i,i->", grad, g0));

    if (iter == 0)
    {
        // (40) gg_last first value : equal gg_now
        gg_last = gg_now;
        // (50) cg direction first value : |g>
        // |cg> = |g>
        cg.sync(grad);
    }
    else
    {
        // (1) Update gg_inter!
        // gg_inter = <g|g0>
        // Attention : the 'g' in g0 is got last time
        auto gg_inter = ct::extract<Real>(ct::op::einsum("i,i->", grad, g0));
        // (4) Update gamma !
        REQUIRES_OK(gg_last != 0.0,
            "DiagoCG::calc_gamma_cg: gg_last is zero, which is not allowed!");
        auto gamma = (gg_now - gg_inter) / gg_last;
        // (5) Update gg_last !
        gg_last = gg_now;
        // (6) Update cg direction !(need gamma and |go> ):
        // for (int i = 0; i < this->n_basis_; i++)
        // {
        //     pcg[i] = gamma * pcg[i] + grad[i];
        // }
        cg = gamma * cg + grad;
        auto normal = gamma * cg_norm * sin(theta);
        // zaxpy_(&this->n_basis_, &znorma, pphi_m, &one, pcg, &one);
        /*for (int i = 0; i < this->n_basis_; i++)
        {
            pcg[i] -= norma * pphi_m[i];
        }*/
        cg += normal * phi_m;
    }
}

template<typename T, typename Device>
bool DiagoCG<T, Device>::update_psi(
    const ct::Tensor& pphi,
    const ct::Tensor& cg,
    const ct::Tensor& scg,
    Real &cg_norm, 
    Real &theta, 
    Real &eigen,
    ct::Tensor& phi_m,
    ct::Tensor& sphi,
    ct::Tensor& hphi)
{
    cg_norm = sqrt(ct::extract<Real>(
        ct::op::einsum("i,i->", cg, scg)));

    if (cg_norm < 1.0e-10)
        return true;

    const Real a0 = ct::extract<Real>(
        ct::op::einsum("i,i->", phi_m, pphi)) * 2.0 / cg_norm;
    
    const Real b0 = ct::extract<Real>(
        ct::op::einsum("i,i->", cg, pphi)) / (cg_norm * cg_norm);

    const Real e0 = eigen;
    theta = atan(a0 / (e0 - b0)) / 2.0;

    const Real new_e = (e0 - b0) * cos(2.0 * theta) + a0 * sin(2.0 * theta);

    const Real e1 = (e0 + b0 + new_e) / 2.0;
    const Real e2 = (e0 + b0 - new_e) / 2.0;

    if (e1 > e2) {
        theta += ModuleBase::PI_HALF;
    }

    eigen = std::min(e1, e2);
    //	OUT("eigenvalue",eigenvalue);

    const Real cost = cos(theta);
    const Real sint_norm = sin(theta) / cg_norm;

    // for (int i = 0; i < this->n_basis_; i++)
    // {
    //     phi_m_pointer[i] = phi_m_pointer[i] * cost + sint_norm * pcg[i];
    // }
    phi_m = phi_m * cost + sint_norm * cg;    

    if (std::abs(eigen - e0) < DiagoIterAssist<T, Device>::PW_DIAG_THR) {
        return true;
    }
    else {
        // for (int i = 0; i < this->n_basis_; i++)
        // {
        //     sphi[i] = sphi[i] * cost + sint_norm * scg[i];
        //     hphi[i] = hphi[i] * cost + sint_norm * this->pphi[i];
        // }
        sphi = sphi * cost + sint_norm * scg;
        hphi = hphi * cost + sint_norm * pphi;
        return false;
    }
}

template<typename T, typename Device>
void DiagoCG<T, Device>::schmit_orth(
    const int& m, 
    const ct::Tensor& psi, 
    const ct::Tensor& sphi, 
    ct::Tensor& phi_m)
{
    // orthogonalize starting eigenfunction to those already calculated
    // phi_m orthogonalize to psi(start) ~ psi(m-1)
    // Attention, the orthogonalize here read as
    // psi(m) -> psi(m) - \sum_{i < m} < psi(i) | S | psi(m) > psi(i)
    // so the orthogonalize is performed about S.
    REQUIRES_OK(m >= 0,
        "DiagoCG::schmit_orth: m < 0");
    REQUIRES_OK(this->n_band_ >= m,
        "DiagoCG::schmit_orth: n_band < m");

    ct::Tensor lagrange_so = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, {m + 1});

    auto psi_map = ct::TensorMap(
        psi.data<T>(), psi.data_type(), psi.device_type(), {m + 1, this->n_basis_});

    ct::EinsumOption option(
        /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&lagrange_so);
    lagrange_so = ct::op::einsum("i,ji->j", psi_map, sphi, option);

    // be careful , here reduce m+1
    Parallel_Reduce::reduce_complex_double_pool(lagrange_so.data<T>(), m + 1);

    psi_map = ct::TensorMap(
        psi.data<T>(), psi.data_type(), psi.device_type(), {m, this->n_basis_});
    option = ct::EinsumOption(
        /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/-1.0, /*beta=*/1.0, /*Tensor out=*/&phi_m);
    phi_m = ct::op::einsum("ij,i->j", psi_map, lagrange_so, option);

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
    auto psi_norm = ct::extract<Real>(
        lagrange_so[m] - ct::op::einsum("i,i->", lagrange_so, lagrange_so));

    REQUIRES_OK(psi_norm > 0.0,
        "DiagoCG::schmit_orth: psi_norm < 0");

    psi_norm = sqrt(psi_norm);

    // scal_op<Real, Device>()(this->ctx_, this->n_basis_, &psi_norm, pphi_m, 1);
    //======================================================================
    // for (int ig = 0; ig < this->n_basis_; ig++)
    // {
    //     pphi_m[ig] /= psi_norm;
    // }
    phi_m /= psi_norm;
}

template<typename T, typename Device>
void DiagoCG<T, Device>::diag(const ct::Tensor& prec, ct::Tensor& psi, ct::Tensor& eigen_in)
{
    /// record the times of trying iterative diagonalization
    int ntry = 0;
    this->notconv_ = 0;
    do
    {
        if(DiagoIterAssist<T, Device>::need_subspace || ntry > 0)
        {
            DiagoIterAssist<T, Device>::diagH_subspace(hpsi_func_, psi, psi, eigen_in, this->ik_);
        }

        DiagoIterAssist<T, Device>::avg_iter += 1.0;

        this->diag_mock(prec, psi, eigen_in);

        ++ntry;
    } while (DiagoIterAssist<T, Device>::test_exit_cond(ntry, this->notconv_));

    if (notconv_ > std::max(5, this->n_band_ / 4)) {
        std::cout << "\n notconv = " << this->notconv_;
        std::cout << "\n DiagoCG::diag', too many bands are not converged! \n";
    }
}

template class DiagoCG<std::complex<float>, ct::DEVICE_CPU>;
template class DiagoCG<std::complex<double>, ct::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoCG<std::complex<float>, ct::DEVICE_GPU>;
template class DiagoCG<std::complex<double>, ct::DEVICE_GPU>;
#endif 
} // namespace hsolver