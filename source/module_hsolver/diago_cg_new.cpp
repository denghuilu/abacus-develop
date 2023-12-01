#include <module_hsolver/diago_cg_new.h>
#include <module_hsolver/diago_iter_assist.h>

#include <module_base/memory.h>
#include <module_base/parallel_reduce.h>
#include <module_base/timer.h>
#include <module_hamilt_pw/hamilt_pwdft/hamilt_pw.h>

#include <ATen/kernels/lapack.h>
#include <ATen/kernels/memory.h>
#include <ATen/core/tensor_map.h>
#include <ATen/core/tensor_utils.h>

#include <ATen/ops/einsum_op.h>
#include <ATen/ops/linalg_op.h>
#include <module_basis/module_pw/test/pw_test.h>

namespace hsolver {

template<typename T, typename Device>
DiagoCG_New<T, Device>::DiagoCG_New(
    const std::string& basis_type,
    const std::string& calculation)
{
    basis_type_ = basis_type;
    calculation_ = calculation;
    this->one_ = &this->cs_.one;
    this->zero_ = &this->cs_.zero;
    this->neg_one_ = &this->cs_.neg_one;
}

template<typename T, typename Device>
DiagoCG_New<T, Device>::DiagoCG_New(
    const std::string& basis_type,
    const std::string& calculation,
    const bool& need_subspace,
    const Real& pw_diag_thr,
    const int& pw_diag_nmax,
    const int& nproc_in_pool)
{
    basis_type_ = basis_type;
    calculation_ = calculation;
    pw_diag_thr_ = pw_diag_thr;
    pw_diag_nmax_ = pw_diag_nmax;
    need_subspace_ = need_subspace;
    nproc_in_pool_ = nproc_in_pool;
    this->one_ = &this->cs_.one;
    this->zero_ = &this->cs_.zero;
    this->neg_one_ = &this->cs_.neg_one;
}

template<typename T, typename Device>
void DiagoCG_New<T, Device>::diag_mock(const ct::Tensor& prec, ct::Tensor& psi, ct::Tensor& eigen)
{
    ModuleBase::timer::tick("DiagoCG_New", "diag_once");
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
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));
    // hphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto hphi = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));
    // sphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto sphi = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));
    // pphi.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto pphi = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));

    // cg = new psi::Psi<T, Device>(phi, 1, 1);
    auto cg = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));
    // scg.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto scg = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));

    // grad.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto grad = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));
    // g0.resize(this->n_basis_max_, ModuleBase::ZERO);
    auto g0 = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_basis_}));
    // lagrange.resize(this->n_band, ModuleBase::ZERO);
    auto lagrange = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {this->n_band_}));

    ModuleBase::Memory::record("DiagoCG_New", this->n_basis_ * 9);

    auto eigen_pack = eigen.accessor<Real, 1>();

    for (int m = 0; m < this->n_band_; m++) {
        phi_m.sync(psi[m]);

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

        } while (!converged && ++iter < pw_diag_nmax_);

        // ModuleBase::GlobalFunc::COPYARRAY(phi_m->get_pointer(), psi_temp, this->n_basis_);
        psi[m].sync(phi_m);

        if (!converged) {
            ++this->notconv_;
        }
        avg += iter + 1;

        // reorder eigenvalues if they are not in the right order
        // (this CAN and WILL happen in not-so-special cases)
        if (m > 0) {
            if (eigen_pack[m] - eigen_pack[m - 1] < -2.0 * pw_diag_thr_) {
                // if the last calculated eigen_pack is not the largest...
                int ii = 0;
                for (ii = m - 2; ii >= 0; ii--) {
                    if (eigen_pack[m] - eigen_pack[ii] > 2.0 * pw_diag_thr_)
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
    avg_iter_ += avg;
    ModuleBase::timer::tick("DiagoCG_New", "diag_once");
} // end subroutine ccgdiagg

template <typename T, typename Device>
void DiagoCG_New<T, Device>::calc_grad(
    const ct::Tensor& prec,
    ct::Tensor& grad,
    ct::Tensor& hphi,
    ct::Tensor& sphi,
    ct::Tensor& pphi) const
{
    ModuleBase::timer::tick("DiagoCG_New", "calc_grad");
    // precondition |g> = P |scg>
    // grad = hphi / prec;
    // pphi = sphi / prec;
    vector_div_vector_op<T, Device>()(ctx_, this->n_basis_, grad.data<T>(), hphi.data<T>(), prec.data<Real>());
    vector_div_vector_op<T, Device>()(ctx_, this->n_basis_, pphi.data<T>(), sphi.data<T>(), prec.data<Real>());

    // Update lambda !
    // (4) <psi|SPH|psi >
    // auto eh = ct::extract<Real>(ct::op::einsum("i,i->", sphi, grad));
    auto eh = dot_real_op<T, Device>()(ctx_, this->n_basis_, sphi.data<T>(), grad.data<T>());

    // (5) <psi|SPS|psi >
    // auto es = ct::extract<Real>(ct::op::einsum("i,i->", sphi, pphi));
    auto es = dot_real_op<T, Device>()(ctx_, this->n_basis_, sphi.data<T>(), pphi.data<T>());
    auto lambda = eh / es;
    // Update g!
    // grad = - lambda * pphi + 1 * grad;
    // ct::op::add_op()(
    //     static_cast<T>(-lambda), pphi, static_cast<T>(1.0), grad, grad);
    constantvector_addORsub_constantVector_op<T, Device>()(
        ctx_, this->n_basis_, grad.data<T>(), grad.data<T>(), 1.0, pphi.data<T>(), (-lambda));
    ModuleBase::timer::tick("DiagoCG_New", "calc_grad");
}

template<typename T, typename Device>
void DiagoCG_New<T, Device>::orth_grad(
    const ct::Tensor& psi, 
    const int& m, 
    ct::Tensor& grad, 
    ct::Tensor& scg,
    ct::Tensor& lagrange)
{
    ModuleBase::timer::tick("DiagoCG_New", "orth_grad");
    this->spsi_func_(grad, scg); // scg = S|grad>
    gemv_op<T, Device>()(
        ctx_,
        'C',
        n_basis_,
        m,
        one_,
        psi.data<T>(),
        n_basis_,
        scg.data<T>(),
        1,
        zero_,
        lagrange.data<T>(),
        1);

    Parallel_Reduce::reduce_pool(lagrange.data<T>(), m);

    // (3) orthogonal |g> and |scg> to all states (0~m-1)
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // haozhihan replace 2022-10-07
    gemv_op<T, Device>()(
        ctx_,
        'N',
        n_basis_,
        m,
        neg_one_,
        psi.data<T>(),
        n_basis_,
        lagrange.data<T>(),
        1,
        one_,
        grad.data<T>(),
        1);

    gemv_op<T, Device>()(
        ctx_,
        'N',
        n_basis_,
        m,
        neg_one_,
        psi.data<T>(),
        n_basis_,
        lagrange.data<T>(),
        1,
        one_,
        scg.data<T>(),
        1);
    ModuleBase::timer::tick("DiagoCG_New", "orth_grad");
}

template<typename T, typename Device>
void DiagoCG_New<T, Device>::calc_gamma_cg(
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
    ModuleBase::timer::tick("DiagoCG_New", "calc_gamma_cg");
    // (1) Update gg_inter!
    // gg_inter = <g|g0>
    // Attention : the 'g' in g0 is got last time
    Real gg_inter = 0.0;
    if (iter > 0) {
        gg_inter = dot_real_op<T, Device>()(ctx_, n_basis_, grad.data<T>(), g0.data<T>());
    }
    // (2) Update for g0!
    // two usage:
    // firstly, for now, calculate: gg_now
    // secondly, prepare for the next iteration: gg_inter
    // |g0> = P | scg >
    // g0 = scg * prec;
    vector_mul_vector_op<T, Device>()(ctx_, n_basis_, g0.data<T>(), scg.data<T>(), prec.data<Real>());

    // (3) Update gg_now!
    // gg_now = < g|P|scg > = < g|g0 >
    auto gg_now = dot_real_op<T, Device>()(ctx_, n_basis_, grad.data<T>(), g0.data<T>());

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
        // (4) Update gamma !
        REQUIRES_OK(gg_last != 0.0,
            "DiagoCG_New::calc_gamma_cg: gg_last is zero, which is not allowed!");
        auto gamma = (gg_now - gg_inter) / gg_last;
        // (5) Update gg_last !
        gg_last = gg_now;
        // (6) Update cg direction !(need gamma and |go> ):
        // cg = gamma * cg + grad;
        constantvector_addORsub_constantVector_op<T, Device>()(
            ctx_, n_basis_, cg.data<T>(), cg.data<T>(), gamma, grad.data<T>(), 1.0);
        T normal = -gamma * cg_norm * sin(theta);
        // cg -= normal * phi_m;
        axpy_op<T, Device>()(ctx_, n_basis_, &normal, phi_m.data<T>(), 1, cg.data<T>(), 1);
    }
    ModuleBase::timer::tick("DiagoCG_New", "calc_gamma_cg");
}

template<typename T, typename Device>
bool DiagoCG_New<T, Device>::update_psi(
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
    ModuleBase::timer::tick("DiagoCG_New", "update_psi");
    cg_norm = sqrt(dot_real_op<T, Device>()(
        ctx_, n_basis_, cg.data<T>(), scg.data<T>()));

    if (cg_norm < 1.0e-10) {
        return true;
    }
    const Real a0 = dot_real_op<T, Device>()(
        ctx_, n_basis_, phi_m.data<T>(), pphi.data<T>()) * 2.0 / cg_norm;
    
    const Real b0 = dot_real_op<T, Device>()(
        ctx_, n_basis_, cg.data<T>(), pphi.data<T>()) / (cg_norm * cg_norm);

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

    // phi_m = phi_m * cost + cg * sint_norm;
    constantvector_addORsub_constantVector_op<T, Device>()(
        ctx_, n_basis_, phi_m.data<T>(), phi_m.data<T>(), cost, cg.data<T>(), sint_norm);

    if (std::abs(eigen - e0) < pw_diag_thr_) {
        ModuleBase::timer::tick("DiagoCG_New", "update_psi");
        return true;
    }
    // sphi = sphi * cost + scg * sint_norm;
    constantvector_addORsub_constantVector_op<T, Device>()(
        ctx_, n_basis_, sphi.data<T>(), sphi.data<T>(), cost, scg.data<T>(), sint_norm);
    // hphi = hphi * cost + pphi * sint_norm;
    constantvector_addORsub_constantVector_op<T, Device>()(
        ctx_, n_basis_, hphi.data<T>(), hphi.data<T>(), cost, pphi.data<T>(), sint_norm);
    
    ModuleBase::timer::tick("DiagoCG_New", "update_psi");
    return false;
}

template<typename T, typename Device>
void DiagoCG_New<T, Device>::schmit_orth(
    const int& m, 
    const ct::Tensor& psi, 
    const ct::Tensor& sphi, 
    ct::Tensor& phi_m)
{
    ModuleBase::timer::tick("DiagoCG_New", "schmit_orth");
    // orthogonalize starting eigenfunction to those already calculated
    // phi_m orthogonalize to psi(start) ~ psi(m-1)
    // Attention, the orthogonalize here read as
    // psi(m) -> psi(m) - \sum_{i < m} < psi(i) | S | psi(m) > psi(i)
    // so the orthogonalize is performed about S.
    REQUIRES_OK(m >= 0,
        "DiagoCG_New::schmit_orth: m < 0");
    REQUIRES_OK(this->n_band_ >= m,
        "DiagoCG_New::schmit_orth: n_band < m");

    ct::Tensor lagrange_so = ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {m + 1});

    gemv_op<T, Device>()(
        ctx_,
        'C',
        n_basis_,
        m + 1,
        one_,
        psi.data<T>(),
        n_basis_,
        sphi.data<T>(),
        1,
        zero_,
        lagrange_so.data<T>(),
        1);
    // be careful , here reduce m+1
    // TODO: implement a device independent reduce_pool
    Parallel_Reduce::reduce_pool(lagrange_so.data<T>(), m + 1);

    gemv_op<T, Device>()(
        ctx_,
        'N',
        n_basis_,
        m,
        neg_one_,
        psi.data<T>(),
        n_basis_,
        lagrange_so.data<T>(),
        1,
        one_,
        phi_m.data<T>(),
        1);

    auto psi_norm = ct::extract<Real>(lagrange_so[m])
        - dot_real_op<T, Device>()(ctx_, m, lagrange_so.data<T>(), lagrange_so.data<T>(), false);

    REQUIRES_OK(psi_norm > 0.0,
        "DiagoCG_New::schmit_orth: psi_norm < 0")

    psi_norm = sqrt(psi_norm);
    vector_div_constant_op<T, Device>()(ctx_, n_basis_, phi_m.data<T>(), phi_m.data<T>(), psi_norm);
    ModuleBase::timer::tick("DiagoCG_New", "schmit_orth");
}

template<typename T, typename Device>
int DiagoCG_New<T, Device>::get_avg_iter() const
{
    return this->avg_iter_;
}

template <typename T, typename Device>
void DiagoCG_New<T, Device>::diagH_subspace(
    const Func& hpsi_func,
    const Func& spsi_func,
    ct::Tensor& psi,
    ct::Tensor& eigen) const
{
    ModuleBase::timer::tick("DiagoCGNew", "diagH_subspace");
    ct::Tensor hpsi = ct::Tensor(psi.data_type(), psi.device_type(), psi.shape());
    ct::Tensor spsi = ct::Tensor(psi.data_type(), psi.device_type(), psi.shape());
    // do the spsi and hpsi calculation
    spsi_func(psi, spsi);
    hpsi_func(psi, hpsi);

    // calc the hsub matrix
    ct::EinsumOption option(
    /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/nullptr);
    ct::Tensor hsub = ct::op::einsum("ij,kj->ik", psi, hpsi, option);
    ct::Tensor ssub = ct::op::einsum("ij,kj->ik", psi, spsi, option);

    if(nproc_in_pool_ > 1) {
        Parallel_Reduce::reduce_pool(hsub.data<T>(), hsub.NumElements());
        Parallel_Reduce::reduce_pool(ssub.data<T>(), ssub.NumElements());
    }
    ct::Tensor eigen_ = eigen.to_device<ct_Device>();
    ct::kernels::lapack_dngvd<T, ct_Device>()(
        1, 'V', 'U', hsub.data<T>(), ssub.data<T>(), hsub.shape().dim_size(0), eigen_.data<Real>());

    if ((basis_type_ == "lcao" || basis_type_ == "lcao_in_pw") && calculation_ == "nscf")
    {
        GlobalV::ofs_running << " Not do zgemm to get evc." << std::endl;
    }
    else {
        option = ct::EinsumOption(
            /*conj_x=*/false, /*conj_y=*/false, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/nullptr);
        const ct::Tensor evc = ct::op::einsum("ij,jk->ik", hsub, psi, option);
        psi.sync(evc);
    }
    ModuleBase::timer::tick("DiagoCGNew", "diagH_subspace");
}

template<typename T, typename Device>
bool DiagoCG_New<T, Device>::test_exit_cond(
    const int& ntry,
    const int& notconv) const
{
    const bool scf = calculation_ != "nscf";
    // If ntry <=5, try to do it better, if ntry > 5, exit.
    const bool f1 = ntry <= 5;
    // In non-self consistent calculation, do until totally converged.
    const bool f2 = !scf && notconv > 0;
    // if self consistent calculation, if not converged > 5,
    // using diagH_subspace and cg method again. ntry++
    const bool f3 = scf && notconv > 5;
    return f1 && (f2 || f3);
}

template<typename T, typename Device>
void DiagoCG_New<T, Device>::diag(
    hamilt::Hamilt<T, Device>* hm,
    const Func& hpsi_func, 
    const Func& spsi_func, 
    ct::Tensor& psi,
    ct::Tensor& eigen,
    const ct::Tensor& prec)
{
    /// record the times of trying iterative diagonalization
    int ntry = 0;
    this->notconv_ = 0;
    this->hpsi_func_ = hpsi_func;
    this->spsi_func_ = spsi_func;
    do
    {
        if (need_subspace_ || ntry > 0) {
            // this->diagH_subspace(hpsi_func, spsi_func, psi, eigen);
            auto psi_wrapper = psi::Psi<T, Device>(
                psi.data<T>(), 1, 
                psi.shape().dim_size(0), 
                psi.shape().dim_size(1));
            DiagoIterAssist<T, Device>::diagH_subspace(hm, psi_wrapper, psi_wrapper, eigen.data<Real>());
        }
        ++ntry;
        avg_iter_ += 1.0;
        this->diag_mock(prec, psi, eigen);

    } while (this->test_exit_cond(ntry, this->notconv_));

    if (notconv_ > std::max(5, this->n_band_ / 4)) {
        std::cout << "\n notconv = " << this->notconv_;
        std::cout << "\n DiagoCG_New::diag', too many bands are not converged! \n";
    }
}

template class DiagoCG_New<std::complex<float>, psi::DEVICE_CPU>;
template class DiagoCG_New<std::complex<double>, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoCG_New<std::complex<float>, psi::DEVICE_GPU>;
template class DiagoCG_New<std::complex<double>, psi::DEVICE_GPU>;
#endif 
} // namespace hsolver