#ifndef DIAGCG_H
#define DIAGCG_H

#include "diagh.h"
#include "module_base/complexmatrix.h"
#include "module_hamilt_pw/hamilt_pwdft/structure_factor.h"

#include "module_psi/kernels/types.h"
#include "module_psi/kernels/device.h"
#include "module_psi/kernels/memory_op.h"

#include "module_hsolver/kernels/math_kernel_op.h"
#include <module_base/macros.h>

#include <functional>
namespace hsolver {

template<typename T = std::complex<double>, typename Device = psi::DEVICE_CPU>
class DiagoCG : public DiagH
{
  private:
    // Note GetTypeReal<T>::type will 
    // return T if T is real type(float, double), 
    // otherwise return the real type of T(complex<float>, complex<double>)
    using Real = typename GetTypeReal<T>::type;
  public:
    DiagoCG(
        const std::function<void(const ct::Tensor&, ct::Tensor&)>& hpsi_func,
        const std::function<void(const ct::Tensor&, ct::Tensor&)>& spsi_func);
    // Destructor for DiagoCG
    ~DiagoCG() override = default;

    // virtual void init(){};
    // refactor hpsi_info
    // this is the override function diag() for CG method
    void diag(const ct::Tensor& prec, ct::Tensor& psi, ct::Tensor& eigen_in) override;

  private:
    /// static variables, used for passing control variables
    /// record for how many bands not have convergence eigenvalues
    int notconv_ = 0;
    /// inside variables and vectors, used by inside functions.
    /// row size for input psi matrix
    int n_band_ = 0;
    /// col size for input psi matrix
    int n_basis_ = 0;

    /// A function object that performs the hPsi calculation.
    std::function<void(const ct::Tensor&, ct::Tensor&)> hpsi_func_ = nullptr;
    /// A function object that performs the sPsi calculation.
    std::function<void(const ct::Tensor&, ct::Tensor&)> spsi_func_ = nullptr;

    void calc_grad(const ct::Tensor& prec, 
        ct::Tensor& grad, ct::Tensor& hphi, 
        ct::Tensor& sphi, ct::Tensor& pphi);

    void orth_grad(const ct::Tensor& psi, const int& m, 
        ct::Tensor& grad, ct::Tensor& scg, ct::Tensor& lagrange);

    void calc_gamma_cg(
        const int& iter,
        const Real& cg_norm, 
        const Real& theta,
        const ct::Tensor& prec,
        const ct::Tensor& scg,
        const ct::Tensor& grad,
        const ct::Tensor& phi_m,
        Real& gg_last,
        ct::Tensor& g0,
        ct::Tensor& cg);

    bool update_psi(
        const ct::Tensor& pphi,
        const ct::Tensor& cg,
        const ct::Tensor& scg,
        Real &cg_norm, 
        Real &theta, 
        Real &eigen,
        ct::Tensor& phi_m,
        ct::Tensor& sphi,
        ct::Tensor& hphi);

    void schmit_orth(
        const int& m, 
        const ct::Tensor& psi, 
        const ct::Tensor& sphi, 
        ct::Tensor& phi_m);

    // used in diag() for template replace Hamilt with Hamilt_PW
    void diag_mock(const ct::Tensor& prec, ct::Tensor& psi, ct::Tensor& eigen);
};

} // namespace hsolver
#endif