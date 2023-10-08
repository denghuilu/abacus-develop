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
    // Constructor need:
    // 1. temporary mock of Hamiltonian "Hamilt_PW"
    // 2. precondition pointer should point to place of precondition array.
    DiagoCG(const ct::Tensor& prec_in, const ct::Tensor& n_basis_in);
    // Destructor for DiagoCG
    ~DiagoCG() = default;

    // virtual void init(){};
    // refactor hpsi_info
    // this is the override function diag() for CG method
    void diag(hamilt::Hamilt* phm_in, ct::Tensor& psi, ct::Tensor& eigenvalue_in) override;

  private:
    /// static variables, used for passing control variables
    /// if eigenvalue and eigenvectors should be reordered after diagonalization, it is always be true.
    bool reorder_ = false;
    /// record for how many bands not have convergence eigenvalues
    int notconv_ = 0;

    int test_cg_ = 0;

    /// inside variables and vectors, used by inside functions.
    /// row size for input psi matrix
    int n_band_ = 0;
    /// col size for input psi matrix
    int n_basis_max_ = 0;
    /// non-zero col size for inputted psi matrix
    int current_n_basis_ = 0;

    /// precondition for cg diag
    ct::Tensor prec_ = {}, h_prec_ = {};
    /// eigenvalue results
    ct::Tensor eigen_ = {};

    /// temp vector for new psi for one band, size dim
    ct::Tensor phi_m_ = {};
    /// temp vector for S|psi> for one band, size dim
    ct::Tensor sphi_ = {};
    /// temp vector for H|psi> for one band, size dim
    ct::Tensor hphi_ = {};

    /// temp vector for , size dim
    ct::Tensor cg_ = {};
    /// temp vector for , size dim
    ct::Tensor scg_ = {};
    /// temp vector for store psi in sorting with eigenvalues, size dim
    ct::Tensor pphi_ = {};

    /// temp vector for , size dim
    ct::Tensor gradient_ = {};
    /// temp vector for , size dim
    ct::Tensor g0_ = {};
    /// temp vector for matrix eigenvector * vector S|psi> , size m_band
    ct::Tensor lagrange_ = {};

    ct::DataType r_type_  = ct::DataType::DT_INVALID;
    ct::DataType t_type_  = ct::DataType::DT_INVALID;
    ct::DeviceType device_type_ = ct::DeviceType::UnKnown;

    void calculate_gradient();

    void orthogonal_gradient(hamilt::Hamilt* phm_in, const ct::Tensor& eigenfunction, const int& m);

    void calculate_gamma_cg(const int& iter, Real& gg_last, const Real& cg0, const Real& theta);

    bool update_psi(Real& cg_norm, Real& theta, Real& eigenvalue);

    void schmit_orth(const int& m, const ct::Tensor& psi);

    // used in diag() for template replace Hamilt with Hamilt_PW
    void diag_mock(hamilt::Hamilt* phm_in, ct::Tensor& phi, ct::Tensor& eigenvalue_in);

    using zdot_real_op = hsolver::zdot_real_op<Real, Device>;

    using setmem_complex_op = psi::memory::set_memory_op<T, Device>;
    using delmem_complex_op = psi::memory::delete_memory_op<T, Device>;
    using resmem_complex_op = psi::memory::resize_memory_op<T, Device>;
    using syncmem_complex_op = psi::memory::synchronize_memory_op<T, Device, Device>;
    using syncmem_complex_d2h_op = psi::memory::synchronize_memory_op<T, psi::DEVICE_CPU, Device>;

    using resmem_var_op = psi::memory::resize_memory_op<Real, Device>;
    using setmem_var_h_op = psi::memory::set_memory_op<Real, psi::DEVICE_CPU>;
    using syncmem_var_h2d_op = psi::memory::synchronize_memory_op<Real, Device, psi::DEVICE_CPU>;

    const ct::Tensor one_ = {}, zero_ = {}, neg_one_ = {};
};

} // namespace hsolver
#endif