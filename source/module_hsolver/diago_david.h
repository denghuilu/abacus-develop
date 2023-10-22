//==========================================================
// AUTHOR : wangjp
// Data :2009-04
// Last Update:
//
// 09-05-10 modify SchmitOrth() diag_zhegvx() as static
// member function
//==========================================================

#ifndef DIAGODAVID_H
#define DIAGODAVID_H

#include "diagh.h"
#include "module_base/complexmatrix.h"
#include "module_base/macros.h"
#include "module_hamilt_pw/hamilt_pwdft/structure_factor.h"
#include "module_psi/kernels/device.h"

template<typename T> struct consts
{
    consts();
    T zero;
    T one;
    T neg_one;
};
namespace hsolver
{

template <typename T = std::complex<double>, typename Device = psi::DEVICE_CPU> 
class DiagoDavid : public DiagH
{
  private:
    // Note GetTypeReal<T>::type will 
    // return T if T is real type(float, double), 
    // otherwise return the real type of T(complex<float>, complex<double>)
    using Real = typename GetTypeReal<T>::type;
  public:
    DiagoDavid(const ct::Tensor& prec_in, const ct::Tensor& n_basis_in);
    ~DiagoDavid() = default;

    // this is the override function diag() for CG method
    void diag(hamilt::Hamilt* phm_in, ct::Tensor& psi, ct::Tensor& eigen_in) override;

    static int PW_DIAG_NDIM;

  private:
    int test_david_ = 0;

    /// record for how many bands not have convergence eigenvalues
    int notconv_ = 0;
    /// row size for input psi matrix
    int n_band_ = 0;
    /// col size for input psi matrix
    int n_basis_max_ = 0;
    /// non-zero col size for inputted psi matrix
    int current_n_basis_ = 0;
    // maximum dimension of the reduced basis set
    int n_base_x_ = 0;
    /// precondition for cg diag
    ct::Tensor prec_ = {}, h_prec_ = {};
    /// eigenvalue results
    ct::Tensor eigen_ = {};

    ct::Tensor hphi_ = {}; // the product of H and psi in the reduced basis set

    ct::Tensor sphi_ = {}; // the Product of S and psi in the reduced basis set

    ct::Tensor hcc_ = {}; // Hamiltonian on the reduced basis

    ct::Tensor scc_ = {}; // Overlap on the reduced basis

    ct::Tensor vcc_ = {}; // Eigenvectors of hc

    ct::Tensor lagrange_matrix_ = {};

    void cal_grad(hamilt::Hamilt* phm_in,
                  const int& current_n_basis,
                  const int& nbase,
                  const int& notconv,
                  ct::Tensor& basis,
                  ct::Tensor& hphi,
                  ct::Tensor& sphi,
                  const ct::Tensor& vcc,
                  const std::vector<int>& unconv,
                  const ct::Tensor& eigenvalue);

    void cal_elem(const int& current_n_basis,
                  int& nbase,
                  const int& notconv,
                  const ct::Tensor& basis,
                  const ct::Tensor& hphi,
                  const ct::Tensor& sphi,
                  ct::Tensor& hcc,
                  ct::Tensor& scc);

    void refresh(const int& current_n_basis,
                 const int& nband,
                 int& nbase,
                 const ct::Tensor& eigenvalue,
                 const ct::Tensor& psi,
                 ct::Tensor& basis,
                 ct::Tensor& hphi,
                 ct::Tensor& sphi,
                 ct::Tensor& hcc,
                 ct::Tensor& scc,
                 ct::Tensor& vcc);

    void SchmitOrth(const int& current_n_basis,
                    const int& nband,
                    const int& m,
                    ct::Tensor& basis,
                    const ct::Tensor& sphi,
                    ct::Tensor& lagrange_m,
                    const int& mm_size,
                    const int& mv_size);

    void planSchmitOrth(const int& nband, std::vector<int>& pre_matrix_mm_m, std::vector<int>& pre_matrix_mv_m);

    void diag_zhegvx(const int& nbase,
                     const int& nband,
                     const ct::Tensor& hcc,
                     const ct::Tensor& scc,
                     const int& nbase_x,
                     ct::Tensor& eigenvalue,
                     ct::Tensor& vcc);

    void diag_mock(hamilt::Hamilt* phm_in,
                   ct::Tensor& psi,
                   ct::Tensor& eigen_in);

    using resmem_complex_op = psi::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = psi::memory::delete_memory_op<T, Device>;
    using setmem_complex_op = psi::memory::set_memory_op<T, Device>;
    using resmem_var_op = psi::memory::resize_memory_op<Real, Device>;
    using delmem_var_op = psi::memory::delete_memory_op<Real, Device>;
    using setmem_var_op = psi::memory::set_memory_op<Real, Device>;

    using syncmem_var_h2d_op = psi::memory::synchronize_memory_op<Real, Device, psi::DEVICE_CPU>;
    using syncmem_var_d2h_op = psi::memory::synchronize_memory_op<Real, psi::DEVICE_CPU, Device>;
    using syncmem_complex_op = psi::memory::synchronize_memory_op<T, Device, Device>;
    using castmem_complex_op = psi::memory::cast_memory_op<std::complex<double>, T, Device, Device>;
    using syncmem_h2d_op = psi::memory::synchronize_memory_op<T, Device, psi::DEVICE_CPU>;
    using syncmem_d2h_op = psi::memory::synchronize_memory_op<T, psi::DEVICE_CPU, Device>;

    ct::Tensor one_ = {}, zero_ = {}, neg_one_ = {};
    // 
    Device* ctx_ = {};
    psi::DEVICE_CPU* cpu_ctx_ = {};
    psi::AbacusDevice_t device = {};
};
template <typename Real, typename Device> int DiagoDavid<Real, Device>::PW_DIAG_NDIM = 4;
} // namespace hsolver

#endif
