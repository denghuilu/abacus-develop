#ifndef HSOLVERPW_H
#define HSOLVERPW_H

#include "hsolver.h"
#include "module_base/macros.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_hamilt_pw/hamilt_pwdft/wavefunc.h"

namespace hsolver {

template<typename T, typename Device = psi::DEVICE_CPU>
class HSolverPW: public HSolver
{
  private:
    // Note GetTypeReal<T>::type will 
    // return T if T is real type(float, double), 
    // otherwise return the real type of T(complex<float>, complex<double>)
    using Real = typename GetTypeReal<T>::type;
  public:
    HSolverPW(ModulePW::PW_Basis_K* wfc_basis_in, wavefunc* pwf_in);

    /*void init(
        const Basis* pbas
        //const Input &in,
    ) override;
    void update(//Input &in
    ) override;*/

    void solve(hamilt::Hamilt* pHamilt,
               ct::Tensor& psi,
               elecstate::ElecState* pes,
               std::string method_in,
               bool skip_charge) override;

    double cal_hsolerror() override;
    double set_diagethr(const int istep, const int iter, const double drho) override;
    double reset_diagethr(std::ofstream& ofs_running, const double hsover_error, const double drho) override;
  protected:
    void initDiagh(const ct::Tensor& psi_in);
    void endDiagh();
    void hamiltSolvePsiK(hamilt::Hamilt* hm, ct::Tensor& psi, Tensor* eigenvalue);

    void updatePsiK(hamilt::Hamilt* pHamilt,
                    ct::Tensor& psi,
                    const int ik);

    ModulePW::PW_Basis_K* wfc_basis = nullptr;
    wavefunc* pwf = nullptr;

    // calculate the precondition array for diagonalization in PW base
    void update_precondition(std::vector<Real> &h_diag, const int ik, const int npw);

    std::vector<Real> precondition;

    bool initialed_psi = false;

    using resmem_var_op = psi::memory::resize_memory_op<Real, psi::DEVICE_CPU>;
    using delmem_var_op = psi::memory::delete_memory_op<Real, psi::DEVICE_CPU>;
    using castmem_2d_2h_op = psi::memory::cast_memory_op<double, Real, psi::DEVICE_CPU, psi::DEVICE_CPU>;
};

} // namespace hsolver

#endif