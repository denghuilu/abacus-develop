#ifndef HSOLVERPW_H
#define HSOLVERPW_H

#include "hsolver.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_hamilt_pw/hamilt_pwdft/wavefunc.h"

#include <module_base/macros.h>

namespace hsolver {

template<typename T, typename Device = psi::DEVICE_CPU>
class HSolverPW: public HSolver<T, Device>
{
  private:
    // Note PossibleComplexToReal<T>::type will 
    // return T if T is real type(float, double), 
    // otherwise return the real type of T(complex<float>, complex<double>)
    using R = typename PossibleComplexToReal<T>::type;
  public:
    HSolverPW(ModulePW::PW_Basis_K* wfc_basis_in, wavefunc* pwf_in);

    /*void init(
        const Basis* pbas
        //const Input &in,
    ) override;
    void update(//Input &in
    ) override;*/

    void solve(hamilt::Hamilt<R, Device>* pHamilt,
               psi::Psi<T, Device>& psi,
               elecstate::ElecState* pes,
               const std::string method_in,
               const bool skip_charge) override;

    virtual R cal_hsolerror() override;
    virtual R set_diagethr(const int istep, const int iter, const R drho) override;
    virtual R reset_diagethr(std::ofstream& ofs_running, const R hsover_error, const R drho) override;
  protected:
    void initDiagh(const psi::Psi<T, Device>& psi_in);
    void endDiagh();
    void hamiltSolvePsiK(hamilt::Hamilt<R, Device>* hm, psi::Psi<T, Device>& psi, R* eigenvalue);

    void updatePsiK(hamilt::Hamilt<R, Device>* pHamilt,
                    psi::Psi<T, Device>& psi,
                    const int ik);

    ModulePW::PW_Basis_K* wfc_basis = nullptr;
    wavefunc* pwf = nullptr;

    // calculate the precondition array for diagonalization in PW base
    void update_precondition(std::vector<R> &h_diag, const int ik, const int npw);

    std::vector<R> precondition;

    bool initialed_psi = false;

    Device * ctx = {};
    using resmem_var_op = psi::memory::resize_memory_op<R, psi::DEVICE_CPU>;
    using delmem_var_op = psi::memory::delete_memory_op<R, psi::DEVICE_CPU>;
    using castmem_2d_2h_op = psi::memory::cast_memory_op<double, R, psi::DEVICE_CPU, psi::DEVICE_CPU>;
    
};

} // namespace hsolver

#endif