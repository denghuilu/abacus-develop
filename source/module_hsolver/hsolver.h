#ifndef HSOLVER_H
#define HSOLVER_H

#include <complex>

#include "diagh.h"
#include "module_elecstate/elecstate.h"
#include "module_hamilt_general/hamilt.h"
#include "module_hamilt_pw/hamilt_pwdft/wavefunc.h"
#include "module_hamilt_pw/hamilt_stodft/sto_wf.h"
#include "module_psi/psi.h"
#include <module_base/macros.h>

namespace hsolver
{

template<typename T, typename Device = psi::DEVICE_CPU>
class HSolver
{
  private:
    using R = typename PossibleComplexToReal<T>::type;
  public:
    HSolver(){};
    virtual ~HSolver(){
        delete pdiagh;
    };
    /*//initialization, used in construct function or restruct a new HSolver
    virtual void init(
        const Basis* pbas //We need Basis class here, use global class for this initialization first
        //const Input &in, //We need new Input class here, use global variable for this initialization first
        //elecstate::ElecState *pes
        )=0;
    //initialization, only be called for change some parameters only
    virtual void update(
        Input &in )=0;*/

    // solve Hamiltonian to electronic density in ElecState
    virtual void solve(hamilt::Hamilt<T, Device>* phm,
                       psi::Psi<T, Device>& ppsi,
                       elecstate::ElecState* pes,
                       const std::string method,
                       const bool skip_charge = false)
    {
        return;
    }

    virtual void solve(hamilt::Hamilt<T, Device>* phm,
                       psi::Psi<R, Device>& ppsi,
                       elecstate::ElecState* pes,
                       const std::string method,
                       const bool skip_charge = false)
    {
        return;
    }

    virtual void solve(hamilt::Hamilt<T, Device>* phm,
                       psi::Psi<T, Device>& ppsi,
                       elecstate::ElecState* pes,
                       ModulePW::PW_Basis_K* wfc_basis,
                       Stochastic_WF& stowf,
                       const int istep,
                       const int iter,
                       const std::string method,
                       const bool skip_charge = false)
    {
        return;
    }

    std::string classname = "none";
    // choose method of DiagH for solve Hamiltonian matrix
    // cg, dav, elpa, scalapack-gvx, cusolver
    std::string method = "none";
  public:
    R diag_ethr=0.0; //threshold for diagonalization
    //set diag_ethr according to drho
    //for lcao, we suppose the error is zero and we set diag_ethr to 0
    virtual R set_diagethr(const int istep, const int iter, const R drho)
    {
        return 0.0;
    }
    //reset diag_ethr according to drho and hsolver_error
    virtual R reset_diagethr(std::ofstream& ofs_running, const R hsover_error, const R drho)
    {
        return 0.0;
    }

    // calculate hsolver_error
    // for sdft and lcao, we suppose the error is zero 
    virtual R cal_hsolerror()
    {
        return 0.0;
    };

  protected:
    DiagH<T, Device>* pdiagh = nullptr; // for single Hamiltonian matrix diagonal solver

};

} // namespace hsolver
#endif