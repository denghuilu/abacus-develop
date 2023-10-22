#ifndef HAMILTPW_H
#define HAMILTPW_H

#include "module_base/macros.h"
#include "module_cell/klist.h"
#include "module_elecstate/potentials/potential_new.h"
#include "module_hamilt_general/hamilt.h"

namespace hamilt
{

template<typename T, typename Device = psi::DEVICE_CPU>
class HamiltPW : public Hamilt
{
  private:
    // Note GetTypeReal<T>::type will 
    // return T if T is real type(float, double), 
    // otherwise return the real type of T(complex<float>, complex<double>)
    using Real = typename GetTypeReal<T>::type;
  public:
    HamiltPW(elecstate::Potential* pot_in, ModulePW::PW_Basis_K* wfc_basis, K_Vectors* p_kv);
    ~HamiltPW() override;

    // for target K point, update consequence of hPsi() and matrix()
    void updateHk(int ik) override;

    // core function: for solving eigenvalues of Hamiltonian with iterative method
    void sPsi(const ct::Tensor& psi_in, ct::Tensor& spsi) const override;
};

} // namespace hamilt

#endif