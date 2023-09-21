#ifndef HAMILTPW_H
#define HAMILTPW_H

#include "module_cell/klist.h"
#include "module_elecstate/potentials/potential_new.h"
#include "module_hamilt_general/hamilt.h"

#include <module_base/macros.h>

namespace hamilt
{

template<typename T, typename Device = psi::DEVICE_CPU>
class HamiltPW : public Hamilt<T, Device>
{
  private:
    // Note GetTypeReal<T>::type will 
    // return T if T is real type(float, double), 
    // otherwise return the real type of T(complex<float>, complex<double>)
    using Real = typename GetTypeReal<T>::type;
  public:
    HamiltPW(elecstate::Potential* pot_in, ModulePW::PW_Basis_K* wfc_basis, K_Vectors* p_kv);
    template<typename T_in, typename Device_in = Device>
    explicit HamiltPW(const HamiltPW<T_in, Device_in>* hamilt);
    ~HamiltPW();

    // for target K point, update consequence of hPsi() and matrix()
    void updateHk(const int ik) override;

    // core function: for solving eigenvalues of Hamiltonian with iterative method
    virtual void sPsi(const T *psi_in, T *spsi, const size_t size) const override;

  private:

    Device *ctx = {};
    using syncmem_complex_op = psi::memory::synchronize_memory_op<T, Device, Device>;
};

} // namespace hamilt

#endif