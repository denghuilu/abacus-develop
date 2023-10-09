#ifndef MODULEHAMILT_H
#define MODULEHAMILT_H

#include "matrixblock.h"
#include "module_psi/psi.h"
#include "operator.h"

#include <complex>
#include <vector>

namespace hamilt
{

class Hamilt
{
  public:
    virtual ~Hamilt() = default;

    /// for target K point, update consequence of hPsi() and matrix()
    virtual void updateHk(const int ik) {}

    /// refresh status of Hamiltonian, for example, refresh H(R) and S(R) in LCAO case
    virtual void refresh() {}

    /// core function: for solving eigenvalues of Hamiltonian with iterative method
    virtual void sPsi(const ct::Tensor& psi_in, ct::Tensor& spsi) const {}

    /// core function: return H(k) and S(k) matrixs for direct solving eigenvalues.
    virtual void matrix(MatrixBlock<std::complex<double>> &hk_in, MatrixBlock<std::complex<double>> &sk_in) {}
    virtual void matrix(MatrixBlock<double> &hk_in, MatrixBlock<double> &sk_in) {}

    // TODO: Use an enum type to specify the hamilt type
    std::string classname = "none";

    /// first node operator, add operations from each operators
    Operator* ops = nullptr;
};

} // namespace hamilt

#endif