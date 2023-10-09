#ifndef DIAGH_H
#define DIAGH_H

#include "module_base/complexmatrix.h"
#include "module_hamilt_general/hamilt.h"
#include "module_psi/psi.h"
#include "string"
 
#include <module_base/macros.h>
#include <ATen/core/tensor.h>

namespace hsolver
{

enum DiagH_Type {
    DiagH_None,
    DiagH_CG,
    DiagH_BPCG,
    DiagH_DAV
};

class DiagH
{
  public:
    virtual ~DiagH() = default;
    // virtual void init()=0;
    DiagH_Type method_ = DiagH_Type::DiagH_None;

    // New interface with container Tensor
    virtual void diag(hamilt::Hamilt* phm_in, ct::Tensor& psi, ct::Tensor& eigenvalue_in) {};
    // Deprecated interface, will be removed in the future
    virtual void diag(hamilt::Hamilt* phm_in, psi::Psi<double>& psi, double* eigenvalue_in) {};
    virtual void diag(hamilt::Hamilt* phm_in, psi::Psi<std::complex<double>>& psi, double* eigenvalue_in) {};
  
  protected:
    // TODO: Remove ik
    // specify current k-point index
    int ik_ = 0;
    // Array which stores the basis number of all k-points
    ct::Tensor n_basis_ = {};
};

} // namespace hsolver

#endif