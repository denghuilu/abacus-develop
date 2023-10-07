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

class DiagH
{
  public:
    virtual ~DiagH() = default;
    // virtual void init()=0;
    // TODO: Use an enum type to specify the type of DiagH
    std::string method = "none";

    virtual void diag(hamilt::Hamilt* phm_in, ct::Tensor& psi, ct::Tensor* eigenvalue_in) = 0;
};

} // namespace hsolver

#endif