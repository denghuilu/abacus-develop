#include "module_hsolver/include/dngvd_op.h"


namespace hsolver {

template <>
void dngvd_op<double, psi::DEVICE_GPU>::operator()(
        const psi::DEVICE_GPU* d,
        const int row,
        const int col,
        const std::complex<double>* A,
        const std::complex<double>* B,
        const int m,
        double* W, 
        std::complex<double>* V)
{
    return;
}


}