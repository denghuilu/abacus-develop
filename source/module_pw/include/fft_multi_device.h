#ifndef MODULE_PW_FFT_MULTI_DEVICE_H
#define MODULE_PW_FFT_MULTI_DEVICE_H

#include "module_psi/psi.h"
#include <complex>

namespace ModulePW {
template <typename FPTYPE, typename Device>
struct set_box_op {
    void operator() (
        const Device* dev,
        const int npwk,
        const FPTYPE* in,
        FPTYPE* out,
        const int* box_index);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
// Partially specialize functor for psi::GpuDevice.
template <typename FPTYPE>
struct set_box_op {
    void operator() (
        const psi::DEVICE_GPU* dev,
        const int npwk,
        const FPTYPE* in,
        FPTYPE* out,
        const int* box_index);
};
#endif // __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
} // namespace hamilt
#endif //MODULE_HAMILT_EKINETIC_H