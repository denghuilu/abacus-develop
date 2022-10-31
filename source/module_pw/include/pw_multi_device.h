#ifndef MODULE_PW_MULTI_DEVICE_H
#define MODULE_PW_MULTI_DEVICE_H

#include "module_psi/psi.h"
#include <complex>

namespace ModulePW {
template <typename FPTYPE, typename Device>
struct set_3d_fft_box_op {
    void operator() (
        const Device* dev,
        const int npwk,
        const std::complex<FPTYPE>* in,
        std::complex<FPTYPE>* out,
        const int* box_index);
};

template <typename FPTYPE, typename Device>
struct set_recip_to_real_output_op {
    void operator() (
        const Device* dev,
        const int nrxx,
        const bool add,
        const FPTYPE factor,
        const std::complex<FPTYPE>* in,
        std::complex<FPTYPE>* out);
};

template <typename FPTYPE, typename Device>
struct set_real_to_recip_output_op {
    void operator() (
        const Device* dev,
        const int npw_k,
        const int nxyz,
        const bool add,
        const FPTYPE factor,
        const int* box_index,
        const std::complex<FPTYPE>* in,
        std::complex<FPTYPE>* out);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
// Partially specialize functor for psi::GpuDevice.
template <typename FPTYPE>
struct set_3d_fft_box_op<FPTYPE, psi::DEVICE_GPU> {
    void operator() (
        const psi::DEVICE_GPU* dev,
        const int npwk,
        const std::complex<FPTYPE>* in,
        std::complex<FPTYPE>* out,
        const int* box_index);
};

template <typename FPTYPE>
struct set_recip_to_real_output_op<FPTYPE, psi::DEVICE_GPU> {
    void operator() (
        const psi::DEVICE_GPU* dev,
        const int nrxx,
        const bool add,
        const FPTYPE factor,
        const std::complex<FPTYPE>* in,
        std::complex<FPTYPE>* out);
};

template <typename FPTYPE>
struct set_real_to_recip_output_op<FPTYPE, psi::DEVICE_GPU> {
    void operator() (
        const psi::DEVICE_GPU* dev,
        const int npw_k,
        const int nxyz,
        const bool add,
        const FPTYPE factor,
        const int* box_index,
        const std::complex<FPTYPE>* in,
        std::complex<FPTYPE>* out);
};

#endif // __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
} // namespace ModulePW
#endif //MODULE_PW_MULTI_DEVICE_H