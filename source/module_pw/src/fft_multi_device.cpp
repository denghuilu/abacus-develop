#include "module_pw/include/fft_multi_device.h"

using ModulePW::set_box_op;

template <typename FPTYPE>
struct set_box_op<FPTYPE, psi::DEVICE_CPU> {
    void operator() (
        const psi::DEVICE_CPU* dev,
        const int npwk,
        const FPTYPE* in,
        FPTYPE* out,
        const int* box_index)
    {

    }
};

namespace ModulePW{
template struct set_box_op<std::complex<double>, psi::DEVICE_CPU>;
}

