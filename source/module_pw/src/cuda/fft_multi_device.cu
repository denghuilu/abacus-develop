#include "module_pw/include/fft_multi_device.h"

using ModulePW::fft::fft_plan;
using ModulePW::fft::fft_destroy_plan_op;

template<typename FPTYPE>
struct __attribute__((unused)) fft_plan<FPTYPE, psi::DEVICE_GPU> {
  __attribute__((unused)) typedef cufftHandle plan;
};

template <>
struct __attribute__((unused)) fft_destroy_plan_op<cufftHandle> {
  void operator() (cufftHandle& plan) {
    cufftDestroy(plan);
  }
};

namespace ModulePW {
namespace fft {
template struct fft_plan<float, psi::DEVICE_GPU>;
template struct fft_plan<double, psi::DEVICE_GPU>;


} // namespace fft
} // namespace ModulePW