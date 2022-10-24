#include "module_pw/include/fft_multi_device.h"

using ModulePW::fft::fft_plan;
using ModulePW::fft::fft_destroy_plan_op;

template<>
struct __attribute__((unused)) fft_plan<float, psi::DEVICE_CPU> {
  __attribute__((unused)) typedef fftwf_plan plan;
};

template<>
struct __attribute__((unused)) fft_plan<double, psi::DEVICE_CPU> {
  __attribute__((unused)) typedef fftw_plan plan;
};

template <>
struct __attribute__((unused)) fft_destroy_plan_op<fftw_plan> {
  void operator() (fftw_plan& plan) {
    fftw_destroy_plan(plan);
  }
};

template <>
struct __attribute__((unused)) fft_destroy_plan_op<fftwf_plan> {
  void operator() (fftwf_plan& plan) {
    fftwf_destroy_plan(plan);
  }
};

namespace ModulePW{
}