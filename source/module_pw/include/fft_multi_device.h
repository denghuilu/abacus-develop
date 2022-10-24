#ifndef MODULE_PW_FFT_MULTI_DEVICEH
#define MODULE_PW_FFT_MULTI_DEVICEH

#include <complex>
#include <string>
#include "module_psi/include/types.h"
#include "module_psi/include/memory.h"

//In this regard, use the device FFT library(cuFFT as well as rocFFT) with the fftw interface.
#if defined(__CUDA) || defined(__ROCM)
#include "cufft.h"
#endif // defined(__CUDA) || defined(__ROCM)

#include "fftw3.h"
#if defined(__FFTW3_MPI) && defined(__MPI)
#include <fftw3-mpi.h>
//#include "fftw3-mpi_mkl.h"
#endif // defined(__FFTW3_MPI) && defined(__MPI)

namespace ModulePW {
namespace fft{

template <typename FPTYPE, typename Device>
struct fft_plan;

template <typename FFT_PLAN>
struct fft_destroy_plan_op {
  void operator()(FFT_PLAN &plan);
};

//template <typename FFT_PLAN>
//struct fft_plan_many_op {
//  FFT_PLAN operator() (
//    FFT_PLAN& plan,
//    const int& rank);
//};


} // namespace fft
} // namespace ModulePW

#endif // MODULE_PW_FFT_MULTI_DEVICEH