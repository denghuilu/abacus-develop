#include <complex>
#include <iostream>
#include <gtest/gtest.h>
#include "module_psi/include/memory.h"
#include "module_hsolver/include/math_kernel.h"


class TestMathKernel : public ::testing::Test
{
  protected:
    // xx = tf.random.uniform([100], minval=-4, maxval=4, dtype = tf.float64)
    std::vector<complex<double> > psi_L = {
        (-0.65412617, -0.74208893), (-2.21731157,  0.42540039), 
        (3.36373004,  -2.51647562), (-2.985111  , -0.53251562), 
        (0.37908265,   0.81605825), ( 1.66281318,  2.71761869), 
        (2.2010268 ,   0.65498149), ( 1.51153638,  0.71501482), 
        (0.53546578,   1.4564317 ), (-2.36701143,  1.23009056), 
        (3.41302551,  -2.3175205 ), (-0.27628221, -1.35701656)
    };

    std::vector<complex<double> > psi_R = {
        (-1.67837557e-01, -1.70017454e-01), (-2.92128115e-02,  2.82765887e-01),
        (-8.71641062e-02, -1.15934278e-01), ( 3.36269232e+00, -1.44692661e-02),
        (-3.81342874e-03, -1.58276988e-01), ( 2.33504238e-01, -1.93195840e-03),
        ( 2.45520665e-01,  6.46854620e-01), ( 1.58255340e+00,  2.70915699e+00),
        (-1.66142311e-01,  6.27839507e-02), ( 2.17077193e+00,  4.87104731e-01),
        ( 1.41257916e+00,  5.45282609e-01), (-1.29333636e-01, -5.04228492e-03)
    };

    const int dim = psi_L.size();

    const double expected_result = 2.206021622595902;

    void SetUp() override {
    }
    void TearDown() override {
    }
};

// template<typename FPTYPE>
// FPTYPE zdot_real(const int &dim, const std::complex<FPTYPE>* psi_L, const std::complex<FPTYPE>* psi_R, const psi::AbacusDevice_t device = psi::CpuDevice, const bool reduce = true);
TEST_F(TestMathKernel, zdot_real)
{
    double result = hsolver::zdot_real(dim, psi_L.data(), psi_R.data(), psi::CpuDevice, false);
    EXPECT_LT(fabs(result - expected_result), 1e-12);
}

#if __UT_USE_CUDA
TEST_F(TestMathKernel, zdot_real_gpu_cuda)
{
  std::complex<double> * psi_L_dev = NULL, * psi_R_dev = NULL;
  psi::memory::abacus_malloc_device_memory_sync_gpu_cuda(psi_L_dev, psi_L);
  psi::memory::abacus_malloc_device_memory_sync_gpu_cuda(psi_R_dev, psi_R);
  double result = hsolver::zdot_real_gpu_cuda(dim, psi_L_dev, psi_R_dev, psi::GpuDevice, false);
  EXPECT_LT(fabs(result - expected_result), 1e-12);
  psi::memory::abacus_delete_memory_gpu_cuda(psi_L_dev);
  psi::memory::abacus_delete_memory_gpu_cuda(psi_R_dev);
}
#endif

/*
#if __UT_USE_ROCM
TEST_F(TestGelu, gelu_gpu_rocm)
{
  std::vector<double> gelu(nloc, 0.0);
  
  double * gelu_dev = NULL, * xx_dev = NULL;
  deepmd::malloc_device_memory_sync(gelu_dev, gelu);
  deepmd::malloc_device_memory_sync(xx_dev, xx);
  deepmd::gelu_gpu_rocm<double> (gelu_dev, xx_dev, nloc);
  deepmd::memcpy_device_to_host(gelu_dev, gelu);
  deepmd::delete_device_memory(gelu_dev);
  deepmd::delete_device_memory(xx_dev);

  EXPECT_EQ(gelu.size(), nloc);
  EXPECT_EQ(gelu.size(), expected_gelu.size());
  for (int jj = 0; jj < gelu.size(); ++jj){
    EXPECT_LT(fabs(gelu[jj] - expected_gelu[jj]) , 1e-5);
  }  
}
#endif // TENSORFLOW_USE_ROCM
*/