#include <complex>
#include <iostream>
#include <gtest/gtest.h>
#include "module_psi/include/memory.h"
#include "module_hsolver/include/math_kernel.h"
#include "module_base/blas_connector.h"
#include "module_base/constants.h"


class TestModuleHsolverMathKernel : public ::testing::Test
{
  protected:
    // xx = tf.random.uniform([100], minval=-4, maxval=4, dtype = tf.float64)
    std::vector<complex<double> > psi_L = {
        std::complex<double>(-0.65412617, -0.74208893), std::complex<double>(-2.21731157,  0.42540039), 
        std::complex<double>(3.36373004,  -2.51647562), std::complex<double>(-2.985111  , -0.53251562), 
        std::complex<double>(0.37908265,   0.81605825), std::complex<double>( 1.66281318,  2.71761869), 
        std::complex<double>(2.2010268 ,   0.65498149), std::complex<double>( 1.51153638,  0.71501482), 
        std::complex<double>(0.53546578,   1.4564317 ), std::complex<double>(-2.36701143,  1.23009056), 
        std::complex<double>(3.41302551,  -2.3175205 ), std::complex<double>(-0.27628221, -1.35701656)
    };

    std::vector<complex<double> > psi_R = {
        std::complex<double>(-1.67837557e-01, -1.70017454e-01), std::complex<double>(-2.92128115e-02,  2.82765887e-01),
        std::complex<double>(-8.71641062e-02, -1.15934278e-01), std::complex<double>( 3.36269232e+00, -1.44692661e-02),
        std::complex<double>(-3.81342874e-03, -1.58276988e-01), std::complex<double>( 2.33504238e-01, -1.93195840e-03),
        std::complex<double>( 2.45520665e-01,  6.46854620e-01), std::complex<double>( 1.58255340e+00,  2.70915699e+00),
        std::complex<double>(-1.66142311e-01,  6.27839507e-02), std::complex<double>( 2.17077193e+00,  4.87104731e-01),
        std::complex<double>( 1.41257916e+00,  5.45282609e-01), std::complex<double>(-1.29333636e-01, -5.04228492e-03)
    };

    const int dim = psi_L.size();

    const double expected_result = 2.206021622595902;

    const psi::DEVICE_CPU * cpu_ctx = {};
    const psi::DEVICE_GPU * gpu_ctx = {};

    void SetUp() override {
    }
    void TearDown() override {
    }

    using zdot_real_cpu_op = hsolver::zdot_real_op<double, psi::DEVICE_CPU>;
    using zdot_real_gpu_op = hsolver::zdot_real_op<double, psi::DEVICE_GPU>;
    using resize_memory_op = psi::memory::resize_memory_op<std::complex<double>, psi::DEVICE_GPU>;
    using delete_memory_op = psi::memory::delete_memory_op<std::complex<double>, psi::DEVICE_GPU>;
    using synchronize_memory_op = psi::memory::synchronize_memory_op<std::complex<double>, psi::DEVICE_GPU, psi::DEVICE_CPU>;



    // haozhihan add
    // (1) for test vector_div_constant_op
    const std::vector<std::complex<double> > input = psi_L;
    const double constant = 5.5;
    const std::vector<std::complex<double> > output_vector_div_constant_op = {
      std::complex<double>(-0.11893203,-0.13492526), std::complex<double>(-0.40314756,0.07734553),
      std::complex<double>(0.61158728,-0.45754102), std::complex<double>(-0.54274745,-0.09682102),
      std::complex<double>(0.06892412,0.14837423), std::complex<double>(0.30232967,0.49411249),
      std::complex<double>(0.40018669,0.11908754), std::complex<double>(0.27482480,0.13000269),
      std::complex<double>(0.09735741,0.26480576), std::complex<double>(-0.43036571,0.22365283),
      std::complex<double>(0.62055009,-0.42136736), std::complex<double>(-0.05023313,-0.24673028)
    };

    // (2) for test vector_mul_vector_op & vector_div_vector_op
    const std::vector<double> input_double = {
        -0.65412617, -0.74208893, -2.21731157,  0.42540039, 
        3.36373004,  -2.51647562, -2.985111  , -0.53251562, 
        0.37908265,   0.81605825,  1.66281318,  2.71761869, 
    };
    const std::vector<std::complex<double> >output_vector_mul_vector_op = {
      std::complex<double>(0.42788105,0.48541979), std::complex<double>(1.64544237,-0.31568492),
      std::complex<double>(-7.45843754,5.57981051), std::complex<double>(-1.26986738,-0.22653235),
      std::complex<double>(1.27513170,2.74499965), std::complex<double>(-4.18442883,-6.83882118),
      std::complex<double>(-6.57030931,-1.95519245), std::complex<double>(-0.80491673,-0.38075656),
      std::complex<double>(0.20298579,0.55210799), std::complex<double>(-1.93161921,1.00382555),
      std::complex<double>(5.67522380,-3.85360363), std::complex<double>(-0.75082970,-3.68785357)
    };

    const std::vector<std::complex<double> >output_vector_div_vector_op = {
      std::complex<double>(1.00000000,1.13447369), std::complex<double>(2.98793242,-0.57324718),
      std::complex<double>(-1.51703084,1.13492197), std::complex<double>(-7.01717974,-1.25179862),
      std::complex<double>(0.11269711,0.24260516), std::complex<double>(-0.66077063,-1.07993047),
      std::complex<double>(-0.73733499,-0.21941613), std::complex<double>(-2.83848271,-1.34271145),
      std::complex<double>(1.41253043,3.84198987), std::complex<double>(-2.90054225,1.50735632),
      std::complex<double>(2.05256102,-1.39373474), std::complex<double>(-0.10166335,-0.49934031)
    };

    // (3) for test constantvector_addORsub_constantVector_op
    const double constant1 = 6.6;
    const double constant2 = 4.4;
    const std::vector<std::complex<double> > input1 = psi_L;
    const std::vector<std::complex<double> > input2 = psi_R;
    const std::vector<std::complex<double> >output_constantvector_addORsub_constantVector_op = {
      std::complex<double>(-5.05571797,-5.64586374), std::complex<double>(-14.76279273,4.05181248), 
      std::complex<double>(21.81709620,-17.11884992), std::complex<double>(-4.90588639,-3.57826786), 
      std::complex<double>(2.48516640,4.68956570), std::complex<double>(12.00198564,17.92778274), 
      std::complex<double>(15.60706781,7.16903816), std::complex<double>(16.93937507,16.63938857), 
      std::complex<double>(2.80304798,9.88869860), std::complex<double>(-6.07087895,10.26185851), 
      std::complex<double>(28.74131667,-12.89639182), std::complex<double>(-2.39253058,-8.97849535)
    };

    // (4) for test axpy_op (compute Y = alpha * X + Y ) 
    const std::complex<double> alpha_axpy{-1.5, -2.5};
    const std::vector<std::complex<double> > X_axpy = psi_L;
    std::vector<std::complex<double> > Y_axpy = psi_R;

    const std::vector<std::complex<double> >output_axpy_op = {
      std::complex<double>(-1.04187063,2.57843137), std::complex<double>(4.36025552,5.18794423), 
      std::complex<double>(-11.42394822,-4.75054595), std::complex<double>(6.50906977,8.24708166), 
      std::complex<double>(1.46770822,-2.33007099), std::complex<double>(4.53333119,-8.23539294), 
      std::complex<double>(-1.41856581,-5.83818462), std::complex<double>(1.10278588,-2.14220619), 
      std::complex<double>(2.67173827,-3.46052805), std::complex<double>(8.79651547,4.55949747), 
      std::complex<double>(-9.50076036,-4.51100042), std::complex<double>(-3.10745172,2.72118808) 
    };

    // (5) for test scal_op (x = alpha * x)
    const std::complex<double> alpha_scal{-1.5, -2.5};
    std::vector<std::complex<double> > X_scal = psi_L;

    const std::vector<std::complex<double> >output_scal_op = {
      std::complex<double>(-0.87403307,2.74844882), std::complex<double>(4.38946833,4.90517834), 
      std::complex<double>(-11.33678411,-4.63461167), std::complex<double>(3.14637745,8.26155093), 
      std::complex<double>(1.47152165,-2.17179400), std::complex<double>(4.29982696,-8.23346099), 
      std::complex<double>(-1.66408648,-6.48503924), std::complex<double>(-0.47976752,-4.85136318), 
      std::complex<double>(2.83788058,-3.52331200), std::complex<double>(6.62574354,4.07239273), 
      std::complex<double>(-10.91333952,-5.05628302), std::complex<double>(-2.97811808,2.72623036)
    };


    // (6) for test gemv_op ( y = alpha * op(A) * x + beta * y )
    const std::vector<std::complex<double> > A_gemv = { // 2 * 3 
      std::complex<double>(-0.87403307,2.74844882), std::complex<double>(4.38946833,4.90517834), 
      std::complex<double>(-11.33678411,-4.63461167), std::complex<double>(3.14637745,8.26155093), 
      std::complex<double>(1.47152165,-2.17179400), std::complex<double>(4.29982696,-8.23346099)
    };

    const std::vector<std::complex<double> > X_gemv = { // 2 * 1 
      std::complex<double>(-0.87403307,2.74844882), std::complex<double>(4.38946833,4.90517834)
    };

    std::vector<std::complex<double> > Y_gemv = { // 3 * 1
      std::complex<double>(1.47152165,-2.17179400), std::complex<double>(4.29982696,-8.23346099),
      std::complex<double>(3.14637745,8.26155093)
    };

    std::vector<std::complex<double> > Y_test_gemv = { // 3 * 1
      std::complex<double>(1.47152165,-2.17179400), std::complex<double>(4.29982696,-8.23346099),
      std::complex<double>(3.14637745,8.26155093)
    };

    // const std::vector<std::complex<double> >output_gemv_op = {

    // };



    // haozhihan add
    using vector_div_constant_op_cpu = hsolver::vector_div_constant_op<double, psi::DEVICE_CPU>;
    using vector_mul_vector_op_cpu = hsolver::vector_mul_vector_op<double, psi::DEVICE_CPU>;
    using vector_div_vector_op_cpu = hsolver::vector_div_vector_op<double, psi::DEVICE_CPU>;
    using constantvector_addORsub_constantVector_op_cpu = hsolver::constantvector_addORsub_constantVector_op<double, psi::DEVICE_CPU>;
    using axpy_op_cpu = hsolver::axpy_op<double, psi::DEVICE_CPU>;
    using scal_op_cpu = hsolver::scal_op<double, psi::DEVICE_CPU>;
    using gemv_op_cpu = hsolver::gemv_op<double, psi::DEVICE_CPU>;



};

// template<typename FPTYPE>
// FPTYPE zdot_real(const int &dim, const std::complex<FPTYPE>* psi_L, const std::complex<FPTYPE>* psi_R, const psi::AbacusDevice_t device = psi::CpuDevice, const bool reduce = true);
TEST_F(TestModuleHsolverMathKernel, zdot_real_op_cpu)
{
  double result = zdot_real_cpu_op()(cpu_ctx, dim, psi_L.data(), psi_R.data(), false);
  EXPECT_LT(fabs(result - expected_result), 1e-12);
}

TEST_F(TestModuleHsolverMathKernel, vector_div_constant_op_cpu)
{
  std::vector<std::complex<double> > output(input.size());
  vector_div_constant_op_cpu()(cpu_ctx, dim, output.data(), input.data(), constant);
  for (int i = 0; i < input.size(); i++)
  {
    EXPECT_LT(fabs(output[i].imag() - output_vector_div_constant_op[i].imag()), 1e-12);
    EXPECT_LT(fabs(output[i].real() - output_vector_div_constant_op[i].real()), 1e-12);
  }
}

// TEST_F(TestModuleHsolverMathKernel, vector_mul_vector_op_cpu)
// {
//   std::vector<std::complex<double> > output(input.size());
//   vector_mul_vector_op_cpu()(cpu_ctx, dim, output.data(), input.data(), input_double.data());
//   for (int i = 0; i < input.size(); i++)
//   {
//     EXPECT_LT(fabs(output[i].imag() - output_vector_mul_vector_op[i].imag()), 1e-12);
//     EXPECT_LT(fabs(output[i].real() - output_vector_mul_vector_op[i].real()), 1e-12);
//   }
// }

// TEST_F(TestModuleHsolverMathKernel, vector_div_vector_op_cpu)
// {
//   std::vector<std::complex<double> > output(input.size());
//   vector_div_vector_op_cpu()(cpu_ctx, dim, output.data(), input.data(), input_double.data());
//   for (int i = 0; i < input.size(); i++)
//   {
//     EXPECT_LT(fabs(output[i].imag() - output_vector_div_vector_op[i].imag()), 1e-12);
//     EXPECT_LT(fabs(output[i].real() - output_vector_div_vector_op[i].real()), 1e-12);
//   }
// }

// TEST_F(TestModuleHsolverMathKernel, constantvector_addORsub_constantVector_op_cpu)
// {
//   std::vector<std::complex<double> > output(input.size());
//   constantvector_addORsub_constantVector_op_cpu()(cpu_ctx, dim, output.data(), input1.data(), constant1, input2.data(), constant2);
//   for (int i = 0; i < input.size(); i++)
//   {
//     EXPECT_LT(fabs(output[i].imag() - output_constantvector_addORsub_constantVector_op[i].imag()), 1e-12);
//     EXPECT_LT(fabs(output[i].real() - output_constantvector_addORsub_constantVector_op[i].real()), 1e-12);
//   }
// }

// TEST_F(TestModuleHsolverMathKernel, axpy_op_cpu)
// {
//   axpy_op_cpu()(cpu_ctx, dim, &alpha_axpy, X_axpy.data(), 1, Y_axpy.data(), 1);
//   for (int i = 0; i < input.size(); i++)
//   {
//     EXPECT_LT(fabs(Y_axpy[i].imag() - output_axpy_op[i].imag()), 1e-12);
//     EXPECT_LT(fabs(Y_axpy[i].real() - output_axpy_op[i].real()), 1e-12);
//   }
// }

// TEST_F(TestModuleHsolverMathKernel, scal_op_cpu)
// {
//   scal_op_cpu()(cpu_ctx, dim, &alpha_scal, X_scal.data(), 1);
//   for (int i = 0; i < input.size(); i++)
//   {
//     EXPECT_LT(fabs(X_scal[i].imag() - output_scal_op[i].imag()), 1e-12);
//     EXPECT_LT(fabs(X_scal[i].real() - output_scal_op[i].real()), 1e-12);
//   }
// }

// TEST_F(TestModuleHsolverMathKernel, gemv_op_cpu)
// {
//   gemv_op_cpu()(cpu_ctx, 'C', 2,  3, &ModuleBase::ONE, A_gemv.data(), 2, X_gemv.data(), 1, &ModuleBase::ZERO, Y_gemv.data(), 1);
//   char trans = 'C';
//   int inc = 1;
//   int row = 2;
//   int col = 3;
//   zgemv_(&trans, &row,  &row, &ModuleBase::ONE, A_gemv.data(), &row, X_gemv.data(), &inc, &ModuleBase::ZERO, Y_test_gemv.data(), &inc);
//   for (int i = 0; i < Y_gemv.size(); i++)
//   {
//     EXPECT_LT(fabs(Y_gemv[i].imag() - Y_test_gemv[i].imag()), 1e-12);
//     EXPECT_LT(fabs(Y_gemv[i].real() - Y_test_gemv[i].real()), 1e-12);
//   }
// }


#if __UT_USE_CUDA || __UT_USE_ROCM
TEST_F(TestModuleHsolverMathKernel, zdot_real_op_gpu)
{
  std::complex<double>* psi_L_dev = NULL, * psi_R_dev = NULL;
  resize_memory_op()(gpu_ctx, psi_L_dev, psi_L.size());
  resize_memory_op()(gpu_ctx, psi_R_dev, psi_R.size());
  synchronize_memory_op()(gpu_ctx, cpu_ctx, psi_L_dev, psi_L.data(), psi_L.size());
  synchronize_memory_op()(gpu_ctx, cpu_ctx, psi_R_dev, psi_R.data(), psi_R.size());
  double result = zdot_real_gpu_op()(gpu_ctx, dim, psi_L_dev, psi_R_dev, false);
  EXPECT_LT(fabs(result - expected_result), 1e-12);
  delete_memory_op()(gpu_ctx, psi_L_dev);
  delete_memory_op()(gpu_ctx, psi_R_dev);
}
#endif // __UT_USE_CUDA || __UT_USE_ROCM

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