// TODO: This is a temperary location for these functions.
// And will be moved to a global module(module base) later.
#ifndef MODULE_HSOLVER_MATH_KERNEL_H
#define MODULE_HSOLVER_MATH_KERNEL_H

#include "module_base/blas_connector.h"
#include "module_psi/psi.h"
#include "src_parallel/parallel_reduce.h"

namespace hsolver
{

template <typename FPTYPE, typename Device> 
struct zdot_real_op {
  FPTYPE operator() (
      const Device* d,
      const int& dim,
      const std::complex<FPTYPE>* psi_L,
      const std::complex<FPTYPE>* psi_R,
      const bool reduce = true);
};

// vector operator: result[i] = vector[i] / constant
template <typename FPTYPE, typename Device> struct vector_div_constant_op
{
    void operator()(const Device* d,
                    const int dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector,
                    const FPTYPE constant);
};

// replace vector_div_constant_op : x = alpha * x
template <typename FPTYPE, typename Device> struct scal_op
{
    void operator()(const Device* d,
                    const int& N,
                    const std::complex<FPTYPE>* alpha,
                    std::complex<FPTYPE>* X,
                    const int& incx);
};

// vector operator: result[i] = vector1[i](complex) * vector2[i](not complex)
template <typename FPTYPE, typename Device> struct vector_mul_vector_op
{
    void operator()(const Device* d,
                    const int& dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector1,
                    const FPTYPE* vector2);
};

// vector operator: result[i] = vector1[i](complex) / vector2[i](not complex)
template <typename FPTYPE, typename Device> struct vector_div_vector_op
{
    void operator()(const Device* d,
                    const int& dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector1,
                    const FPTYPE* vector2);
};

// vector operator: result[i] = vector1[i] * constant1 + vector2[i] * constant2
template <typename FPTYPE, typename Device> struct constantvector_addORsub_constantVector_op
{
    void operator()(const Device* d,
                    const int& dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector1,
                    const FPTYPE constant1,
                    const std::complex<FPTYPE>* vector2,
                    const FPTYPE constant2);
};

//  compute Y = alpha * X + Y
template <typename FPTYPE, typename Device> struct axpy_op
{
    void operator()(const Device* d,
                    const int& N,
                    const std::complex<FPTYPE>* alpha,
                    const std::complex<FPTYPE>* X,
                    const int& incX,
                    std::complex<FPTYPE>* Y,
                    const int& incY);
};

// compute y = alpha * op(A) * x + beta * y
template <typename FPTYPE, typename Device> struct gemv_op
{
    void operator()(const Device* d,
                    const char& trans,
                    const int& m,
                    const int& n,
                    const std::complex<FPTYPE>* alpha,
                    const std::complex<FPTYPE>* A,
                    const int& lda,
                    const std::complex<FPTYPE>* X,
                    const int& incx,
                    const std::complex<FPTYPE>* beta,
                    std::complex<FPTYPE>* Y,
                    const int& incy);
};

// ==================================================================
//     full specialization of template classes
// ==================================================================
// template <> struct axpy_op<double, psi::DEVICE_CPU>
// {
//     void operator()(const psi::DEVICE_CPU* d,
//                     const int& dim,
//                     const std::complex<double>* alpha,
//                     const std::complex<double>* X,
//                     const int& incX,
//                     std::complex<double>* Y,
//                     const int& incY);
//     // {
//     //     zaxpy_(&dim, alpha, X, &incX, Y, &incY);
//     // }
// };
// template <> struct axpy_op<float, psi::DEVICE_CPU>
// {
//     void operator()(const psi::DEVICE_CPU* d,
//                     const int& dim,
//                     const std::complex<float>* alpha,
//                     const std::complex<float>* X,
//                     const int& incX,
//                     std::complex<float>* Y,
//                     const int& incY);
//     // {
//     //     caxpy_(&dim, alpha, X, &incX, Y, &incY);
//     // }
// };
// template <> struct gemv_op<float, psi::DEVICE_CPU>
// {
//     void operator()(const psi::DEVICE_CPU* d,
//                     const char& trans,
//                     const int& m,
//                     const int& n,
//                     const std::complex<float>* alpha,
//                     const std::complex<float>* A,
//                     const int& lda,
//                     const std::complex<float>* X,
//                     const int& incx,
//                     const std::complex<float>* beta,
//                     std::complex<float>* Y,
//                     const int& incy)
//     {
//         // cgemv_(&trans, &m, &n, alpha, A, &lda, X, &incx, beta, Y, &incy);
//     }
// };
// template <> struct gemv_op<double, psi::DEVICE_CPU>
// {
//     void operator()(const psi::DEVICE_CPU* d,
//                     const char& trans,
//                     const int& m,
//                     const int& n,
//                     const std::complex<double>* alpha,
//                     const std::complex<double>* A,
//                     const int& lda,
//                     const std::complex<double>* X,
//                     const int& incx,
//                     const std::complex<double>* beta,
//                     std::complex<double>* Y,
//                     const int& incy)
//     {
//         zgemv_(&trans, &m, &n, alpha, A, &lda, X, &incx, beta, Y, &incy);
//     }
// };

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM

// Partially specialize functor for psi::GpuDevice.
template <typename FPTYPE> struct zdot_real_op<FPTYPE, psi::DEVICE_GPU>
{
    FPTYPE operator()(const psi::DEVICE_GPU* d,
                      const int& dim,
                      const std::complex<FPTYPE>* psi_L,
                      const std::complex<FPTYPE>* psi_R,
                      const bool reduce = true);
};

// vector operator: result[i] = vector[i] / constant
template <typename FPTYPE> struct vector_div_constant_op<FPTYPE, psi::DEVICE_GPU>
{
    void operator()(const psi::DEVICE_GPU* d,
                    const int dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector,
                    const FPTYPE constant);
};

// vector operator: result[i] = vector1[i](complex) * vector2[i](not complex)
template <typename FPTYPE> struct vector_mul_vector_op<FPTYPE, psi::DEVICE_GPU>
{
    void operator()(const psi::DEVICE_GPU* d,
                    const int& dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector1,
                    const FPTYPE* vector2);
};

// vector operator: result[i] = vector1[i](complex) / vector2[i](not complex)
template <typename FPTYPE> struct vector_div_vector_op<FPTYPE, psi::DEVICE_GPU>
{
    void operator()(const psi::DEVICE_GPU* d,
                    const int& dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector1,
                    const FPTYPE* vector2);
};

// vector operator: result[i] = vector1[i] * constant1 + vector2[i] * constant2
template <typename FPTYPE> struct constantvector_addORsub_constantVector_op<FPTYPE, psi::DEVICE_GPU>
{
    void operator()(const psi::DEVICE_GPU* d,
                    const int& dim,
                    std::complex<FPTYPE>* result,
                    const std::complex<FPTYPE>* vector1,
                    const FPTYPE constant1,
                    const std::complex<FPTYPE>* vector2,
                    const FPTYPE constant2);
};

// //  compute Y = alpha * X + Y
// template <typename FPTYPE> struct axpy_op<FPTYPE, psi::DEVICE_GPU>
// {
//     void operator()(const psi::DEVICE_GPU* d,
//                     const int& N,
//                     const std::complex<FPTYPE>* alpha,
//                     const std::complex<FPTYPE>* X,
//                     const int& incX,
//                     std::complex<FPTYPE>* Y,
//                     const int& incY);
// };

// // compute y = alpha * op(A) * x + beta * y
// template <typename FPTYPE> struct gemv_op<FPTYPE, psi::DEVICE_GPU>
// {
//     void operator()(const psi::DEVICE_GPU* d,
//                     const char& trans,
//                     const int& m,
//                     const int& n,
//                     const std::complex<FPTYPE>* alpha,
//                     const std::complex<FPTYPE>* A,
//                     const int& lda,
//                     const std::complex<FPTYPE>* X,
//                     const int& incx,
//                     const std::complex<FPTYPE>* beta,
//                     std::complex<FPTYPE>* Y,
//                     const int& incy);
// };

#endif // __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
} // namespace hsolver

#endif // MODULE_HSOLVER_MATH_KERNEL_H