#include "module_hsolver/include/math_kernel.h"

#include "module_base/constants.h"
#include "module_hsolver/include/cuda/math_kernel_cuda.cuh"

#include <iostream>

template <typename FPTYPE>
FPTYPE hsolver::zdot_real(const int &dim,
                          const std::complex<FPTYPE> *psi_L,
                          const std::complex<FPTYPE> *psi_R,
                          const psi::AbacusDevice_t device,
                          const bool reduce)
{
    if (device == psi::CpuDevice)
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // qianrui modify 2021-3-14
        // Note that  ddot_(2*dim,a,1,b,1) = REAL( zdotc_(dim,a,1,b,1) )
        const FPTYPE *pL = reinterpret_cast<const FPTYPE *>(psi_L);
        const FPTYPE *pR = reinterpret_cast<const FPTYPE *>(psi_R);
        FPTYPE result = BlasConnector::dot(2 * dim, pL, 1, pR, 1);
        if (reduce)
            Parallel_Reduce::reduce_double_pool(result);
        return result;
    }
    else if (device == psi::GpuDevice)
    {
#if __CUDA
        zdot_real_gpu_cuda(dim, psi_L, psi_R, device, reduce);
#elif __ROCM
        zdot_real_gpu_rocm(dim, psi_L, psi_R, device, reduce);
#endif
    }
}

template <typename FPTYPE>
void hsolver::gemv(const char *const trans,
                   const int *const m,
                   const int *const n,
                   const FPTYPE *const alpha,
                   const FPTYPE *const A, // 矩阵
                   const int *const lda,
                   const FPTYPE *const X, // 向量
                   const int *const incx,
                   const FPTYPE *const beta,
                   FPTYPE *const Y, // result 向量
                   const int *const incy,
                   const psi::AbacusDevice_t device)
{
    if (device == psi::CpuDevice)
    {
        if (std::is_same<FPTYPE, std::complex<double>>())
        {
            zgemv_(trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
        }
        else if (std::is_same<FPTYPE, std::complex<float>>())
        {
            // "module_base/constants.h" 中并没有实现 cgemv()
            // cgemv_(trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
        }
        else if (std::is_same<FPTYPE, double>())
        {
            // dgemv_(trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
        }
        else if (std::is_same<FPTYPE, float>())
        {
            // "module_base/constants.h" 中并没有实现 cgemv()
            // sgemv_(trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
        }

        // if (reduce)
        // {
        //     Parallel_Reduce::reduce_complex_double_pool(Y, n);
        // }
    }
    else if (device == psi::GpuDevice)
    {
#if __CUDA
        gemv_gpu_cuda(trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
#elif __ROCM
#endif
    }
}

template <typename FPTYPE>
void hsolver::vector_div_constant(const int &dim,
                                  std::complex<FPTYPE> *result,
                                  const std::complex<FPTYPE> *input,
                                  const FPTYPE constant,
                                  const psi::AbacusDevice_t device)
{
    if (device == psi::CpuDevice)
    {
        for (int i = 0; i < dim; i++)
        {
            result[i] = input[i] / constant;
        }
    }
    else if (device == psi::GpuDevice)
    {
    }
}

template <typename FPTYPE>
void hsolver::vector_div_vector(const int &dim,
                                std::complex<FPTYPE> *result,
                                const std::complex<FPTYPE> *input1,
                                const FPTYPE *input2,
                                const psi::AbacusDevice_t device)
{
    if (device == psi::CpuDevice)
    {
        for (int i = 0; i < dim; i++)
        {
            result[i] = input1[i] / input2[i];
        }
    }
    else if (device == psi::GpuDevice)
    {
    }
}

// 向量操作：一个向量  🟰  其本身  ➖  一个常数  ✖️ 另一个向量
template <typename FPTYPE>
void hsolver::vector_sub_constantVector(const int &dim,
                                        std::complex<FPTYPE> *result,
                                        const std::complex<FPTYPE> *vector1,
                                        const std::complex<FPTYPE> *vector2,
                                        const FPTYPE constant,
                                        const psi::AbacusDevice_t device)
{
    if (device == psi::CpuDevice)
    {
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] - constant * vector2[i];
        }
    }
    else if (device == psi::GpuDevice)
    {
    }
}

template <typename FPTYPE>
void hsolver::vector_add_constantVector(const int &dim,
                                        std::complex<FPTYPE> *result,
                                        const std::complex<FPTYPE> *vector1,
                                        const std::complex<FPTYPE> *vector2,
                                        const FPTYPE constant,
                                        const psi::AbacusDevice_t device)
{
    if (device == psi::CpuDevice)
    {
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] + constant * vector2[i];
        }
    }
    else if (device == psi::GpuDevice)
    {
    }
}

template <typename FPTYPE>
void hsolver::vector_mul_vector(const int &dim,
                                std::complex<FPTYPE> *result,
                                const std::complex<FPTYPE> *vector1,
                                const FPTYPE *vector2,
                                const psi::AbacusDevice_t device)
{
    if (device == psi::CpuDevice)
    {
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] * vector2[i];
        }
    }
    else if (device == psi::GpuDevice)
    {
    }
}

template <typename FPTYPE>
void hsolver::constantvector_add_constantVector(const int &dim,
                                                std::complex<FPTYPE> *result,
                                                const std::complex<FPTYPE> *vector1,
                                                const FPTYPE constant1,
                                                const std::complex<FPTYPE> *vector2,
                                                const FPTYPE constant2,
                                                const psi::AbacusDevice_t device)
{
    if (device == psi::CpuDevice)
    {
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] * constant1 + vector2[i] * constant2;
        }
    }
    else if (device == psi::GpuDevice)
    {
    }
}

namespace hsolver
{
template double zdot_real<double>(const int &dim,
                                  const std::complex<double> *psi_L,
                                  const std::complex<double> *psi_R,
                                  const psi::AbacusDevice_t device,
                                  const bool reduce);

template void gemv<std::complex<double>>(const char *const trans,
                                         const int *const m,
                                         const int *const n,
                                         const std::complex<double> *const alpha,
                                         const std::complex<double> *const A, // 矩阵
                                         const int *const lda,
                                         const std::complex<double> *const X, // 向量
                                         const int *const incx,
                                         const std::complex<double> *const beta,
                                         std::complex<double> *const Y, // result 向量
                                         const int *const incy,
                                         const psi::AbacusDevice_t device);

template void vector_div_constant<double>(const int &dim,
                                          std::complex<double> *result,
                                          const std::complex<double> *input,
                                          const double constant,
                                          const psi::AbacusDevice_t device);

template void vector_div_vector<double>(const int &dim,
                                        std::complex<double> *result,
                                        const std::complex<double> *input1,
                                        const double *input2,
                                        const psi::AbacusDevice_t device);

template void vector_sub_constantVector<double>(const int &dim,
                                                std::complex<double> *result,
                                                const std::complex<double> *vector1,
                                                const std::complex<double> *vector2,
                                                const double constant,
                                                const psi::AbacusDevice_t device);

template void vector_add_constantVector<double>(const int &dim,
                                                std::complex<double> *result,
                                                const std::complex<double> *vector1,
                                                const std::complex<double> *vector2,
                                                const double constant,
                                                const psi::AbacusDevice_t device);

template void vector_mul_vector<double>(const int &dim,
                                        std::complex<double> *result,
                                        const std::complex<double> *vector1,
                                        const double *vector2,
                                        const psi::AbacusDevice_t device);

template void constantvector_add_constantVector<double>(const int &dim,
                                                        std::complex<double> *result,
                                                        const std::complex<double> *vector1,
                                                        const double constant1,
                                                        const std::complex<double> *vector2,
                                                        const double constant2,
                                                        const psi::AbacusDevice_t device);

} // namespace hsolver