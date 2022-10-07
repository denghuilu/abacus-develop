#include "module_hsolver/include/cuda/math_kernel_cuda.cuh"
#include "module_psi/psi.h"
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include "cublas_v2.h"

static cublasHandle_t diag_handle; // cublas handle

void hsolver::createCUBLAShandle(){
    cublasCreate(&diag_handle);
}
void hsolver::destroyCUBLAShandle(){
    cublasDestroy(diag_handle);
}


/*  for this implementation, please check https://thrust.github.io/doc/group__transformed__reductions_ga321192d85c5f510e52300ae762c7e995.html
    denghui modify 2022-10-03
    Note that ddot_(2*dim,a,1,b,1) = REAL( zdotc_(dim,a,1,b,1) )
*/
template<typename FPTYPE>
FPTYPE hsolver::zdot_real_gpu_cuda(
    const int &dim, 
    const std::complex<FPTYPE>* psi_L, 
    const std::complex<FPTYPE>* psi_R, 
    const psi::AbacusDevice_t device,
    const bool reduce)
{
    const FPTYPE* pL = reinterpret_cast<const FPTYPE*>(psi_L);
    const FPTYPE* pR = reinterpret_cast<const FPTYPE*>(psi_R);
    FPTYPE result = thrust::inner_product(thrust::device, pL, pL + dim * 2, pR, FPTYPE(0.0));
    if (reduce) {
        Parallel_Reduce::reduce_double_pool(result);
    }
    return result;
}



template <typename FPTYPE>
void hsolver::gemv_gpu_cuda( const char *const trans,
                            const int *const m,
                            const int *const n,
                            const FPTYPE *const alpha,
                            const FPTYPE *const A, // 矩阵
                            const int *const lda,
                            const FPTYPE *const X, // 向量
                            const int *const incx,
                            const FPTYPE *const beta,
                            FPTYPE *const Y, // result 向量
                            const int *const incy)
{
    if (std::is_same<FPTYPE, std::complex<double>>())
    {
        // cublasZgemv(diag_handle, trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
    }
    else if (std::is_same<FPTYPE, std::complex<float>>())
    {
        // cublasCgemv(diag_handle, trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
    }
    else if (std::is_same<FPTYPE, double>())
    {
        // cublasDgemv(diag_handle, trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
    }
    else if (std::is_same<FPTYPE, float>())
    {
        // cublasSgemv(diag_handle, trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);
    }
}






namespace hsolver {
    template double zdot_real_gpu_cuda<double>(const int &dim, const std::complex<double>* psi_L, const std::complex<double>* psi_R, const psi::AbacusDevice_t device, const bool reduce);


    template void gemv_gpu_cuda<std::complex<double>>( const char *const trans,
                                                const int *const m,
                                                const int *const n,
                                                const std::complex<double> *const alpha,
                                                const std::complex<double> *const A, // 矩阵
                                                const int *const lda,
                                                const std::complex<double> *const X, // 向量
                                                const int *const incx,
                                                const std::complex<double> *const beta,
                                                std::complex<double> *const Y, // result 向量
                                                const int *const incy
                                                );
}

