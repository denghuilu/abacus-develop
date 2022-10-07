#ifndef HSOLVER_MATH_KERNEL_CUDA_H
#define HSOLVER_MATH_KERNEL_CUDA_H

#include "module_psi/psi.h"
#include "src_parallel/parallel_reduce.h"


namespace hsolver
{

void createCUBLAShandle();

void destroyCUBLAShandle();

// TODO
template <typename FPTYPE>
FPTYPE zdot_real_gpu_cuda(const int &dim,
                            const std::complex<FPTYPE> *psi_L,
                            const std::complex<FPTYPE> *psi_R,
                            const psi::AbacusDevice_t device = psi::GpuDevice,
                            const bool reduce = true);

template <typename FPTYPE>
void gemv_gpu_cuda( const char *const trans,
                    const int *const m,
                    const int *const n,
                    const FPTYPE *const alpha,
                    const FPTYPE *const A, // 矩阵
                    const int *const lda,
                    const FPTYPE *const X, // 向量
                    const int *const incx,
                    const FPTYPE *const beta,
                    FPTYPE *const Y, // result 向量
                    const int *const incy);


}


#endif