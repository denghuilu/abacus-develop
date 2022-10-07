#ifndef HSOLVER_MATH_KERNEL_H
#define HSOLVER_MATH_KERNEL_H

#include "module_base/blas_connector.h"
#include "module_psi/psi.h"
#include "src_parallel/parallel_reduce.h"

#include <complex.h>

namespace hsolver
{

template <typename FPTYPE>
FPTYPE zdot_real(const int &dim,
                 const std::complex<FPTYPE> *psi_L,
                 const std::complex<FPTYPE> *psi_R,
                 const psi::AbacusDevice_t device = psi::CpuDevice,
                 const bool reduce = true);

// const int *const (const 既针对指针又针对内容，都为不可变)
template <typename FPTYPE>
void gemv(const char *const trans,
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
          const psi::AbacusDevice_t device = psi::CpuDevice);

// 向量操作：向量 🟰 一个向量 ➗ 一个常数
template <typename FPTYPE>
void vector_div_constant(const int &dim,
                         std::complex<FPTYPE> *result,
                         const std::complex<FPTYPE> *input,
                         const FPTYPE constant,
                         const psi::AbacusDevice_t device = psi::CpuDevice);

// 向量操作：向量 🟰 一个向量 ➗ 另一个向量（对应索引的元素）
// TODO 主要vector2的模版问题
template <typename FPTYPE>
void vector_div_vector(const int &dim,
                       std::complex<FPTYPE> *result,
                       const std::complex<FPTYPE> *vector1,
                       const FPTYPE *vector2,
                       const psi::AbacusDevice_t device = psi::CpuDevice);

// 向量操作：一个向量  🟰  其本身  ➖  一个常数  ✖️ 另一个向量
template <typename FPTYPE>
void vector_sub_constantVector(const int &dim,
                               std::complex<FPTYPE> *result,
                               const std::complex<FPTYPE> *vector1,
                               const std::complex<FPTYPE> *vector2,
                               const FPTYPE constant,
                               const psi::AbacusDevice_t device = psi::CpuDevice);

// 向量操作：一个向量 🟰 其本身 ✖️ 一个常数 ➕ 另一个向量
template <typename FPTYPE>
void vector_add_constantVector(const int &dim,
                               std::complex<FPTYPE> *result,
                               const std::complex<FPTYPE> *vector1,
                               const std::complex<FPTYPE> *vector2,
                               const FPTYPE constant,
                               const psi::AbacusDevice_t device = psi::CpuDevice);

// 向量操作：向量 🟰 一个向量 ✖️ 另一个向量（对应索引的元素）
// TODO 主要vector2的模版问题
template <typename FPTYPE>
void vector_mul_vector(const int &dim,
                       std::complex<FPTYPE> *result,
                       const std::complex<FPTYPE> *vector1,
                       const FPTYPE *vector2,
                       const psi::AbacusDevice_t device = psi::CpuDevice);

// 向量操作：一个向量 🟰 一个向量 ✖️ 一个常数 ➕ 另一个向量 ✖️ 另一个常数
template <typename FPTYPE>
void constantvector_add_constantVector(const int &dim,
                                       std::complex<FPTYPE> *result,
                                       const std::complex<FPTYPE> *vector1,
                                       const FPTYPE constant1,
                                       const std::complex<FPTYPE> *vector2,
                                       const FPTYPE constant2,
                                       const psi::AbacusDevice_t device = psi::CpuDevice);

// #if __CUDA
// void createCUBLAShandle();
// void destroyCUBLAShandle();
// // TODO
// template <typename FPTYPE>
// FPTYPE zdot_real_gpu_cuda(const int &dim,
//                           const std::complex<FPTYPE> *psi_L,
//                           const std::complex<FPTYPE> *psi_R,
//                           const psi::AbacusDevice_t device = psi::GpuDevice,
//                           const bool reduce = true);
// template <typename FPTYPE>
// void gemv_gpu_cuda( const char *const trans,
//                     const int *const m,
//                     const int *const n,
//                     const FPTYPE *const alpha,
//                     const FPTYPE *const A, // 矩阵
//                     const int *const lda,
//                     const FPTYPE *const X, // 向量
//                     const int *const incx,
//                     const FPTYPE *const beta,
//                     FPTYPE *const Y, // result 向量
//                     const int *const incy);
// #elif __ROCM
//     template <typename FPTYPE>
//     FPTYPE zdot_real_gpu_rocm(const int &dim,
//                               const std::complex<FPTYPE> *psi_L,
//                               const std::complex<FPTYPE> *psi_R,
//                               const psi::AbacusDevice_t device = psi::GpuDevice,
//                               const bool reduce = true);
// #endif //

} // namespace hsolver

#endif