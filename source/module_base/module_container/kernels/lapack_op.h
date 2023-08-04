#ifndef CONTAINER_KERNELS_LAPACK_OP_H_
#define CONTAINER_KERNELS_LAPACK_OP_H_

#include "../tensor.h"
#include "../tensor_types.h"
#include "third_party/lapack_connector.h"

#if defined(__CUDA) || defined(__UT_USE_CUDA)
#include "gpu_utils.h"
#endif

namespace container {
namespace op {

/**
 * @brief A struct representing the DNGVD operation for computing eigenvalues and eigenvectors of a complex generalized Hermitian-definite eigenproblem.
 *
 * The DNGVD operation computes all the eigenvalues and eigenvectors of a complex generalized Hermitian-definite eigenproblem.
 * If eigenvectors are desired, it uses a divide and conquer algorithm. The operation supports both CPU and CUDA versions.
 *
 * API Documentation:
 * 1. LAPACK zhegvd: https://netlib.org/lapack/explore-html/df/d9a/group__complex16_h_eeigen_ga74fdf9b5a16c90d8b7a589dec5ca058a.html
 * 2. cuSOLVER cusolverDnZhegvd: https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
 *
 * @tparam T The type of the elements in the matrices (e.g., float, double, etc.).
 * @tparam Device The device where the matrices reside (e.g., CPU, GPU, etc.).
 */
template <typename T, typename Device>
struct dngvd_op {
    /**
     * @brief Compute all the eigenvalues and eigenvectors of a complex generalized Hermitian-definite eigenproblem.
     *
     * This function computes all the eigenvalues and eigenvectors of a complex generalized Hermitian-definite eigenproblem.
     * The problem is defined as A * x = lambda * B * x, where A is the hermitian matrix, B is the overlap matrix, x is the eigenvector,
     * and lambda is the eigenvalue. The operation supports both CPU and CUDA versions.
     *
     * @param nstart The number of columns of the matrix.
     * @param ldh The leading dimension (stride) of the matrix A and B.
     * @param A A pointer to the hermitian matrix A (col major).
     * @param B A pointer to the overlap matrix B (col major).
     * @param W A pointer to store the calculated eigenvalues.
     * @param V A pointer to store the calculated eigenvectors (col major).
     *
     * @note The leading dimension ldh should be at least max(1, nstart) for row-major storage and column-major storage.
     * @note The length of the A and B arrays should be at least nstart * ldh.
     * @note The length of the W array should be at least nstart, and the length of the V array should be at least nstart * ldh.
     */
    void operator()(
        const int nstart,
        const int ldh,
        const std::complex<T>* A,
        const std::complex<T>* B,
        T* W,
        std::complex<T>* V);
};


/**
 * @brief dnevx_op computes the first m eigenvalues and their corresponding eigenvectors of
 * a complex generalized Hermitian-definite eigenproblem.
 *
 * In this op, the CPU version is implemented through the `evx` interface, and the CUDA version
 * is implemented through the `evd` interface and acquires the first m eigenpairs.
 * 
 * API doc:
 * 1. zheevx: https://netlib.org/lapack/explore-html/df/d9a/group__complex16_h_eeigen_gaabef68a9c7b10df7aef8f4fec89fddbe.html
 * 2. cusolverDnZheevd: https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-syevd
 *
 * @tparam T The data type of the matrix and vectors (e.g., float, double)
 * @tparam Device The device type (e.g., DEVICE_CPU, DEVICE_CUDA)
 */
template <typename T, typename Device>
struct dnevx_op {
    /**
     * @brief Computes the eigenvalues and eigenvectors of a complex generalized Hermitian-definite eigenproblem.
     *
     * @param nstart The number of cols of the matrix.
     * @param ldh The number of rows of the matrix.
     * @param A The hermitian matrix A in A x=lambda B x (row major).
     * @param m The number of eigenvalues and eigenvectors to compute.
     * @param W Array to store the calculated eigenvalues.
     * @param V Array to store the calculated eigenvectors (row major).
     */
    void operator()(
        const int nstart,
        const int ldh,
        const std::complex<T>* A,
        const int m,
        T* W,
        std::complex<T>* V);
};


#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
void createCusolverHandle();  // create cusolver handle
void destroyCusolverHandle(); // destroy cusolver handle
#endif

} // namespace container
} // namespace op

#endif // CONTAINER_KERNELS_LAPACK_OP_H_