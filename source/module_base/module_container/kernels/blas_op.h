#ifndef CONTAINER_KERNELS_BLAS_OP_H_
#define CONTAINER_KERNELS_BLAS_OP_H_

#include "../tensor.h"
#include "../tensor_types.h"
#include "third_party/blas_connector.h"
#include "third_party/lapack_connector.h"

#if defined(__CUDA) || defined(__UT_USE_CUDA)
#include <cuda_runtime.h>
#include "cublas_v2.h"
static const char * cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "Unknown";
    }
}
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"cuBLAS Assert: %s %s %d\n", cublasGetErrorEnum(code), file, line);
        if (abort) exit(code);
    }
}

#define cublasErrcheck(res) { cublasAssert((res), __FILE__, __LINE__); }

#define cudaErrcheck(res) {                                             \
    if (res != cudaSuccess) {                                           \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}
#endif //__CUDA || __UT_USE_CUDA

namespace container {
namespace op {

/**
 * @brief A struct representing a scaling operation on a vector.
 *
 * This struct defines the operation of scaling a vector with a scalar.
 * The scaling operation modifies the input vector by multiplying each element
 * of the vector by the given scalar alpha.
 *
 * @tparam T The type of the elements in the vector (e.g., float, double, etc.).
 * @tparam DEVICE The device where the vector resides (e.g., CPU, GPU, etc.).
 */
template <typename T, typename DEVICE>
struct scal_op {
    /**
     * @brief Perform the scaling operation on the input vector.
     *
     * This function applies the scaling operation to the input vector X.
     * It multiplies each element of the vector by the given scalar alpha.
     *
     * @param N The number of elements in the vector.
     * @param alpha A pointer to the scalar used for scaling.
     * @param X A pointer to the input vector to be scaled.
     * @param incx The increment between consecutive elements of the vector X.
     *
     * @note The length of the vector X should be at least (1 + (N - 1) * abs(incx)).
     */
    void operator()(
        const int64_t& N,
        const T* alpha,
        T* X,
        const int64_t& incx);
};


//  compute Y = alpha * X + Y
template <typename T, typename DEVICE>
struct axpy_op {
    /// @brief Y = alpha * X + Y
    ///
    /// Input Parameters
    /// \param N : array size
    /// \param alpha : input constant alpha
    /// \param X : input array X
    /// \param incX : computing strip of X
    /// \param Y : computing strip of Y
    /// \param incY : computing strip of Y
    ///
    /// Output Parameters
    /// \param Y : output array Y
    void operator()(
            const int& N,
            const std::complex<T>* alpha,
            const std::complex<T>* X,
            const int& incX,
            std::complex<T>* Y,
            const int& incY);
};

// compute y = alpha * op(A) * x + beta * y
template <typename T, typename DEVICE>
struct gemv_op {
    /// @brief y = alpha * op(A) * x + beta * y
    ///
    /// Input Parameters
    /// \param trans : whether to transpose A
    /// \param m : first dimension of matrix
    /// \param n : second dimension of matrix
    /// \param alpha : input constant alpha
    /// \param A : input matrix A
    /// \param lda : leading dimention of A
    /// \param X : input array X
    /// \param incx : computing strip of X
    /// \param beta : input constant beta
    /// \param Y : input array Y
    /// \param incy : computing strip of Y
    ///
    /// Output Parameters
    /// \param Y : output array Y
    void operator()(
            const char &trans,
            const int &m,
            const int &n,
            const std::complex<T> *alpha,
            const std::complex<T> *A,
            const int &lda,
            const std::complex<T> *X,
            const int &incx,
            const std::complex<T> *beta,
            std::complex<T> *Y,
            const int &incy);
};

// compute C = alpha * op(A) * op(B) + beta * C
template <typename T, typename DEVICE>
struct gemm_op {
    /// @brief C = alpha * op(A) * op(B) + beta * C
    ///
    /// Input Parameters
    /// \param transa : whether to transpose matrix A
    /// \param transb : whether to transpose matrix B
    /// \param m : first dimension of matrix mulplication
    /// \param n : second dimension of matrix mulplication
    /// \param k : third dimension of matrix mulplication
    /// \param alpha : input constant alpha
    /// \param a : input matrix A
    /// \param lda : leading dimention of A
    /// \param b : input matrix B
    /// \param ldb : leading dimention of A
    /// \param beta : input constant beta
    /// \param c : input matrix C
    /// \param ldc : leading dimention of C
    ///
    /// Output Parameters
    /// \param c : output matrix C
    void operator()(
            const char& transa,
            const char& transb,
            const int& m,
            const int& n,
            const int& k,
            const std::complex<T> *alpha,
            const std::complex<T> *a,
            const int& lda,
            const std::complex<T> *b,
            const int& ldb,
            const std::complex<T> *beta,
            std::complex<T> *c,
            const int& ldc);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM

void createBlasHandle();
void destoryBlasHandle();

#endif // __CUDA || __UT_USE_CUDA
} // namespace op
} // namespace container

#endif // CONTAINER_KERNELS_BLAS_OP_H_