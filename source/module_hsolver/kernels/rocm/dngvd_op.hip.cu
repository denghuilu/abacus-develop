#include "module_hsolver/kernels/dngvd_op.h"

#include <hip/hip_runtime.h>
#include <base/macros/macros.h>
#include <base/third_party/lapack.h>

namespace hsolver {

static hipsolverDnHandle_t hipsolver_handle = nullptr;

void createGpuSolverHandle()
{
    if (hipsolver_handle == nullptr)
    {
        hipsolverErrcheck(hipsolverDnCreate(&hipsolver_handle));
    }
}

void destroyGpuSolverHandle()
{
    if (hipsolver_handle != nullptr)
    {
        hipsolverErrcheck(hipsolverDnDestroy(hipsolver_handle));
        hipsolver_handle = nullptr;
    }
}

static inline
void xhegvd_wrapper(
    const hipsolverFillMode_t& uplo,
    const int& n,
    double* A, const int& lda,
    double* B, const int& ldb,
    double* W)
{
    // prepare some values for hipsolverDnZhegvd_bufferSize
    int* devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double* work = nullptr;
    hipErrcheck(hipMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    hipsolverErrcheck(hipsolverDnDsygvd_bufferSize(hipsolver_handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
        A, lda, B, ldb, W, &lwork));
    // allocate memery
    hipErrcheck(hipMalloc((void**)&work, sizeof(double) * lwork));

    // compute eigenvalues and eigenvectors.
    hipsolverErrcheck(hipsolverDnDsygvd(hipsolver_handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
        A, lda, B, ldb, W, work, lwork, devInfo));

    hipErrcheck(hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost));
    assert(0 == info_gpu);
    // free the buffer
    hipErrcheck(hipFree(work));
    hipErrcheck(hipFree(devInfo));
}

static inline
void xhegvd_wrapper (
        const hipsolverFillMode_t& uplo,
        const int& n,
        std::complex<float> * A, const int& lda,
        std::complex<float> * B, const int& ldb,
        float * W)
{
    // prepare some values for hipsolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    float2 * work = nullptr;
    hipErrcheck(hipMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    hipsolverErrcheck(hipsolverDnChegvd_bufferSize(hipsolver_handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const float2 *>(A), lda,
                                                 reinterpret_cast<const float2 *>(B), ldb, W, &lwork));
    // allocate memery
    hipErrcheck(hipMalloc((void**)&work, sizeof(float2) * lwork));

    // compute eigenvalues and eigenvectors.
    hipsolverErrcheck(hipsolverDnChegvd(hipsolver_handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<float2 *>(A), lda, reinterpret_cast<float2 *>(B), ldb, W, work, lwork, devInfo));

    hipErrcheck(hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost));
    assert(0 == info_gpu);
    // free the buffer
    hipErrcheck(hipFree(work));
    hipErrcheck(hipFree(devInfo));
}

static inline
void xhegvd_wrapper (
        const hipsolverFillMode_t& uplo,
        const int& n,
        std::complex<double> * A, const int& lda,
        std::complex<double> * B, const int& ldb,
        double * W)
{
    // prepare some values for hipsolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double2 * work = nullptr;
    hipErrcheck(hipMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    hipsolverErrcheck(hipsolverDnZhegvd_bufferSize(hipsolver_handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const double2 *>(A), lda,
                                                 reinterpret_cast<const double2 *>(B), ldb, W, &lwork));
    // allocate memery
    hipErrcheck(hipMalloc((void**)&work, sizeof(double2) * lwork));

    // compute eigenvalues and eigenvectors.
    hipsolverErrcheck(hipsolverDnZhegvd(hipsolver_handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<double2 *>(A), lda, reinterpret_cast<double2 *>(B), ldb, W, work, lwork, devInfo));

    hipErrcheck(hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost));
    assert(0 == info_gpu);
    // free the buffer
    hipErrcheck(hipFree(work));
    hipErrcheck(hipFree(devInfo));
}

static inline
void xheevd_wrapper(
    const hipsolverFillMode_t& uplo,
    const int& n,
    double* A, const int& lda,
    double* W)
{
    // prepare some values for hipsolverDnZhegvd_bufferSize
    int* devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double* work = nullptr;
    hipErrcheck(hipMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    hipsolverErrcheck(hipsolverDnDsyevd_bufferSize(hipsolver_handle, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
        A, lda, W, &lwork));
    // allocate memery
    hipErrcheck(hipMalloc((void**)&work, sizeof(double) * lwork));
    // compute eigenvalues and eigenvectors.
    hipsolverErrcheck(hipsolverDnDsyevd(hipsolver_handle, HIPSOLVER_EIG_MODE_VECTOR, uplo, n, A, lda, W, work, lwork, devInfo));

    hipErrcheck(hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost));
    assert(0 == info_gpu);
    hipErrcheck(hipFree(work));
    hipErrcheck(hipFree(devInfo));
}

static inline
void xheevd_wrapper (
        const hipsolverFillMode_t& uplo,
        const int& n,
        std::complex<float> * A, const int& lda,
        float * W)
{
    // prepare some values for hipsolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    float2 * work = nullptr;
    hipErrcheck(hipMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    hipsolverErrcheck(hipsolverDnCheevd_bufferSize(hipsolver_handle, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const float2 *>(A), lda, W, &lwork));
    // allocate memery
    hipErrcheck(hipMalloc((void**)&work, sizeof(float2) * lwork));
    // compute eigenvalues and eigenvectors.
    hipsolverErrcheck(hipsolverDnCheevd(hipsolver_handle, HIPSOLVER_EIG_MODE_VECTOR, uplo, n, reinterpret_cast<float2 *>(A), lda, W, work, lwork, devInfo));

    hipErrcheck(hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost));
    assert(0 == info_gpu);
    hipErrcheck(hipFree(work));
    hipErrcheck(hipFree(devInfo));
}

static inline
void xheevd_wrapper (
        const hipsolverFillMode_t& uplo,
        const int& n,
        std::complex<double> * A, const int& lda,
        double * W)
{
    // prepare some values for hipsolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double2 * work = nullptr;
    hipErrcheck(hipMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    hipsolverErrcheck(hipsolverDnZheevd_bufferSize(hipsolver_handle, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const double2 *>(A), lda, W, &lwork));
    // allocate memery
    hipErrcheck(hipMalloc((void**)&work, sizeof(double2) * lwork));
    // compute eigenvalues and eigenvectors.
    hipsolverErrcheck(hipsolverDnZheevd(hipsolver_handle, HIPSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<double2 *>(A), lda, W, work, lwork, devInfo));

    hipErrcheck(hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost));
    assert(0 == info_gpu);
    hipErrcheck(hipFree(work));
    hipErrcheck(hipFree(devInfo));
}

template <typename T>
struct dngvd_op<T, psi::DEVICE_GPU> {
    using Real = typename GetTypeReal<T>::type;
    void operator()(
        const psi::DEVICE_GPU* d,
        const int nstart,
        const int ldh,
        const T* A, // hcc
        const T* B, // scc
        Real* W, // eigenvalue
        T* V)
    {
        assert(nstart == ldh);
        // A to V
        hipErrcheck(hipMemcpy(V, A, sizeof(T) * ldh * nstart, hipMemcpyDeviceToDevice));
        xhegvd_wrapper(HIPSOLVER_FILL_MODE_UPPER, nstart, V, ldh,
            (T*)B, ldh, W);
    }
};

template <typename T>
struct dnevx_op<T, psi::DEVICE_GPU> {
    using Real = typename GetTypeReal<T>::type;
    void operator()(
        const psi::DEVICE_GPU* d,
        const int nstart,
        const int ldh,
        const T* A, // hcc
        const int m,
        Real* W, // eigenvalue
        T* V)
    {
        assert(nstart <= ldh);
        // A to V
        hipErrcheck(hipMemcpy(V, A, sizeof(T) * nstart * ldh, hipMemcpyDeviceToDevice));
        xheevd_wrapper(HIPSOLVER_FILL_MODE_LOWER, nstart, V, ldh, W);
    }
};

template struct dngvd_op<std::complex<float>, psi::DEVICE_GPU>;
template struct dnevx_op<std::complex<float>, psi::DEVICE_GPU>;
template struct dngvd_op<std::complex<double>, psi::DEVICE_GPU>;
template struct dnevx_op<std::complex<double>, psi::DEVICE_GPU>;

#ifdef __LCAO
template struct dngvd_op<double, psi::DEVICE_GPU>;
template struct dnevx_op<double, psi::DEVICE_GPU>;
#endif

} // namespace hsolver