#include "module_hsolver/kernels/dngvd_op.h"
#include "helper_cuda.h"

#include <cusolverDn.h>

#define cusolverErrcheck(res)                      \
    {                                              \
        cusolverAssert((res), __FILE__, __LINE__); \
    }

// cuSOLVER API errors
static const char* _cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
    }
    return "<unknown>";
}

template<typename FPTYPE>
cudaDataType_t get_cuda_data_type (FPTYPE * /* datatype */);

template<>
cudaDataType_t get_cuda_data_type (float * /* datatype */) {
    return CUDA_C_32F;
}

template<>
cudaDataType_t get_cuda_data_type (double * /* datatype */) {
    return CUDA_C_64F;
}


inline void cusolverAssert(cusolverStatus_t code, const char* file, int line, bool abort = true)
{
    if (code != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuSOLVER Assert: %s %s %d\n", _cusolverGetErrorEnum(code), file, line);
        if (abort)
            exit(code);
    }
}

namespace hsolver
{

static cusolverDnHandle_t cusolver_H = nullptr;

void createCUSOLVERhandle()
{
    if (cusolver_H == nullptr)
    {
        cusolverErrcheck(cusolverDnCreate(&cusolver_H));
    }
}

void destoryCUSOLVERhandle()
{
    if (cusolver_H != nullptr)
    {
        cusolverErrcheck(cusolverDnDestroy(cusolver_H));
        cusolver_H = nullptr;
    }
}

static inline
void xpotrf_wapper (const cublasFillMode_t& uplo, const int& n, std::complex<float> * A, const int& lda)
{
    int lwork;
    cusolverDnCpotrf_bufferSize(cusolver_H, uplo, n, reinterpret_cast<float2 *>(A), n, &lwork);
    float2* work;
    cudaMalloc((void**)&work, lwork * sizeof(float2));
    // Perform Cholesky decomposition
    cusolverDnCpotrf(cusolver_H, uplo, n, reinterpret_cast<float2 *>(A), n, work, lwork, nullptr);
    cudaFree(work);
}

static inline
void xpotrf_wapper (const cublasFillMode_t& uplo, const int& n, std::complex<double> * A, const int& lda)
{
    int lwork;
    cusolverDnZpotrf_bufferSize(cusolver_H, uplo, n, reinterpret_cast<double2 *>(A), n, &lwork);
    double2* work;
    cudaMalloc((void**)&work, lwork * sizeof(double2));
    // Perform Cholesky decomposition
    cusolverDnZpotrf(cusolver_H, uplo, n, reinterpret_cast<double2 *>(A), n, work, lwork, nullptr);
    cudaFree(work);
}

static inline
void xhegvd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<float> * A, const int& lda,
        std::complex<float> * B, const int& ldb,
        float * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    float2 * work = nullptr;
    checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnChegvd_bufferSize(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const float2 *>(A), lda,
                                                 reinterpret_cast<const float2 *>(B), ldb, W, &lwork));
    // allocate memery
    checkCudaErrors(cudaMalloc((void**)&work, sizeof(float2) * lwork));

    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnChegvd(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<float2 *>(A), lda, reinterpret_cast<float2 *>(B), ldb, W, work, lwork, devInfo));

    checkCudaErrors(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    assert(0 == info_gpu);
    // free the buffer
    checkCudaErrors(cudaFree(work));
    checkCudaErrors(cudaFree(devInfo));
}

static inline
void xhegvd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<double> * A, const int& lda,
        std::complex<double> * B, const int& ldb,
        double * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double2 * work = nullptr;
    checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnZhegvd_bufferSize(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const double2 *>(A), lda,
                                                 reinterpret_cast<const double2 *>(B), ldb, W, &lwork));
    // allocate memery
    checkCudaErrors(cudaMalloc((void**)&work, sizeof(double2) * lwork));

    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnZhegvd(cusolver_H, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<double2 *>(A), lda, reinterpret_cast<double2 *>(B), ldb, W, work, lwork, devInfo));

    checkCudaErrors(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    assert(0 == info_gpu);
    // free the buffer
    checkCudaErrors(cudaFree(work));
    checkCudaErrors(cudaFree(devInfo));
}

static inline
void xheevd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<float> * A, const int& lda,
        float * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    float2 * work = nullptr;
    checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnCheevd_bufferSize(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const float2 *>(A), lda, W, &lwork));
    // allocate memery
    checkCudaErrors(cudaMalloc((void**)&work, sizeof(float2) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnCheevd(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n, reinterpret_cast<float2 *>(A), lda, W, work, lwork, devInfo));

    checkCudaErrors(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    assert(0 == info_gpu);
    checkCudaErrors(cudaFree(work));
    checkCudaErrors(cudaFree(devInfo));
}

static inline
void xheevd_wrapper (
        const cublasFillMode_t& uplo,
        const int& n,
        std::complex<double> * A, const int& lda,
        double * W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int * devInfo = nullptr;
    int lwork = 0, info_gpu = 0;
    double2 * work = nullptr;
    checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnZheevd_bufferSize(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                                 reinterpret_cast<const double2 *>(A), lda, W, &lwork));
    // allocate memery
    checkCudaErrors(cudaMalloc((void**)&work, sizeof(double2) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnZheevd(cusolver_H, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                                      reinterpret_cast<double2 *>(A), lda, W, work, lwork, devInfo));

    checkCudaErrors(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    assert(0 == info_gpu);
    checkCudaErrors(cudaFree(work));
    checkCudaErrors(cudaFree(devInfo));
}

template <typename FPTYPE>
struct dngvd_op<FPTYPE, psi::DEVICE_GPU> {
    void operator()(
            const psi::DEVICE_GPU *d,
            const int nstart,
            const int ldh,
            const std::complex<FPTYPE> *A, // hcc
            const std::complex<FPTYPE> *B, // scc
            FPTYPE *W, // eigenvalue
            std::complex<FPTYPE> *V)
    {
        assert(nstart == ldh);
        // A to V
        checkCudaErrors(cudaMemcpy(V, A, sizeof(std::complex<FPTYPE>) * ldh * nstart, cudaMemcpyDeviceToDevice));
        xhegvd_wrapper(CUBLAS_FILL_MODE_UPPER, nstart, V, ldh,
                       (std::complex<FPTYPE> *)B, ldh, W);
    }
};

template <typename FPTYPE>
struct dnevx_op<FPTYPE, psi::DEVICE_GPU> {
    void operator()(
            const psi::DEVICE_GPU *d,
            const int nstart,
            const int ldh,
            const std::complex<FPTYPE> *A, // hcc
            const int m,
            FPTYPE *W, // eigenvalue
            std::complex<FPTYPE> *V)
    {
        assert(nstart <= ldh);
        // A to V
        checkCudaErrors(cudaMemcpy(V, A, sizeof(std::complex<FPTYPE>) * nstart * ldh, cudaMemcpyDeviceToDevice));
        xheevd_wrapper(CUBLAS_FILL_MODE_LOWER, nstart, V, ldh, W);
    }
};

template <typename FPTYPE>
struct dnevd_op<FPTYPE, psi::DEVICE_GPU> {
    void operator()(const psi::DEVICE_GPU* dev, std::complex<FPTYPE>* A, const int& dim, FPTYPE* W)
    {
        // A to V
        xheevd_wrapper(CUBLAS_FILL_MODE_UPPER, dim, A, dim, W);
    }
};

template <typename FPTYPE>
struct zpotrf_op<FPTYPE, psi::DEVICE_GPU> {
    void operator()(const psi::DEVICE_GPU* /*dev*/,std::complex<FPTYPE>* A, const int& dim)
    {
        xpotrf_wapper(CUBLAS_FILL_MODE_UPPER, dim, A, dim);
    }
};

template <typename FPTYPE>
struct ztrtri_op<FPTYPE, psi::DEVICE_GPU> {
    void operator()(const psi::DEVICE_GPU* /*dev*/,std::complex<FPTYPE>* A, const int& dim) {
        FPTYPE *d_work = nullptr;
        size_t d_lwork = 0;
        FPTYPE *h_work = nullptr;
        size_t h_lwork = 0;

        cusolverDnXtrtri_bufferSize(cusolver_H, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, dim,
                                    get_cuda_data_type<FPTYPE>(d_work),
                                    reinterpret_cast<void *>(A), dim, &d_lwork,
                                    &h_lwork);

        cudaMalloc(reinterpret_cast<void **>(&d_work), d_lwork);

        if (h_lwork) {
            h_work = (FPTYPE *)malloc(h_lwork);
        }

        cusolverDnXtrtri(cusolver_H, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, dim, get_cuda_data_type<FPTYPE>(d_work),
                         reinterpret_cast<void *>(A), dim,
                         d_work, d_lwork, h_work, h_lwork, nullptr);

        cudaFree(d_work);
        free(h_work);
    }
};

template struct zpotrf_op<float, psi::DEVICE_GPU>;
template struct ztrtri_op<float, psi::DEVICE_GPU>;
template struct dngvd_op<float, psi::DEVICE_GPU>;
template struct dnevd_op<float, psi::DEVICE_GPU>;
template struct dnevx_op<float, psi::DEVICE_GPU>;
template struct dngvd_op<double, psi::DEVICE_GPU>;
template struct dnevx_op<double, psi::DEVICE_GPU>;
template struct dnevd_op<double, psi::DEVICE_GPU>;
template struct zpotrf_op<double, psi::DEVICE_GPU>;
template struct ztrtri_op<double, psi::DEVICE_GPU>;

} // namespace hsolver