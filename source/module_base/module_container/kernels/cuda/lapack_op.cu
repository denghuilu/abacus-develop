#include "../lapack_op.h"
#include "../third_party/lapack_connector.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cassert>
#include <algorithm>


namespace container {
namespace op {


static cusolverDnHandle_t cusolver_handle = nullptr;

void createCusolverHandle() {
    if (cusolver_handle == nullptr) {
        cusolverErrcheck(cusolverDnCreate(&cusolver_handle));
    }
}

void destoryCusolverHandle() {
    if (cusolver_handle != nullptr) {
        cusolverErrcheck(cusolverDnDestroy(cusolver_handle));
        cusolver_handle = nullptr;
    }
}

template <typename T>
struct potrf_op<T, psi::DEVICE_GPU> {
    void operator()(T* Mat, const int& dim) {
        cuSolverConector::potrf(cusolver_handle, 'U', dim, Mat, dim);
    }
};

template <typename T>
struct trtri_op<T, psi::DEVICE_GPU> {
    void operator()(T* Mat, const int& dim) {
        cuSolverConector::potri(cusolver_handle, 'L', dim, Mat, dim);
    }
};

template <typename T>
struct dnevd_op<T, psi::DEVICE_GPU> {
    using Real = typename PossibleComplexToReal<T>::type;
    void operator()(
        T* Mat,
        const int& dim,
        Real* eigen_val)
    {
        
    }
};

template <typename T>
struct dngvd_op<T, DEVICE_GPU> {
    using Real = typename PossibleComplexToReal<T>::type;
    void operator()(
        const int& dim,
        const T* Mat_A,
        const T* Mat_B,
        Real *eigen_val,
        T *eigen_vec)
    {
        
    }
};


template <typename T>
struct dnevx_op<T, DEVICE_GPU> {
    using Real = typename PossibleComplexToReal<T>::type;
    void operator()(
        const int row,
        const int col,
        const T* Mat_A,
        const int nband, 
        Real* eigen_val,
        T* eigen_vec)
    {
        
    }
};

template struct potrf_op<float,  DEVICE_GPU>;
template struct potrf_op<double, DEVICE_GPU>;
template struct potrf_op<std::complex<float>,  DEVICE_GPU>;
template struct potrf_op<std::complex<double>, DEVICE_GPU>;

template struct trtri_op<float,  DEVICE_GPU>;
template struct trtri_op<double, DEVICE_GPU>;
template struct trtri_op<std::complex<float>,  DEVICE_GPU>;
template struct trtri_op<std::complex<double>, DEVICE_GPU>;

template struct dnevd_op<float,  DEVICE_GPU>;
template struct dnevd_op<double, DEVICE_GPU>;
template struct dnevd_op<std::complex<float>,  DEVICE_GPU>;
template struct dnevd_op<std::complex<double>, DEVICE_GPU>;

template struct dngvd_op<float,  DEVICE_GPU>;
template struct dngvd_op<double, DEVICE_GPU>;
template struct dngvd_op<std::complex<float>,  DEVICE_GPU>;
template struct dngvd_op<std::complex<double>, DEVICE_GPU>;

template struct dnevx_op<float,  DEVICE_GPU>;
template struct dnevx_op<double, DEVICE_GPU>;
template struct dnevx_op<std::complex<float>,  DEVICE_GPU>;
template struct dnevx_op<std::complex<double>, DEVICE_GPU>;

} // namespace op
} // namespace container