#ifndef CONTAINER_KERNELS_TESTS_OP_TEST_UTILS_H_
#define CONTAINER_KERNELS_TESTS_OP_TEST_UTILS_H_
#include <gtest/gtest.h>
#include "module_base/module_container/kernels/blas_op.h"
#include "module_base/module_container/kernels/lapack_op.h"

namespace container {
namespace tests_utils {

# if __CUDA || __ROCM
using Types = ::testing::Types<
        std::tuple<float, DEVICE_CPU>, std::tuple<float, DEVICE_GPU>,
        std::tuple<double, DEVICE_CPU>, std::tuple<double, DEVICE_GPU>,
        std::tuple<std::complex<float>, DEVICE_CPU>, std::tuple<std::complex<float>, DEVICE_GPU>,
        std::tuple<std::complex<double>, DEVICE_CPU>,  std::tuple<std::complex<double>, DEVICE_GPU>>;
#else 
using Types = ::testing::Types<
        std::tuple<float, DEVICE_CPU>, 
        std::tuple<double, DEVICE_CPU>,
        std::tuple<std::complex<float>, DEVICE_CPU>,
        std::tuple<std::complex<double>, DEVICE_CPU>>;
#endif 

static inline void init_blas_handle() {
    #if __CUDA || __ROCM
        op::createBlasHandle();
    #endif
}

static inline void delete_blas_handle() {
    #if __CUDA || __ROCM
        op::destroyBlasHandle();
    #endif
}

static inline void init_cusolver_handle() {
    #if __CUDA || __ROCM
        op::createCusolverHandle();
    #endif
}

static inline void delete_cusolver_handle() {
    #if __CUDA || __ROCM
        op::destroyCusolverHandle();
    #endif
}

} // namespace tests_utils
} // namespace container

#endif // CONTAINER_KERNELS_TESTS_OP_TEST_UTILS_H_