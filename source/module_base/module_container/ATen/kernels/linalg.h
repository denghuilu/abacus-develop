#ifndef ATEN_KERNELS_LINALG_H_
#define ATEN_KERNELS_LINALG_H_

#include <ATen/core/tensor.h>

#include <ATen/ops/einsum_op.h>

namespace container {
namespace kernels {

template <typename T_1, typename T_2, typename Device>
struct add {
    // z = alpha * x + beta * y
    void operator()(
        const int& num_element,
        const T_1& alpha,
        const T_1* x,
        const T_1& beta,
        const T_2* y,
        T_1* z);
};

template <typename T_1, typename T_2, typename Device>
struct mul {
    void operator()(
        const int& num_element,
        const T_1& alpha,
        const T_1* x,
        T_1* y);
    // z = alpha * x * y
    void operator()(
        const int& num_element,
        const T_1& alpha,
        const T_1* x,
        const T_2* y,
        T_1* z);
};

template <typename T_1, typename T_2, typename Device>
struct div {
    // z = alpha * x / y
    void operator()(
        const int& num_element,
        const T_1& alpha,
        const T_1* x,
        const T_2* y,
        T_1* z);
    
    void operator()(
        const int& num_element,
        const T_1& alpha,
        const T_1* x,
        const T_2& y,
        T_1* z);
};

template <typename T, typename Device>
struct fma {
    // out = alpha * x * y + beta * z
    void operator()(
        const int& num_element,
        const T& alpha,
        const T* x,
        const T* y,
        const T& beta,
        const T* z,
        T* out);
};

template <typename T, typename Device, bool Conjugate = false>
struct transpose {
    void operator()(
        const std::vector<int>& perm,
        const std::vector<int64_t>& p_shape,
        const std::vector<int64_t>& q_shape,
        const T* p,
        T* q);
};


template <typename T, typename Device>
struct stride {
    void operator()(
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& p_shape,
        const std::vector<int64_t>& q_shape,
        const T* p,
        T* q);
};

template <typename T, typename Device>
struct inflate {
    void operator()(
        const std::vector<int64_t>& inflate,
        const std::vector<int64_t>& p_shape,
        const std::vector<int64_t>& q_shape,
        const T* p,
        T* q);
};


template <typename T, typename Device>
struct reduce {
    void operator()(
        const int64_t& num_element,
        const int64_t& inner_most_dim,
        const T* p,
        T* q);
};


} // namespace op
} // namespace container

#endif // ATEN_KERNELS_LINALG_H_