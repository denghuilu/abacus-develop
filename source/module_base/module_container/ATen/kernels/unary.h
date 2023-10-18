#ifndef ATEN_KERNELS_UNARY_H_
#define ATEN_KERNELS_UNARY_H_

#include <base/macros/macros.h>

namespace container {
namespace kernels {

/**
 * @brief A functor to perform unary operations on tensors.
 *
 * This functor performs unary operations on tensors. The unary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the unary operation.
 */
template <typename T, typename Device>
struct negnate_func {
    /**
     * @brief Perform the unary operation on the input tensor.
     *
     * @param self The input tensor.
     * @param op The function object that performs the unary operation.
     *
     * @return The result tensor.
     */
    Func operator()(const T& item);
};

/**
 * @brief A functor to perform unary operations on tensors.
 *
 * This functor performs unary operations on tensors. The unary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the unary operation.
 */
template <typename T>
struct sin_func {
    /**
     * @brief Perform the unary operation on the input tensor.
     *
     * @param self The input tensor.
     * @param op The function object that performs the unary operation.
     *
     * @return The result tensor.
     */
    ATEN_HOST_DEVICE ATEN_ALWAYS_INLINE T operator()(const T& item) {
        return sin(item);
    }
};

/**
 * @brief A functor to perform unary operations on tensors.
 *
 * This functor performs unary operations on tensors. The unary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the unary operation.
 */
template <typename T>
struct cos_func {
    /**
     * @brief Perform the unary operation on the input tensor.
     *
     * @param self The input tensor.
     * @param op The function object that performs the unary operation.
     *
     * @return The result tensor.
     */
    ATEN_HOST_DEVICE ATEN_ALWAYS_INLINE T operator()(const T& item) {
        return cos(item);
    }
};

/**
 * @brief A functor to perform unary operations on tensors.
 *
 * This functor performs unary operations on tensors. The unary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the unary operation.
 */
template <typename T>
struct conj_func {
    /**
     * @brief Perform the unary operation on the input tensor.
     *
     * @param self The input tensor.
     * @param op The function object that performs the unary operation.
     *
     * @return The result tensor.
     */
    ATEN_HOST_DEVICE ATEN_ALWAYS_INLINE T operator()(const T& item) {
        return conj(item);
    }
};

} // namespace kernels
} // namespace container

#endif // ATEN_KERNELS_UNARY_H_