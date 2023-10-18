#ifndef ATEN_KERNELS_BINARY_H_
#define ATEN_KERNELS_BINARY_H_

#include <base/macros/macros.h>

namespace container {
namespace kernels {
/**
 * @brief A functor to perform binary operations on tensors.
 *
 * This functor performs binary operations on tensors. The binary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the binary operation.
 */
template <typename T, typename Device>
struct add_func {
    /**
     * @brief Perform the binary operation on the input tensors.
     *
     * @param lhs The first input element.
     * @param rhs The second input element.
     * @param op The function object that performs the binary operation.
     *
     * @return The result element.
     */
    ATEN_HOST_DEVICE ATEN_ALWAYS_INLINE T operator()(const T& lhs, const T& rhs) {
        return lhs + rhs;
    }
};

/**
 * @brief A functor to perform binary operations on tensors.
 *
 * This functor performs binary operations on tensors. The binary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the binary operation.
 */
template <typename T>
struct sub_func {
    /**
     * @brief Perform the binary operation on the input tensors.
     *
     * @param lhs The first input element.
     * @param rhs The second input element.
     * @param op The function object that performs the binary operation.
     *
     * @return The result element.
     */
    ATEN_HOST_DEVICE ATEN_ALWAYS_INLINE T operator()(const T& lhs, const T& rhs) {
        return lhs - rhs;
    }
};

/**
 * @brief A functor to perform binary operations on tensors.
 *
 * This functor performs binary operations on tensors. The binary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the binary operation.
 */
template <typename T, typename Device>
struct mul_func {
    /**
     * @brief Perform the binary operation on the input tensors.
     *
     * @param lhs The first input element.
     * @param rhs The second input element.
     * @param op The function object that performs the binary operation.
     *
     * @return The result element.
     */
    ATEN_HOST_DEVICE ATEN_ALWAYS_INLINE T operator()(const T& lhs, const T& rhs) {
        return lhs * rhs;
    }
};

/**
 * @brief A functor to perform binary operations on tensors.
 *
 * This functor performs binary operations on tensors. The binary operation is
 * specified by the provided function object.
 *
 * @tparam Func The type of the function object that performs the binary operation.
 */
template <typename T>
struct div_func {
    /**
     * @brief Perform the binary operation on the input tensors.
     *
     * @param lhs The first input element.
     * @param rhs The second input element.
     * @param op The function object that performs the binary operation.
     *
     * @return The result element.
     */
    ATEN_HOST_DEVICE ATEN_ALWAYS_INLINE T operator()(const T& lhs, const T& rhs) {
        return lhs / rhs;
    }
};

} // namespace func
} // namespace base

#endif // ATEN_KERNELS_BINARY_H_