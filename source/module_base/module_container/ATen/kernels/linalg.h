#ifndef ATEN_KERNELS_LINALG_H_
#define ATEN_KERNELS_LINALG_H_

#include <ATen/core/tensor.h>

#include <ATen/ops/einsum_op.h>

namespace container {
namespace kernels {

template <bool Conjugate = false>
struct transpose {
    /**
     * @brief Perform the transpose operation on the input tensor.
     *
     * @param input The input tensor to be transposed.
     * @param permutation A vector representing the desired permutation of the
     *                    dimensions. The elements of the vector indicate the new
     *                    order of the dimensions, and the size of the vector must
     *                    be equal to the rank of the input tensor.
     * @param output The output tensor that will store the result of the transpose
     *               operation. The output tensor should be pre-allocated and have
     *               the correct shape for the transpose result.
     *
     * @return True if the transpose operation is successful; otherwise, false.
     *
     * @note The size and shape of the output tensor must match the new dimensions
     *       specified by the permutation vector. If the permutation is not valid or
     *       the output tensor is not pre-allocated with the correct shape, the
     *       function will return false.
     */
    void operator()(
        const Tensor& input,
        const std::vector<int>& permutation,
        Tensor& output);
};

/**
 * @brief A functor to perform stride operation on a Tensor.
 *
 * This functor applies a stride operation on the input Tensor, resulting in a new
 * Tensor with data elements selected based on the specified stride. The stride
 * specifies how elements are skipped in each dimension during the selection process.
 *
 * @tparam T The data type of the Tensor.
 * @tparam Device The execution device (e.g., CPU or GPU).
 */
template <typename T, typename Device>
struct stride {
    /**
     * @brief Perform stride operation on the input Tensor.
     *
     * This function applies the stride operation on the input Tensor, generating a new
     * Tensor with data elements selected based on the specified stride.
     *
     * @param input The input Tensor on which the stride operation is performed.
     * @param stride The stride specification, represented as a TensorShape.
     *               The stride specifies how elements are skipped in each dimension during
     *               the selection process.
     * @param output The output Tensor that will hold the result of the stride operation.
     *               It must have the appropriate size to store the selected elements.
     */
    void operator()(
        const Tensor& input,
        const TensorShape& stride,
        Tensor& output);
};

/**
 * @brief A functor for inflating a tensor.
 *
 * This struct defines a functor that can be used to inflate a tensor using the specified stride.
 */
template <typename T, typename Device>
struct inflate {
    /**
     * @brief Inflate the input tensor.
     *
     * This function inflates the input tensor using the given stride and stores the result in the output tensor.
     *
     * @param input The input tensor to be inflated.
     * @param stride The stride to use for inflation.
     * @param output The output tensor where the inflated data will be stored.
     */
    void operator()(
        const Tensor& input,
        const TensorShape& stride,
        Tensor& output);
};


/**
 * @brief A functor to perform reduction operation on a Tensor along the inner-most dimension.
 *
 * This functor applies a reduction operation on a given input Tensor along the inner-most dimension.
 * The reduction operation could be any operation that combines the elements along a specific dimension
 * into a single result, such as sum, mean, max, etc.
 *
 * @tparam T The data type of the elements in the Tensor.
 * @tparam Device The device on which the Tensor resides (CPU/GPU).
 * 
 * @TOTO: Add support for axis reduction operations.
 */
template <typename T, typename Device>
struct reduce {
    /**
     * @brief Perform the reduction operation on the input Tensor.
     *
     * This function applies the reduction operation on the input Tensor along the inner-most dimension
     * and stores the result in the output Tensor.
     *
     * @param input The input Tensor on which the reduction operation will be applied.
     * @param inner_most_dim The dimension along which the reduction will be performed.
     * @param output The output Tensor to store the result of the reduction operation.
     */
    void operator()(
        const Tensor& input,
        const int64_t& inner_most_dim,
        Tensor& output);
};

} // namespace op
} // namespace container

#endif // ATEN_KERNELS_LINALG_H_