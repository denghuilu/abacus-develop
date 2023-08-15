#ifndef CONTAINER_KERNELS_EINSUM_OP_H_
#define CONTAINER_KERNELS_EINSUM_OP_H_

#include <tuple>
#include "../tensor.h"
#include "einsum_op_utils.h"

namespace container {
namespace op {

template<typename T>
std::vector<T*> mapToVector(const T& arg) {
    return std::vector<T*>{&arg};
}

// TODO: implement this method this week!
// piapia pat face

/**
 *
 * @brief Computes the Einstein summation convention on the given tensors
 * based on the expression passed as a string.
 *
 * @tparam Tensors Variadic template parameter pack for the input tensors.
 *
 * @param equation The Einstein summation convention expression.
 * @param tensors The input tensors for the summation operation.
 * @return The resulting tensor after applying the Einstein summation convention on the input tensors.
 *
 * @throws std::invalid_argument if the expression or the input tensors are invalid.
 * @throws std::runtime_error if an error occurs while performing the summation operation.
 */
template <typename... Tensors>
typename std::enable_if<std::is_same<
        typename std::common_type<Tensors...>::type, Tensor>::value, Tensor>::type
    einsum(const std::string& equation, const Tensors&... tensors)
{
    // Check the input dimension
    auto _tensors = std::make_tuple(tensors...);
    constexpr int num_inputs = sizeof...(Tensors);
    if (num_inputs > 2) {
        throw std::invalid_argument("Einstein notation only support two or less tensors!");
    }
    const std::vector<const Tensor*> inputs{reinterpret_cast<const Tensor*>(&tensors)...};
    // Init the input and output labels
    std::vector<std::vector<int>> input_labels = {};
    std::vector<int> output_labels = {};
    std::vector<utils::EinsumDimensionType> label_types = {};
    std::vector<std::vector<int>> input_label_counts = {};
    std::vector<int> output_label_counts = {};
    std::vector<bool> input_has_ellipsis = {};
    bool output_has_ellipsis = {};

    utils::ParseEinsumEquation(
        equation, label_types, 
        input_labels, output_labels, 
        input_label_counts, output_label_counts, 
        input_has_ellipsis, output_has_ellipsis);
    
    if (input_labels.size() != num_inputs) {
        throw std::runtime_error("The number of input tensors does not match the number of input labels!");
    }
    
    std::unordered_map<int, int64_t> label_to_dim_sizes = {};

    utils::ProcessDimensions(
        inputs, label_types,
        input_labels, output_labels, 
        input_label_counts, output_label_counts,
        input_has_ellipsis, output_has_ellipsis,
        label_to_dim_sizes);

    std::vector<std::vector<int>> free_labels(num_inputs);
    std::vector<int> swap_free_and_contract(num_inputs);
    std::vector<Tensor> inputs_reduced(num_inputs, Tensor(DataType::DT_INT, {}));

    for (int ii = 0; ii < num_inputs; ++ii) {
        utils::ReduceOperand(
            *inputs[ii], label_types,
            input_labels[ii], input_label_counts[ii],
            free_labels[ii], swap_free_and_contract[ii], inputs_reduced[ii]);
    }

    return std::move(Tensor(DataType::DT_INT, {}));
}

} // namespace op
} // namespace container

#endif // CONTAINER_KERNELS_EINSUM_OP_H_