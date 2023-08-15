#ifndef CONTAINER_KERNELS_EINSUM_OP_UTILS_H_
#define CONTAINER_KERNELS_EINSUM_OP_UTILS_H_

#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include "../tensor.h"

namespace container {
namespace utils {

class BCast {
  public:

    bool valid = true;
    bool requires_broadcast = false;

    int64_t x_batch_size = 1; // input a batch size
    int64_t y_batch_size = 1; // input b batch size
    int64_t z_batch_size = 1; // output c batch size

    std::vector<int64_t> x_batch_shape = {}; // input a shape
    std::vector<int64_t> y_batch_shape = {}; // input b shape
    std::vector<int64_t> z_batch_shape = {}; // output c shape

    std::vector<int64_t> x_bcast_shape = {}; // input a bcast shape
    std::vector<int64_t> y_bcast_shape = {}; // input b bcast shape

    static void reverse(std::vector<int64_t>& vec) {
        std::reverse(vec.begin(), vec.end());
    }
};

// Dummy axis label used to denote an ellipsis in an input or output subscript.
constexpr int kEllipsisLabel = -1;

// Each dimension is categorized into exactly one of five types based on
// whether its corresponding label is present in the input and/or the output
// subscripts.
enum EinsumDimensionType {
    // Batch dimensions are those present in two inputs as well as the output.
    // They are part of the batch dimensions during Tensor contraction. Such
    // dimensions may be broadcasting dimensions (those mapping to ellipsis)
    // or explicit batch dimensions corresponding to named axis labels.
    kBroadcasting = 0,
    kBatch = 1,
    // Free dimensions are present in exactly one of the inputs, and also the
    // output. These are non-contracted axes in the Tensor contraction.
    kFree = 2,
    // Contract dimensions are present in two inputs, but not the output. These
    // dimensions are contracted in Tensor contraction.
    kContract = 3,
    // Reduce dimensions are present in exactly one input; and not in the output
    // and are summed over prior to Tensor contraction.
    kReduce = 4,
};

// Parses and validates an einsum equation in explicit form.
bool ValidateEinsumEquation(
    const std::string& equation,
    std::vector<std::string>& input_subscripts,
    std::string& output_subscript);

// Parses and validates the equation and the input shapes. Single character
// labels are integerized, and we populate input and output label subscripts
// and corresponding counts. Also create the mapping from (named) labels to
// their EinsumDimensionType.
bool ParseEinsumEquation(
    const std::string& equation, 
    std::vector<EinsumDimensionType>& label_types,
    std::vector<std::vector<int>>& input_labels,
    std::vector<int>& output_labels,
    std::vector<std::vector<int>>& input_label_counts,
    std::vector<int>& output_label_counts,
    std::vector<bool>& input_has_ellipsis,
    bool& output_has_ellipsis);

bool ProcessDimensions(
    const std::vector<const Tensor*>& inputs,
    std::vector<EinsumDimensionType>& label_types,
    std::vector<std::vector<int>>& input_labels,
    std::vector<int>& output_labels,
    std::vector<std::vector<int>>& input_label_counts,
    std::vector<int>& output_label_counts,
    const std::vector<bool>& input_has_ellipsis,
    const bool output_has_ellipsis,
    std::unordered_map<int, int64_t>& label_to_dim_sizes);

// This function records the mapping of a label to its corresponding dimension for a specific axis in the input tensor.
// It also validates that the label and dimension mapping is consistent with previous recordings, ensuring that the 
// same label is not mapped to different dimensions along different axes.
bool RecordLabelToDimension(
    const int label,
    const int axis,
    const Tensor& input,
    std::unordered_map<int, int64_t>& label_to_dim_sizes);

bool ReduceOperand(
    const Tensor& input,
    const std::vector<EinsumDimensionType>& label_types,
    std::vector<int>& labels,
    const std::vector<int>& label_counts,
    std::vector<int>& free_labels,
    int& swap_free_and_contract,
    Tensor& output);

/**
 * @brief A functor to perform contraction operation on multiple Tensors.
 *
 * This functor applies a contraction operation on multiple input Tensors and computes the result.
 * The contraction operation combines the input Tensors based on a specific contraction pattern to produce
 * a single output Tensor. The contraction pattern is defined by the `swap_free_and_contract` vector, which
 * specifies whether each input Tensor should be contracted or simply copied to the output.
 *
 * @tparam T The data type of the elements in the Tensors.
 * @tparam Device The device on which the Tensors reside (CPU/GPU).
 */
template <typename T, typename Device>
struct contract_oprands {
    /**
     * @brief Perform the contraction operation on the input Tensors.
     *
     * This function applies the contraction operation on the input Tensors based on the specified
     * `swap_free_and_contract` pattern and stores the result in the output Tensor.
     *
     * @param inputs The vector of input Tensors to be contracted or copied.
     * @param swap_free_and_contract A vector specifying the contraction pattern for the input Tensors.
     *                              If an element is true, the corresponding input Tensor will be contracted;
     *                              otherwise, it will be copied to the output.
     * @param output The output Tensor to store the result of the contraction operation.
     *
     * @return True if the contraction operation was successful, false otherwise.
     */
    bool operator()(
        std::vector<Tensor>& inputs,
        const std::vector<bool>& swap_free_and_contract,
        Tensor& output);
};


}   // namespace utils
}   // namespace container

#endif  // CONTAINER_KERNELS_EINSUM_OP_UTILS_H_
