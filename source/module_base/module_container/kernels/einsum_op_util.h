#ifndef CONTAINER_KERNELS_EINSUM_OP_UTIL_H_
#define CONTAINER_KERNELS_EINSUM_OP_UTIL_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "../tensor.h"

namespace container {
namespace utils {


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
    std::vector<std::vector<int>>& input_labels,
    std::vector<int>& output_labels,
    std::vector<EinsumDimensionType>& label_types,
    std::vector<std::vector<int>>& input_label_counts,
    std::vector<int>& output_label_counts,
    std::vector<bool>& input_has_ellipsis,
    bool& output_has_ellipsis);

/**
 * @brief Records and validates the label to dimension mapping for a given axis in the input tensor.
 *
 * This function records the mapping of a label to its corresponding dimension for a specific axis in the input tensor.
 * It also validates that the label and dimension mapping is consistent with previous recordings, ensuring that the 
 * same label is not mapped to different dimensions along different axes.
 *
 * @param label The label representing a dimension in the input tensor.
 * @param axis The axis index in the input tensor for which the label is being recorded.
 * @param input The input tensor for which the label-to-dimension mapping is being recorded.
 * @param label_to_dim_sizes An unordered map that stores the label-to-dimension mapping for all axes.
 *                           It maps each label to its corresponding dimension for each axis.
 *
 * @return Returns true if the label is successfully recorded and validated, false otherwise.
 */
bool RecordLabelToDimension(
    const int label,
    const int axis,
    const Tensor& input,
    std::unordered_map<int, int64_t>& label_to_dim_sizes);


/**
 * @brief Functor for performing reduce operation on a given Tensor.
 * 
 * This functor is used to execute a reduce operation on the input Tensor
 * based on the specified label types, counts, and labels. The resulting 
 * reduced Tensor is stored in the output Tensor.
 *
 * @tparam T The data type of the Tensor elements.
 * @tparam Device The device type where the Tensor operation is performed.
 */
template <typename T, typename Device>
struct reduce_op {
    /**
     * @brief Execute the reduce operation on the input Tensor.
     *
     * This function performs a reduce operation on the input Tensor based on
     * the provided label information. It reduces dimensions with specified 
     * labels and stores the result in the output Tensor.
     *
     * @param input The input Tensor on which the reduce operation is performed.
     * @param label_types A vector specifying the type of each input dimension. 
     *                    Possible types are EinsumDimensionType::kBroadcasting,
     *                    EinsumDimensionType::kFree, and EinsumDimensionType::kBatch.
     * @param label_counts A vector containing the counts of each label in the input Tensor.
     * @param labels A vector of integers representing the labels of the input Tensor dimensions.
     * @param free_labels A vector of integers representing the labels of free dimensions.
     * @param swap_free_and_contract A pointer to a boolean variable. If the value is set to true, 
     *                               it indicates that free and contract dimensions should be swapped.
     * @param output The output Tensor where the reduced result is stored.
     */
    bool operator()(
        const Tensor& input,
        const std::vector<EinsumDimensionType>& label_types,
        const std::vector<int>& label_counts,
        const std::vector<int>& labels,
        const std::vector<int>& free_labels,
        bool * swap_free_and_contract,
        Tensor& output);
};

} // namespace utils
}   // namespace container

#endif  // CONTAINER_KERNELS_EINSUM_OP_UTIL_H_
