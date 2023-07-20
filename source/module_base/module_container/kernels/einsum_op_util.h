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
    std::vector<std::string>& output_subscript);

// Parses and validates the equation and the input shapes. Single character
// labels are integerized, and we populate input and output label subscripts
// and corresponding counts. Also create the mapping from (named) labels to
// their EinsumDimensionType.
bool ParseEinsumEquation(
    const std::string& equation, std::vector<std::string>& input_labels,
    std::vector<std::string>& output_labels,
    std::vector<EinsumDimensionType>* label_types,
    std::vector<int>& input_label_counts,
    std::vector<int>& output_label_counts,
    std::vector<bool>& input_has_ellipsis,
    std::vector<bool>& output_has_ellipsis);

bool RecordLabelToDimension(
    const int label,
    const int axis,
    const Tensor& input,
    std::unordered_map<int, int64_t>& label_to_dim_sizes);

}   // namespace utils
}   // namespace container

#endif  // CONTAINER_KERNELS_EINSUM_OP_UTIL_H_
