#include "einsum_op_util.h"

#include <string>
#include <vector>
#include <numeric>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

namespace container {
namespace utils {

// Returns true if the input dimensions are already sorted in the order
// [broadcasting, batch, contract, free, reduce]. Used to implement an optimization to avoid
// an extra transpose and instead uses (adj_x and adj_y) in BatchMatMul.
static bool ShouldSwapFreeAndContract(
        const std::vector<int>& labels,
        const std::vector<EinsumDimensionType>& label_types)
{
    // Check that ordering is according to dimension type, with the role of
    // free and contract dimensions swapped.
    std::vector<int> remap = {0, 1, 3, 2, 4};
    for (int ii = 0; ii < labels.size() - 1; ii++) {
        const int dimtype_a = remap[label_types[labels[ii]]];
        const int dimtype_b = remap[label_types[labels[ii + 1]]];
        if (dimtype_a > dimtype_b || (dimtype_a == dimtype_b && labels[ii] > labels[ii + 1])) {
            return false;
        }
    }
    return true;
}

// Insert new (unnamed) broadcasting labels at the location of ellipsis.
static void InsertBroadcastLabels(
        int num_bcast_labels, 
        int num_named_labels,
        int ellipsis_idx,
        std::vector<int>& labels,
        std::vector<int>& label_counts) 
{
    labels.erase(labels.begin() + ellipsis_idx);
    labels.insert(labels.begin() + ellipsis_idx, num_bcast_labels, 0);

    // Does the padding ellipsis overlap with any named labels?
    // Insert new labels at the location of ellipsis.
    // Now I understand finally!
    // Start from the num_named_labels, and insert num_bcast_labels
    // These broadcasting labels are not overlapped with the named labels
    std::iota(labels.begin() + ellipsis_idx,
              labels.begin() + ellipsis_idx + num_bcast_labels,
              num_named_labels);

    label_counts.resize(num_named_labels + num_bcast_labels, 1);
}

// Returns the EinsumDimensionType given whether the corresponding label is
// present in exactly one input subscript (is_unique) and whether it is absent
// from the output subscripts (is_removed). Does not handle broadcasting
// dimensions.
static EinsumDimensionType GetDimensionType(bool is_removed, bool is_unique)
{
    if (!is_removed && !is_unique)
        return kBatch;
    else if (!is_removed && is_unique)
        return kFree;
    else if (is_removed && !is_unique)
        return kContract;
    else  // is_removed && is_unique
        return kReduce;
}

// Maps the character labels to consecutive integers.
static void MapToLabels(const std::string& subscript, std::vector<int>& labels,
                 std::unordered_map<char, int>& label_mapping)
{
    for (int ii = 0; ii < subscript.size(); ii++) {
        const char label_char = subscript[ii];
        if (label_char == '.') {
            // Ellipsis is a special case.
            if (subscript[ii + 1] != '.' || subscript[ii + 2] != '.') {
                throw std::invalid_argument("Invalid ellipsis in subscript: " + subscript);
            }
            labels.push_back(kEllipsisLabel);
            ii += 2;  // Skip next 2 characters as well.
            continue;
        }
        // Check that the label is a valid character.
        // Don't worry about the ellipsis character as it is handled above.
        if (label_mapping.find(label_char) == label_mapping.end()) {
            const int next_label = label_mapping.size();
            label_mapping[label_char] = next_label;
        }
        // Map the label to an integer.
        const int mapped_label = label_mapping[label_char];
        labels.push_back(mapped_label);
    }
}

/// Check the validation of the input equations
bool ValidateEinsumEquation(
        const std::string& equation,
        std::vector<std::string>& input_subscripts,
        std::string& output_subscript)
{
    /// Part 1: Check the "->" flag
    std::vector<std::string> inputs_and_output_subscripts;
    auto delimiter_pos = equation.find("->");
    if (delimiter_pos == std::string::npos) {
        throw std::invalid_argument("No '->' in einsum equation: " + equation);
    }
    else if (equation.find("->", delimiter_pos + 1) != std::string::npos) {
        throw std::invalid_argument("Expecting exactly one '->' in einsum equation: " + equation);
    }
    inputs_and_output_subscripts.push_back(equation.substr(0, delimiter_pos));
    inputs_and_output_subscripts.push_back(equation.substr(delimiter_pos + 2));

    output_subscript = std::move(inputs_and_output_subscripts[1]);

    auto comma_pos = inputs_and_output_subscripts[0].find(',');
    while (comma_pos != std::string::npos) {
        input_subscripts.push_back(inputs_and_output_subscripts[0].substr(0, comma_pos));
        inputs_and_output_subscripts[0] = inputs_and_output_subscripts[0].substr(comma_pos + 1);
        comma_pos = inputs_and_output_subscripts[0].find(',');
    }
    input_subscripts.push_back(inputs_and_output_subscripts[0]);

    if (input_subscripts.size() != 1 && input_subscripts.size() != 2) {
        throw std::invalid_argument("Expecting 1 or 2 input subscripts in equation '" + equation +
                                    "' but got: " + std::to_string(input_subscripts.size()));
    }
    return true;
}

// Preprocessing for the input equation expr
bool ParseEinsumEquation(
        const std::string& equation,
        std::vector<std::vector<int>>& input_labels,
        std::vector<int>& output_labels,
        std::vector<EinsumDimensionType>& label_types,
        std::vector<std::vector<int>>& input_label_counts,
        std::vector<int>& output_label_counts,
        std::vector<bool>& input_has_ellipsis,
        bool& output_has_ellipsis)
{
    // Check the equation's validation
    std::vector<std::string> input_str;
    std::string output_str;
    if (!ValidateEinsumEquation(equation, input_str, output_str)) {
        return false;
    }
    std::unordered_map<char, int> label_mapping;
    int num_inputs = input_str.size();
    input_labels.resize(num_inputs);

    // Map from single characters to integer labels.
    // Labels that are identical in the output equation and distinct input equations are assigned the same integer mapping.
    for (int ii = 0; ii < num_inputs; ii++) {
        MapToLabels(input_str[ii], input_labels[ii], label_mapping);
    }
    MapToLabels(output_str, output_labels, label_mapping);

    // Compute counts for input and output labels.
    int num_labels = label_mapping.size();
    input_label_counts.resize(num_inputs);
    input_has_ellipsis.resize(num_inputs);
    for (int ii = 0; ii < num_inputs; ii++) {
        input_label_counts[ii].resize(num_labels, 0);
        input_has_ellipsis[ii] = false;
        for (const int label : input_labels[ii]) {
            if (label != kEllipsisLabel) {
                input_label_counts[ii][label] += 1;
            } else {
                input_has_ellipsis[ii] = true;
            }
        }
    }
    output_label_counts.resize(num_labels, 0);
    output_has_ellipsis = false;
    for (const int label : output_labels) {
        if (label != kEllipsisLabel) {
            output_label_counts[label] += 1;
        } else {
            output_has_ellipsis = true;
        }
    }

    // Map each label to a unique EinsumDimensionType.
    label_types.resize(num_labels);
    for (int label = 0; label < num_labels; label++) {
        if (label == kEllipsisLabel) continue;
        bool removed = (output_label_counts)[label] == 0;
        bool unique = num_inputs == 1 || input_label_counts[0][label] == 0 ||
                      input_label_counts[1][label] == 0;
        (label_types)[label] = GetDimensionType(removed, unique);
    }
    return true;
}

// Records the dimension size for the given label. Checks that the dimension
bool RecordLabelToDimension(const int label, const int axis, const Tensor& input,
                            std::unordered_map<int, int64_t>& label_to_dim_sizes) {
    const int64_t input_dim = input.shape().dim_size(axis);
    auto& label_dim = label_to_dim_sizes[label];
    if (label_dim != 0 && label_dim != input_dim) {
        throw std::invalid_argument(
            "Expected dimension " + std::to_string(label_to_dim_sizes[label]) + " at axis " +
            std::to_string(axis) + " of the input shaped " +
            " but got dimension " + std::to_string(input_dim));
    }
    label_to_dim_sizes[label] = input_dim;
    return true;
}

// Validate input dimensions and populate unnamed labels and their label counts.
// Also populate the mapping from named labels to their dimension sizes.
bool ProcessDimensions(
        const std::vector<Tensor>& inputs,
        const std::vector<bool>& input_has_ellipsis,
        const bool output_has_ellipsis,
        std::vector<std::vector<int>>& input_labels,
        std::vector<int>& output_labels,
        std::vector<EinsumDimensionType>& label_types,
        std::vector<std::vector<int>>& input_label_counts,
        std::vector<int>& output_label_counts,
        std::unordered_map<int, int64_t>& label_to_dim_sizes) 
{
    const int num_inputs = inputs.size();
    const int num_labels = label_types.size();
    int max_bcast_dims = 0;
    // Check that the number of dimensions match for each label.
    for (int ii = 0; ii < num_inputs; ii++) {
        const Tensor& input = inputs[ii];
        const int num_dims = input.shape().ndim();
        const std::vector<int>& labels = input_labels[ii];
        const std::vector<int>& label_counts = input_label_counts[ii];
        bool has_ellipsis = input_has_ellipsis[ii];
        // We infer the number of broadcasting dimensions by taking the maximum rank
        // among the broadcasting subshapes of the input.
        if (!has_ellipsis) {
            // If there is no ellipsis, the number of dimensions must match the number
            if (num_dims != labels.size()) {
                throw std::invalid_argument(
                    "Input " + std::to_string(ii) + " has " + std::to_string(num_dims) +
                    " dimensions but " + std::to_string(num_labels) + " labels");
            }
            for (int label_idx = 0; label_idx < labels.size(); label_idx++) {
                const int label = labels[label_idx];
                // if (label == kEllipsisLabel) continue;
                // Double counting of labels is allowed. No need to check.
                // if (label_counts[label] > 1) {
                //     throw std::invalid_argument(
                //         "Label " + std::to_string(label) + " appears more than once in input " +
                //         std::to_string(ii));
                // }
                RecordLabelToDimension(label, label_idx, input, label_to_dim_sizes);
            }
            continue;
        }

        // Input has an ellipsis.
        // There are two cases:
        // 1. The ellipsis shadows at least one label: num_dims >= labels.size().
        // 2. The ellipsis shadows no labels: num_dims == labels.size() - 1.
        // In both cases, num_dims >= labels.size() - 1.
        if (num_dims < labels.size() - 1) {
            throw std::invalid_argument(
                "Input " + std::to_string(ii) + " has " + std::to_string(num_dims) +
                " dimensions but " + std::to_string(num_labels) + " labels");
        }
        int ellipsis_idx = -1;
        // Number of shadowed labels = num_dims - (labels.size() - 1).
        const int num_bcast_labels = num_dims - labels.size() + 1;
        for (int label_idx = 0; label_idx < labels.size(); label_idx++) {
            const int label = labels[label_idx];
            // Find the ellipsis axis.
            if (label == kEllipsisLabel) {
                ellipsis_idx = label_idx;
                continue;
            }
            // Current label is not an ellipsis.
            // There are two cases:
            // 1. The ellipsis axis is not found yet: axis = label_idx.
            // 2. The ellipsis axis is found: axis = label_idx - 1 + num_bcast_labels.
            const int axis = label_idx + (ellipsis_idx == -1 ? 0 : num_bcast_labels - 1);
            RecordLabelToDimension(label, axis, input, label_to_dim_sizes);
        }
        // Found an ellipsis. Replace it with the appropriate number of broadcasting
        // labels.
        if (ellipsis_idx != -1) {
            InsertBroadcastLabels(num_bcast_labels, num_labels, ellipsis_idx, input_labels[ii],
                                  input_label_counts[ii]);
            max_bcast_dims = std::max(max_bcast_dims, num_bcast_labels);
        }
    }
    if (!input_has_ellipsis[0] &&!input_has_ellipsis[1] && !output_has_ellipsis) {
        return true;
    }
    // Insert broadcasting labels into the output labels.
    auto it = std::find(output_labels.begin(), output_labels.end(), kEllipsisLabel);
    if (it != output_labels.end()) {
        const int ellipsis_idx = it - output_labels.begin();
        InsertBroadcastLabels(max_bcast_dims, num_labels, ellipsis_idx,
                              output_labels, output_label_counts);
    } else if (max_bcast_dims > 0) {
        throw std::invalid_argument(
            "Output has no ellipsis but input has ellipsis. Cannot insert broadcasting labels.");
    }
    // Polupate EinsumDimensionType for the new broadcasting labels.
    label_types.resize(num_labels + max_bcast_dims, EinsumDimensionType::kBroadcasting);
    return true;
}

template <typename T, typename Device>
inline bool reduce_op<T, Device>::operator()(
        const Tensor& input,
        const std::vector<EinsumDimensionType>& label_types,
        const std::vector<int>& label_counts,
        const std::vector<int>& labels,
        const std::vector<int>& free_labels,
        bool* swap_free_and_contract,
        Tensor& output)
{
    // Find the permutation to transpose the input dimensions in the order of
    // EinsumDimensionType; i.e. batch, free, contract and reduce dimensions.
    // This makes it more convenient to invoke Reduce/Contract operations.
    std::vector<int> permutation(input.shape().ndim(), 0);

    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    
}

}  // namespace utils
}  // namespace container
