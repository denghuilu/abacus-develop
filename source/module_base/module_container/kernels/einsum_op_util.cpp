#include "einsum_op_util.h"

#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <unordered_map>

namespace container {
namespace utils {

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

// Returns the EinsumDimensionType given whether the corresponding label is
// present in exactly one input subscript (is_unique) and whether it is absent
// from the output subscripts (is_removed). Does not handle broadcasting
// dimensions.
EinsumDimensionType GetDimensionType(bool is_removed, bool is_unique)
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
void MapToLabels(const std::string &subscript, std::vector<int>& labels,
                 std::unordered_map<char, int>& label_mapping)
{
    for (int ii = 0; ii < subscript.size(); ii++) {
        const char label_char = subscript[ii];
        if (label_char == '.') {
            labels.push_back(kEllipsisLabel);
            ii += 2;  // Skip next 2 characters as well.
            continue;
        }
        if (label_mapping.find(label_char) == label_mapping.end()) {
            const int next_label = label_mapping.size();
            label_mapping[label_char] = next_label;
        }
        const int mapped_label = label_mapping[label_char];
        labels.push_back(mapped_label);
    }
}

// Preprocessing for the input equation expr
bool ParseEinsumEquation(
        const std::string& equation,
        std::vector<std::vector<int>>& input_labels,
        std::vector<int>& output_labels,
        std::vector<EinsumDimensionType>& label_types,
        std::vector<std::unordered_map<int, int>>& input_label_counts,
        std::unordered_map<int, int>& output_label_counts,
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
    for (int ii = 0; ii < num_inputs; ii++) {
        MapToLabels(input_str[ii], input_labels[ii], label_mapping);
    }
    MapToLabels(output_str, output_labels, label_mapping);

    // Compute counts for input and output labels.
    int num_labels = label_mapping.size();
    input_label_counts.resize(num_inputs);
    input_has_ellipsis.resize(num_inputs);
    for (int ii = 0; ii < num_inputs; ii++) {
        input_label_counts[ii].reserve(num_labels);
        input_has_ellipsis[ii] = false;
        for (const int label : input_labels[ii]) {
            if (label != kEllipsisLabel) {
                input_label_counts[ii][label] += 1;
            } else {
                input_has_ellipsis[ii] = true;
            }
        }
    }
    output_label_counts.reserve(num_labels);
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
        bool unique = num_inputs == 1 || (input_label_counts)[0][label] == 0 ||
                      (input_label_counts)[1][label] == 0;
        (label_types)[label] = GetDimensionType(removed, unique);
    }
    return true;
}

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

}  // namespace op
}  // namespace container
