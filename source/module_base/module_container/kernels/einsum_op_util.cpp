#include "einsum_op_util.h"
#include "../tensor_map.h"

#include <string>
#include <vector>
#include <numeric>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

namespace container {
namespace einsum_utils {

// Do some initialization work for bcast dimensions
static BCast prepare_bcast(
        std::vector<int64_t>& x_,
        std::vector<int64_t>& y_)
{
    const std::vector<int64_t> x(x_.begin(), x_.end() - 2);
    const std::vector<int64_t> y(y_.begin(), y_.end() - 2);

    // Safely multiplies dimensions taking into account symbolic shapes.
    auto mul_dims = [](int64_t dim1, int64_t dim2) -> int64_t {
        if (dim1 != 0 && dim2 != 0 && (dim1 < 0 || dim2 < 0)) {
            return -1;
        }
        return dim1 * dim2;
    };

    BCast bcast = {};
    bool all_equal = x == y;
    size_t lagest_batch_rank = std::max(x.size(), y.size());

    // calculate the all_equal and lagest_rank
    // There can be at most two operands, so we can use a 2 for loop size
    if (all_equal) {
        bcast.requires_broadcast = false;
        // Fast path for common case of identical shapes.
        int64_t batch_size = 1;
        const int rank = x.size();
        for (int ii = 0; ii < rank; ++ii) {
            bcast.z_batch_shape.resize(rank);
            for (int ii = 0; ii < x.size(); ii++) {
                bcast.z_batch_shape[ii] = x[ii];
                batch_size = mul_dims(batch_size, x[ii]);
            }
        }
        bcast.x_bcast_shape.push_back(1);
        bcast.y_bcast_shape.push_back(1);
        bcast.z_batch_size = batch_size;
        bcast.x_batch_shape.push_back(batch_size);
        bcast.y_batch_shape.push_back(batch_size);
        
        return std::move(bcast);
    }
    
    std::vector<int64_t> inv_x = x;
    std::vector<int64_t> inv_y = y;
    BCast::reverse(inv_x);
    BCast::reverse(inv_y);

    // 1-extend and align all vectors.
    inv_x.resize(lagest_batch_rank, 1);
    inv_y.resize(lagest_batch_rank, 1);

    // going through each dimension starting from the inner-most
    // dimension, compares dimension of x and y. They are compatible if
    // they are equal or either is 1.

    // indices of j-th component of each input.
    int64_t output_dim = -1;
    bool x_prev_is_one = false, y_prev_is_one = false;
    bool x_current_is_one = false, y_current_is_one = false;
    bool output_dim_set = false, none_is_one = true, set_one = false;
    for (int ii = 0; ii < lagest_batch_rank; ii++) {
        // Pre condition setting
        output_dim = -1;
        output_dim_set = false;
        none_is_one = true;
        if (inv_x[ii] == 1) {
            none_is_one = false;
            x_current_is_one = true;
        }
        else {
            x_current_is_one = false;
            output_dim = inv_x[ii];
            output_dim_set = true;
        }
        if (inv_x[ii] == 1) {
            none_is_one = false;
            y_current_is_one = true;
        }
        else {
            y_current_is_one = false;
            if (!output_dim_set || output_dim == inv_y[ii]) {
                output_dim = inv_y[ii];
                output_dim_set = true;
            }
            else {
                bcast.valid = false;
                return std::move(bcast);
            }
        }
        bcast.z_batch_shape.push_back(output_dim_set ? output_dim : 1);
        bcast.z_batch_size = mul_dims(bcast.z_batch_size, bcast.z_batch_shape.back());
        
        // All dimensions are 1
        if (!output_dim_set) {
            // This will skip updating the previous state to the current one. We'll
            // explain why this is safe below.
            // Consider the previous state P, current state C and the next state N.
            // In the case where N also is all ones (N == C), we'll do the same
            // optimization here (push back one dimensions if we need to), which is
            // safe and is expected.
            //
            // When N != C, we'll continue as usual. However, we might trigger the
            // next block if N == P (because we didn't update the previous state).
            // We trigger the next block if `fewer_dims_optimization` is true.
            // This means that we did not modify and broadcast / reshapes in this
            // block (we skipped updating, since the one dimensions can be ignored).
            // In essence, we only need to check whether the previous non-one state is
            // equal to the current non-one state.
            continue;
        }
        else if (x_current_is_one == x_prev_is_one && y_current_is_one == y_prev_is_one && set_one) {
            // fewer_dims_optimization
            // If the previous state is the same as the current state, we can skip
            // broadcasting / reshaping. This is because we can ignore dimensions of
            // size 1. This is safe because we know that the previous state is not
            // all ones (otherwise we would have continued in the previous block).

            // It is a run of the same broadcasting case as last time.
            // We can reshape the input so that fewer dimensions
            // are involved in the intermediate computation.
            bcast.x_batch_shape.back() = mul_dims(bcast.x_batch_shape.back(), inv_x[ii]);
            bcast.y_batch_shape.back() = mul_dims(bcast.y_batch_shape.back(), inv_y[ii]);

            bcast.x_bcast_shape.back() = mul_dims(bcast.x_bcast_shape.back(), x_current_is_one ? output_dim : 1);
            bcast.y_bcast_shape.back() = mul_dims(bcast.y_bcast_shape.back(), y_current_is_one ? output_dim : 1);
        }
        else {
            bcast.x_batch_shape.push_back(inv_x[ii]);
            bcast.y_batch_shape.push_back(inv_y[ii]);

            bcast.x_bcast_shape.push_back(x_current_is_one ? output_dim : 1);
            bcast.y_bcast_shape.push_back(y_current_is_one ? output_dim : 1);
        }
        set_one = true;
        x_prev_is_one = x_current_is_one;
        y_prev_is_one = y_current_is_one;
    }
    if (bcast.x_batch_shape.empty()) {
        bcast.x_batch_shape.push_back(1);
        bcast.x_bcast_shape.push_back(1);
    }
    if (bcast.y_batch_shape.empty()) {
        bcast.y_batch_shape.push_back(1);
        bcast.y_bcast_shape.push_back(1);
    }

    // Do something about batches
    BCast::reverse(bcast.x_batch_shape);
    BCast::reverse(bcast.x_bcast_shape);
    BCast::reverse(bcast.y_batch_shape);
    BCast::reverse(bcast.y_bcast_shape);
    BCast::reverse(bcast.z_batch_shape);

    return std::move(bcast);
}

static int64_t IPow(int64_t base, int64_t exponent) {
    int64_t result = 1;
    for (int64_t i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

// Returns a reshaped input Tensor. The underlying buffer is not copied.
static bool CopyFrom(Tensor& input, const TensorShape& shape, Tensor& output)
{
    output = TensorMap(input.data(), input.data_type(), input.device_type(), input.shape());
    return true;
}

// Reshapes a Tensor of shape [b0,b1...bk,N,M] to [prod(b0,b1...bk),N,M].
static bool ReshapeToRank3(Tensor& input, int batch_size, Tensor& output)
{
    const int rank = input.shape().ndim();
    TensorShape output_shape = {batch_size, input.shape().dim_size(rank - 2),
                                input.shape().dim_size(rank - 1)};
    return CopyFrom(input, output_shape, output);
}

template <typename T>
static bool all_of(const std::vector<T>& vec, bool (*predicate)(T)) {
    for (const auto& element : vec) {
        if (!predicate(element)) {
            return false;
        }
    }
    return true;
}

// If there are repeated labels in either the input or output, then this
// strides the input (e.g. iii->i) or inflates it (e.g. i->iii), respectively.
template <typename T, typename Device>
static bool StrideOrInflate(
        const Tensor& input,
        const std::vector<int>& labels,
        const std::vector<int>& label_counts,
        const bool should_inflate,
        Tensor& output) // output is the result of stride or inflate
{
    // Return early if there are no repeated indices.
    if (all_of(label_counts, [](int c) { return c <= 1; })) {
      return CopyFrom(input, input.shape(), output);
    }

    // We reshape so that each repeated label is compressed to one dimension.
    // E.g. For iiij -> ij, The shape [3, 3, 3, 5] would be compressed to [27,
    // 5]. Striding appropriately (in this case with strides 14 (=1+3+9) and 1)
    // recovers the generalized diagonal of shape [3, 5].
    std::vector<int64_t> reshape;
    std::vector<int64_t> strides;
    // Strided and inflated shapes correspond to input and output shapes,
    // respectively, should_inflate is true (vice-versa if should_inflate is
    // false). E.g. they are [3, 5] and [3, 3, 3, 5] in the above example.
    std::vector<int64_t> strided_shape;
    std::vector<int64_t> inflated_shape;
    for (int label : labels) {
        const int count = label_counts[label];
        const int current_axis =
            should_inflate ? strided_shape.size() : inflated_shape.size();
        const int64_t dim = input.shape().dim_size(current_axis);
        strided_shape.push_back(dim);
        inflated_shape.insert(inflated_shape.end(), count, dim);
        const int64_t reshape_dim = IPow(dim, count);
        reshape.push_back(reshape_dim);
        // While taking the d-diagonal in a rank k Tensor, we take d
        // equally-spaced elements including the first and last element. Then, (k
        // - 1) * stride = d^k - 1, or, stride = (d^k - 1)/(d - 1).
        const int64_t stride =
            (dim > 1 && count > 1) ? (reshape_dim - 1) / (dim - 1) : 1;
        strides.push_back(stride);
    }

    TensorShape output_shape =
        TensorShape(should_inflate ? inflated_shape : strided_shape);
    output.resize(output_shape);

    if (should_inflate) {
        input.reshape(strided_shape);
        output.reshape(reshape);
        op::inflate_op<T, DEVICE>()(input, strides, output);
    } else {
        input.reshape(reshape);
        op::stride_op<T, DEVICE>()(input, strides, output);
    }

    return true;     
}

// Permutes the labels according to the given permutation.
static void PermuteLabels(
        const std::vector<int>& permutation,
        std::vector<int>& labels)
{
    const int num_labels = labels.size();
    std::vector<int> permuted_labels(num_labels, 0);
    for (int ii = 0; ii < num_labels; ii++) {
        permuted_labels[ii] = labels[permutation[ii]];
    }
    labels.swap(permuted_labels);
}

// Returns whether transposing would be a no-op; whether input has rank < 2 or
// the permutation is the identity permutation.
static bool ShouldTranspose(
        const TensorShape& input_shape,
        const std::vector<int>& permutation) 
{
    if (input_shape.ndim() < 2) return false;
    for (int ii = 0; ii < permutation.size(); ++ii) {
      if (permutation[ii] != ii) return true;
    }
    return false;
}

// Transpose the input given a permutation. Returns a reference to the input
// if transposing is not necessary.
template <typename T, typename Device>
static bool TransposeOperand(
        const Tensor& input,
        const std::vector<int>& permutation,
        Tensor& output)
{
    if (!ShouldTranspose(input.shape(), permutation)) {
        return CopyFrom(input, input.shape(), output);
    }
    TensorShape transposed_shape;
    for (int ii = 0; ii < input.dims(); ++ii) {
        transposed_shape.AddDim(input.dim_size(permutation[ii]));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input.NumElements() == 0) {
        return CopyFrom(input, input.shape(), output);
    }

    // Allocate a temporary buffer for the transposed Tensor.
    output.resize(transposed_shape);
    if (input.data_type() != DataType::DT_COMPLEX || 
        input.data_type() != DataType::DT_COMPLEX_DOUBLE) {
        op::transpose_op<T, DEVICE, false>()(input, permutation, output);
    } else {
        op::transpose_op<T, DEVICE, true>()(input, permutation, output);
    }
    return true;
}

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
    // Check the number of ellipsis.
    if (std::count(labels.begin(), labels.end(), kEllipsisLabel) > 1) {
        throw std::invalid_argument("More than one ellipsis in subscript: " + subscript);
    }
}

/// Check the validation of the input equations
bool ValidateEinsumEquation(
        std::string& equation,
        std::vector<std::string>& input_subscripts,
        std::string& output_subscript)
{
    // Part 1: Check the equation's validation
    if (equation.empty()) {
        throw std::invalid_argument("Empty einsum equation");
    }

    // Part 2: Remove the white space in the equation
    std::string equation_no_space;
    for (const char c : equation) {
        if (c != ' ') {
            equation_no_space.push_back(c);
        }
    }
    equation = std::move(equation_no_space);

    // Part 3: Check the "->" flag
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

    // Part 4: Address the comma in the input subscripts
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
        // if (label == kEllipsisLabel) continue; Not necessary here.
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
inline bool reduce_oprand<T, Device>::operator()(
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
    std::iota(permutation.begin(), permutation.end(), 0);

    Tensor input_transposed;
    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    if(ShouldSwapFreeAndContract(labels, label_types)) {
        *swap_free_and_contract = true;
    } else {
        std::sort(permutation, [&](int ii, int jj) {
            int label_ii = labels[ii];
            int label_jj = labels[jj];
            return std::tie(label_types[label_ii], label_ii) <
                   std::tie(label_types[label_jj], label_jj);
        });
    }

    // Transpose the input so that EinsumDimensionTypes are in order.
    TransposeOperand<T, Device>(input, permutation, input_transposed);
    // Permutes labels
    PermuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    // This is necessary for the Reduce/Contract operations.
    Tensor input_deduped;
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

    StrideOrInflate<T, Device>(input_transposed, labels, label_counts, false, input_deduped);

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each EinsumDimensionType.
    std::vector<int64_t> reshape(5, 1);
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.   
    TensorShape output_shape;
    for (int label_idx = 0; label_idx < labels.size(); label_idx++) {
        const int label = labels[label_idx];
        int64_t dim = input_deduped.dim_size(label_idx);
        if (label_types[label] == EinsumDimensionType::kBroadcasting ||
            label_types[label] == EinsumDimensionType::kBatch) {
              output_shape.add_dim(dim);
        } else if (label_types[label] == EinsumDimensionType::kFree) {
              free_labels.push_back(label);
        }
        // All together, the reshape is [broadcast, batch, free, contract, reduce]
        reshape[label_types[label]] *= dim;
    }

    if (*swap_free_and_contract) {
        std::swap(reshape[EinsumDimensionType::kFree], reshape[EinsumDimensionType::kContract]);
    }
    output_shape.add_dim(reshape[EinsumDimensionType::kFree]);
    output_shape.add_dim(reshape[EinsumDimensionType::kContract]);

    if (reshape[EinsumDimensionType::kReduce] ==
        1) {  // No need to actually reduce.
      return CopyFrom(input_deduped, output_shape, output);
    }

    output.resize(output_shape);
    output.reshape({output.NumElements()});
    input_deduped.rehsape({output.NumElements(), reshape[EinsumDimensionType::kReduce]})
    
    op::reduce_op<T, Device>()(input_deduped, reshape[EinsumDimensionType::kReduce], output);
    return true;
}

// Contracts the inputs along the last axis (or the second last if the
// corresponding value of swap_free_and_contract is true). The batch
// dimensions are broadcast to the output shape.
// TODO(anudhyan): BatchMatMul might devolve into a component-wise
// multiplication when the matrix shape is [1,1]; in this case BatchMatMul
// functor would be very inefficient. The functor should detect if this is the
// case and perform componentwise multiplication functor instead.
template <typename T, typename Device>
bool contract_oprands<T, Device>::operator()(
        const std::vector<Tensor>& inputs,
        const std::vector<bool>& swap_free_and_contract,
        Tensor& output)
{
    if (inputs.size() == 1) {
        return CopyFrom(inputs[0], inputs[0].shape(), output);
    }
    BCast bcast = prepare_bcast(inputs[0].shape().dims(), inputs[1].shape().dims());

    if (bcast.valid == false) {
        throw std::invalid_argument("Invalid broadcast shape");
    }
    Tensor lhs, rhs;
    ReshapeToRank3(inputs[0], bcast.x_batch_size, lhs);
    ReshapeToRank3(inputs[1], bcast.y_batch_size, lhs);

    TensorShape output_shape = bcast.z_batch_shape;
    for (int ii = 0; ii < inputs.size(); ii++) {
      const int64_t free_axis =
          inputs[ii].shape().ndim() - (swap_free_and_contract[i] ? 1 : 2);
          output_shape.add_dim(inputs[i].dim_size(free_axis)));
    }
    bool trans_x = swap_free_and_contract[0];
    bool trans_y = !swap_free_and_contract[1];

    output.resize(output_shape);
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
        output.zero();
        return true;
    }

    Tensor output_reshaped;
    ReshapeToRank3(output, bcast.z_batch_size, output_reshaped);
    op::contract_op<T, Device>()(lhs, rhs, trans_x, trans_y, bcast, output_reshaped);

    return true;
}

}   // namespace einsum_utils
}   // namespace container
