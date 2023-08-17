#include <ATen/core/tensor_types.h>
#include <ATen/kernels/einsum_op.h>
#include <ATen/kernels/linalg_op.h>

#include <algorithm>

namespace container {
namespace einsum_utils {

struct BCast {
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

// Do some initialization work for bcast dimensions
static BCast prepare_bcast(
    const std::vector<int64_t>& x_, 
    const std::vector<int64_t>& y_) 
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

static inline int64_t IPow(int64_t base, int64_t exponent) {
    int64_t result = 1;
    for (int64_t ii = 0; ii < exponent; ++ii) {
        result *= base;
    }
    return result;
}

// Returns a reshaped input Tensor. The underlying buffer is not copied.
// Note: This method will not allocate memory for output tensor, 
// intead use the reference of the input buffer.
static inline bool CopyFrom(const Tensor& input, const TensorShape& shape, Tensor* output) {
    return output->CopyFrom(input, shape);
}

// Note: This method will allocate memory with the given shape for output tensor,
// also keep the data_type, device_type consistant with the input tensor.
static inline bool CopyFromWithAllocate(const Tensor& input, const TensorShape& shape, Tensor* output) {
    return output->CopyFromWithAllocate(input, shape);
}

// Reshapes a Tensor of shape [b0,b1...bk,N,M] to [prod(b0,b1...bk),N,M].
static bool ReshapeToRank3(Tensor& input, int batch_size, Tensor& output)
{
    const int rank = input.shape().ndim();
    TensorShape output_shape = {batch_size, input.shape().dim_size(rank - 2),
                                input.shape().dim_size(rank - 1)};
    return CopyFrom(input, output_shape, &output);
}

template <typename T>
static inline bool all_of(
    const std::vector<T>& vec, 
    const std::function<bool(T)>& predicate) 
{
    for (const auto& element : vec) {
        if (!predicate(element)) {
            return false;
        }
    }
    return true;
}

// If there are repeated labels in either the input or output, then this
// strides the input (e.g. iii->i) or inflates it (e.g. i->iii), respectively.
static bool StrideOrInflateOperand(
        Tensor& input,
        const std::vector<int>& labels,
        const std::vector<int>& label_counts,
        const bool should_inflate,
        Tensor& output) // output is the result of stride or inflate
{
    // Return early if there are no repeated indices.
    if (all_of<int>(label_counts, [](int var) {return var <= 1;})) {
        output = input;
        return true;
    }

    // We reshape so that each repeated label is compressed to one dimension.
    // E.g. For iiij -> ij, The shape [3, 3, 3, 5] would be compressed to [27,
    // 5]. Striding appropriately (in this case with strides 14 (=1+3+9) and 1)
    // recovers the generalized diagonal of shape [3, 5].
    TensorShape reshape = {};
    TensorShape strides = {};
    // Strided and inflated shapes correspond to input and output shapes,
    // respectively, should_inflate is true (vice-versa if should_inflate is
    // false). E.g. they are [3, 5] and [3, 3, 3, 5] in the above example.
    std::vector<int64_t> strided_shape = {};
    std::vector<int64_t> inflated_shape = {};
    for (int label : labels) {
        const int count = label_counts[label];
        const int current_axis =
            should_inflate ? strided_shape.size() : inflated_shape.size();
        const int dim = input.shape().dim_size(current_axis);
        strided_shape.push_back(dim);
        inflated_shape.insert(inflated_shape.end(), count, dim);
        const int reshape_dim = IPow(dim, count);
        reshape.add_dim(reshape_dim);
        // While taking the d-diagonal in a rank k Tensor, we take d
        // equally-spaced elements including the first and last element. Then, (k
        // - 1) * stride = d^k - 1, or, stride = (d^k - 1)/(d - 1).
        const int stride =
            (dim > 1 && count > 1) ? (reshape_dim - 1) / (dim - 1) : 1;
        strides.add_dim(stride);
    }

    TensorShape output_shape =
        TensorShape(should_inflate ? inflated_shape : strided_shape);
    // Also allocate memory for the output tensor with the given shape.
    CopyFromWithAllocate(input, output_shape, &output);
 
    if (should_inflate) {
        input.reshape(strided_shape);
        output.reshape(reshape);
        TEMPLATE_ALL_2(input.data_type(), input.device_type(),
                   op::inflate_op<T_, DEVICE_>()(input, strides, output))
    } 
    else {
        input.reshape(reshape);
        TEMPLATE_ALL_2(input.data_type(), input.device_type(),
                   op::stride_op<T_, DEVICE_>()(input, strides, output))
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
static bool TransposeOperand(
        const Tensor& input,
        const std::vector<int>& permutation,
        Tensor& output)
{
    if (!ShouldTranspose(input.shape(), permutation)) {
        return CopyFrom(input, input.shape(), &output);
    }
    TensorShape transposed_shape;
    for (int ii = 0; ii < input.shape().ndim(); ++ii) {
        transposed_shape.add_dim(input.shape().dim_size(permutation[ii]));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input.NumElements() == 0) {
        return CopyFrom(input, input.shape(), &output);
    }

    // Note: This functor will allocate a buffer for the transposed Tensor.
    TEMPLATE_ALL_2(input.data_type(), input.device_type(),
                   op::transpose_op<T_, DEVICE_, false>()(input, permutation, output))

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
    std::array<int, 5> remap = {0, 1, 3, 2, 4};
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
        const std::string& equation,
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

    // Part 3: Check the "->" flag
    std::vector<std::string> inputs_and_output_subscripts;
    auto delimiter_pos = equation_no_space.find("->");
    if (delimiter_pos == std::string::npos) {
        throw std::invalid_argument("No '->' in einsum equation: " + equation_no_space);
    }
    else if (equation_no_space.find("->", delimiter_pos + 1) != std::string::npos) {
        throw std::invalid_argument("Expecting exactly one '->' in einsum equation: " + equation_no_space);
    }
    inputs_and_output_subscripts.push_back(equation_no_space.substr(0, delimiter_pos));
    inputs_and_output_subscripts.push_back(equation_no_space.substr(delimiter_pos + 2));

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
        throw std::invalid_argument("Expecting 1 or 2 input subscripts in equation '" + equation_no_space +
                                    "' but got: " + std::to_string(input_subscripts.size()));
    }

    // Part 5: Check the characters in the equation
    std::regex pattern("[^a-zA-Z\\.]+");
    if (input_subscripts.size() == 2) {
        if (std::regex_search(input_subscripts[0] + input_subscripts[1] + output_subscript, pattern)) {
            throw std::invalid_argument("Invalid character in einsum equation: " + equation);
        }
    }
    else if (input_subscripts.size() == 1) {
        if (std::regex_search(input_subscripts[0] + output_subscript, pattern)) {
            throw std::invalid_argument("Invalid character in einsum equation: " + equation);
        }
    }
    else {
        throw std::invalid_argument("Invalid einsum equation: " + equation);
    }
    return true;
}

// Preprocessing for the input equation expr
bool ParseEinsumEquation(
    const std::string& equation, 
    std::vector<EinsumDimensionType>& label_types,
    std::vector<std::vector<int>>& input_labels,
    std::vector<int>& output_labels,
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
        bool removed = output_label_counts[label] == 0;
        bool unique = num_inputs == 1 || input_label_counts[0][label] == 0 ||
                      input_label_counts[1][label] == 0;
        label_types[label] = GetDimensionType(removed, unique);
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
    const std::vector<const Tensor*>& inputs,
    std::vector<EinsumDimensionType>& label_types,
    std::vector<std::vector<int>>& input_labels,
    std::vector<int>& output_labels,
    std::vector<std::vector<int>>& input_label_counts,
    std::vector<int>& output_label_counts,
    const std::vector<bool>& input_has_ellipsis,
    const bool output_has_ellipsis,
    std::unordered_map<int, int64_t>& label_to_dim_sizes) 
{
    const int num_inputs = inputs.size();
    const int num_labels = label_types.size();
    int max_bcast_dims = 0;
    // Check that the number of dimensions match for each label.
    for (int ii = 0; ii < num_inputs; ii++) {
        const Tensor& input = *inputs[ii];
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
                    " dimensions but got" + std::to_string(num_labels) + " labels");
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
    if (!input_has_ellipsis[0] 
        && (input_has_ellipsis.size() == 1 || !input_has_ellipsis[1])
        && !output_has_ellipsis) 
    {
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


bool ReduceOperand(
    const Tensor& input,
    const std::vector<EinsumDimensionType>& label_types,
    std::vector<int>& labels,
    const std::vector<int>& label_counts,
    std::vector<int>& free_labels,
    int& swap_free_and_contract,
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
        swap_free_and_contract = 1;
    }
    else {
        std::sort(permutation.begin(), permutation.end(), [&](int ii, int jj) {
            int label_ii = labels[ii];
            int label_jj = labels[jj];
            return std::tie(label_types[label_ii], label_ii) <
                   std::tie(label_types[label_jj], label_jj);
        });
    }

    // Transpose the input so that EinsumDimensionTypes are in order.
    TransposeOperand(input, permutation, input_transposed);
    // Permutes labels
    PermuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    // This is necessary for the Reduce/Contract operations.
    Tensor input_deduped;
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());


    StrideOrInflateOperand(input_transposed, labels, label_counts, false, input_deduped);

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each EinsumDimensionType.
    std::array<int, 5> reshape = {1, 1, 1, 1, 1};
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.   
    TensorShape output_shape;
    for (int label_idx = 0; label_idx < labels.size(); label_idx++) {
        const int label = labels[label_idx];
        int dim = input_deduped.shape().dim_size(label_idx);
        if (label_types[label] == EinsumDimensionType::kBroadcasting ||
            label_types[label] == EinsumDimensionType::kBatch) {
              output_shape.add_dim(dim);
        } else if (label_types[label] == EinsumDimensionType::kFree) {
              free_labels.push_back(label);
        }
        // All together, the reshape is [broadcast, batch, free, contract, reduce]
        reshape[label_types[label]] *= dim;
    }

    if (swap_free_and_contract) {
        std::swap(reshape[EinsumDimensionType::kFree], reshape[EinsumDimensionType::kContract]);
    }
    output_shape.add_dim(reshape[EinsumDimensionType::kFree]);
    output_shape.add_dim(reshape[EinsumDimensionType::kContract]);


    if (reshape[EinsumDimensionType::kReduce] ==
        1) {  // No need to actually reduce.
      return CopyFrom(input_deduped, output_shape, &output);
    }

    input_deduped.reshape({-1, reshape[EinsumDimensionType::kReduce]});
    CopyFromWithAllocate(input_deduped, output_shape, &output);
    output.reshape({-1});
    
    TEMPLATE_ALL_2(input_deduped.data_type(), input_deduped.device_type(),
                   op::reduce_op<T_, DEVICE_>()(input_deduped, reshape[EinsumDimensionType::kReduce], output))

    return true;
}

// Contracts the inputs along the last axis (or the second last if the
// corresponding value of swap_free_and_contract is true). The batch
// dimensions are broadcast to the output shape.
// TODO(anudhyan): BatchMatMul might devolve into a component-wise
// multiplication when the matrix shape is [1,1]; in this case BatchMatMul
// functor would be very inefficient. The functor should detect if this is the
// case and perform componentwise multiplication functor instead.
bool ContractOperands(
    std::vector<Tensor>& inputs,
    const std::vector<bool>& swap_free_and_contract,
    Tensor& output)
{
    if (inputs.size() == 1) {
        return CopyFrom(inputs[0], inputs[0].shape(), &output);
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
          inputs[ii].shape().ndim() - (swap_free_and_contract[ii] ? 1 : 2);
          output_shape.add_dim(inputs[ii].shape().dim_size(free_axis));
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

    // TEMPLATE_ALL_2(output_reshaped.data_type(), output_reshaped.device_type(),
    //                einsum_utils::contract_op<T_, DEVICE_>()(lhs, rhs, trans_x, trans_y, bcast, output_reshaped))

    return true;
}


template <typename T, typename Device>
void contract_op<T, Device>::operator()(
        const Tensor& in_x,
        const Tensor& in_y,
        const bool& trans_x,
        const bool& trans_y,
        const einsum_utils::BCast& bcast,
        Tensor& out_z)
{
    const int64_t m = in_x.shape().dim_size(trans_x ? 2 : 1);
    const int64_t k = in_x.shape().dim_size(trans_x ? 1 : 2);
    const int64_t n = in_y.shape().dim_size(trans_y ? 1 : 2);

    const int64_t batch_size = bcast.z_batch_size;

    std::vector<T> x_device_memory = {}; x_device_memory.reserve(bcast.x_batch_size);
    std::vector<T> y_device_memory = {}; y_device_memory.reserve(bcast.y_batch_size);
    std::vector<T> z_device_memory = {}; z_device_memory.reserve(bcast.z_batch_size);

    std::vector<T*> x_device_memory_ptrs = {}; x_device_memory_ptrs.reserve(batch_size);
    std::vector<T*> y_device_memory_ptrs = {}; y_device_memory_ptrs.reserve(batch_size);
    std::vector<T*> z_device_memory_ptrs = {}; z_device_memory_ptrs.reserve(batch_size);

    auto* x_base_ptr = in_x.data<T>();
    auto* y_base_ptr = in_y.data<T>();
    auto* z_base_ptr = out_z.data<T>();

    int64_t x_stride = 0;
    int64_t y_stride = 0;
    int64_t z_stride = 0;

    bool is_full_broadcast = 
        std::min(bcast.x_batch_size, bcast.y_batch_size) == 1;

    bool use_strided_batched = 
        (!bcast.requires_broadcast || is_full_broadcast) && batch_size > 1;
    
    if (use_strided_batched) {
        x_stride = bcast.x_batch_size != 1 ? m * k : 0;
        y_stride = bcast.y_batch_size != 1 ? k * n : 0;
        z_stride = m * n;

        x_device_memory.push_back(x_base_ptr);
        y_device_memory.push_back(y_base_ptr);
        z_device_memory.push_back(z_base_ptr);

        x_device_memory_ptrs.push_back(&x_device_memory.back());
        y_device_memory_ptrs.push_back(&y_device_memory.back());
        z_device_memory_ptrs.push_back(&z_device_memory.back());
    }
    else if (!bcast.requires_broadcast) {
        for (int ii = 0; ii < batch_size; ii++) {
            x_device_memory.push_back(x_base_ptr + ii * m * k);
            y_device_memory.push_back(y_base_ptr + ii * k * n);
            z_device_memory.push_back(z_base_ptr + ii * m * n);

            x_device_memory_ptrs.push_back(&x_device_memory.back());
            y_device_memory_ptrs.push_back(&y_device_memory.back());
            z_device_memory_ptrs.push_back(&z_device_memory.back());
        }
    }
    else {
        for (int ii = 0; ii < bcast.x_batch_size; ii++) {
            x_device_memory.push_back(x_base_ptr + ii * m * k);
        }
        for (int ii = 0; ii < bcast.y_batch_size; ii++) {
            y_device_memory.push_back(y_base_ptr + ii * k * n);
        }
        for (int ii = 0; ii < bcast.z_batch_size; ii++) {
            z_device_memory.push_back(z_base_ptr + ii * m * n);
            
            x_device_memory_ptrs.push_back(&x_device_memory[bcast.x_batch_shape[ii]]);
            y_device_memory_ptrs.push_back(&y_device_memory[bcast.y_batch_shape[ii]]);
            z_device_memory_ptrs.push_back(&z_device_memory.back());
        }
    }

    // Do GEMM operations finally!
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A', where ' stands for transpose (not adjoint).
    // TODO (yangzihao): Choose the best of the three strategies using
    // autotune.
    if (batch_size == 1) {
        if (n == 1) {
            // op::gemv<>(m, n, k, x_device_memory_ptrs[0], y_device_memory_ptrs[0], z_device_memory_ptrs[0]);
        }
        else {
            // op::gemm<>(m, n, k, x_device_memory_ptrs[0], y_device_memory_ptrs[0], z_device_memory_ptrs[0]);
        }
        return;
    }
    else if (use_strided_batched) {
        // op::gemm_batched_strided<>(m, n, k, x_device_memory_ptrs, y_device_memory_ptrs, z_device_memory_ptrs, x_stride, y_stride, z_stride);
    }
    else {
        // op::gemm_batched_scrach<> scratchpad(batch_size, m, n, k);
    }
}

}   // namespace utils
}   // namespace container
