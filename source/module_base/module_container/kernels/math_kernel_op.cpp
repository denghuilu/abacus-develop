#include "math_kernel_op.h"

#include <vector>

namespace container {
namespace op {

template <typename T>
static std::vector<T> ComputeStride(const std::initializer_list<int>& shape) {
    int ndims = shape.size();
    std::vector<T> strides(ndims);
    T stride = 1;

    auto it = shape.end(); // Start from the last element
    for (int i = ndims - 1; i >= 0; --i) {
        --it;
        strides[i] = stride;
        stride *= static_cast<T>(*it);
    }
    return strides;
}

template <typename T, typename Device, bool Conjugate>
void transpose_op<T, Device, Conjugate>::operator()(
    const Tensor& input,
    const std::vector<int>& perm,
    Tensor& output)
{
    const int ndim = input.shape().ndim();
    auto in_strides = ComputeStride<int64_t>(input.shape());
    auto out_strides = ComputeStride<int64_t>(output.shape());

    const T* p = reinterpret_cast<const T*>(in.data());
    T* q = reinterpret_cast<T*>((out.data()));

    // Define a lambda expression 'transpose_fn' to implement transpose operation.
    auto transpose_fn =
        [=, &in_strides, &out_strides, &perm](
        int64_t begin, int64_t end) 
    {
        // Perform transpose operation for the specified range [begin, end) in the output Tensor.
        for (int64_t o_idx = begin; o_idx < end; o_idx++) {
            int64_t i_idx = 0; // Initialize the index for the input Tensor element.
            int64_t t = o_idx; // Calculate the index for the output Tensor element.

            // Iterate over each dimension of the output Tensor.
            for (int ii = 0; ii < ndims; ++ii) {
                // Calculate the ratio of the current output Tensor index 't' in the current dimension.
                const int64_t ratio = t / out_strides[ii];
                // Update the output Tensor index 't' by removing the offset in the current dimension.
                t -= ratio * out_strides[ii];
                // Calculate the offset for the corresponding index position in the input Tensor and accumulate it in 'i_idx'.
                i_idx += ratio * in_strides[perm[ii]];
            }
            // Check if conjugation is needed.
            if (conjugate) {
                // Assign the conjugate value of the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
                q[o_idx] = std::conj(p[i_idx]);
            } else {
                // Assign the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
                q[o_idx] = p[i_idx];
            }
        }
    };
    // Perform transpose operation on the output Tensor.
    transpose_fn(0, output.shape().num_elements());
}


// TODO: implement the stride operation within the Tensor class.
template <typename T, typename Device>
void stride_op<T, Device>::operator()(
        const Tensor& input,
        const TensorShape& stride,
        Tensor& output)
{
    const int ndim = input.shape().ndim();
    auto in_strides = ComputeStride<int64_t>(input.shape());
    auto out_strides = ComputeStride<int64_t>(output.shape());

    const T* p = reinterpret_cast<const T*>(in.data());
    T* q = reinterpret_cast<T*>((out.data()));

    // Define a lambda expression 'stride_fn' to implement stride operation.
    auto stride_fn =
        [=, &in_strides, &out_strides, &stride](
        int64_t begin, int64_t end) 
    {
        // Perform stride operation for the specified range [begin, end) in the output Tensor.
        for (int64_t o_idx = begin; o_idx < end; o_idx++) {
            int64_t i_idx = 0; // Initialize the index for the input Tensor element.
            int64_t t = o_idx; // Calculate the index for the output Tensor element.

            // Iterate over each dimension of the output Tensor.
            for (int ii = 0; ii < ndim; ++ii) {
                // Calculate the ratio of the current output Tensor index 't' in the current dimension.
                const int64_t ratio = t / out_strides[ii];
                // Update the output Tensor index 't' by removing the offset in the current dimension.
                t -= ratio * out_strides[ii];
                // Calculate the offset for the corresponding index position in the input Tensor and accumulate it in 'i_idx'.
                i_idx += ratio * (in_strides[ii] + stride[ii]);
            }
            // Assign the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
            q[o_idx] = p[i_idx];
        }
    };
    // Perform stride operation on the output Tensor.
    stride_fn(0, output.shape().num_elements());
}

// TODO: implement the stride operation within the Tensor class.
template <typename T, typename Device>
void inflate_op<T, Device>::operator()(
        const Tensor& input,
        const TensorShape& stride,
        Tensor& output)
{
    const int ndim = input.shape().ndim();
    auto in_strides = ComputeStride<int64_t>(input.shape());
    auto out_strides = ComputeStride<int64_t>(output.shape());

    const T* p = reinterpret_cast<const T*>(in.data());
    T* q = reinterpret_cast<T*>((out.data()));

    // Define a lambda expression 'stride_fn' to implement stride operation.
    auto inflate_fn =
        [=, &in_strides, &out_strides, &stride](
        int64_t begin, int64_t end) 
    {
        // Perform stride operation for the specified range [begin, end) in the output Tensor.
        for (int64_t o_idx = begin; o_idx < end; o_idx++) {
            int64_t i_idx = 0; // Initialize the index for the input Tensor element.
            int64_t t = o_idx; // Calculate the index for the output Tensor element.

            bool valid = true;
            // Iterate over each dimension of the output Tensor.
            for (int ii = 0; ii < ndim; ++ii) {
                // Calculte the ratio of the current output Tensor index 't' in the current dimension.
                const int64_t ratio = t / out_strides[ii];
                // Update the output Tensor index 't' by removing the offset in the current dimension.
                t -= ratio * out_strides[ii];
                // Calculate the offset for the corresponding index position in the input Tensor and accumulate it in 'i_idx'.
                if (ratio % stride[ii] == 0) {
                    i_idx += ratio * (in_strides[ii] / stride[ii]);
                } else {
                    valid = false;
                    break;
                }
            }
            // Assign the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
            q[o_idx] = p[i_idx] * static_cast<T>(valid);
        }
    };
    // Perform stride operation on the output Tensor.
    stride_fn(0, output.shape().num_elements());
}

template <typename T, typename Device>
void reduce_op<T, Device>::operator()(
        const Tensor& input,
        const int64_t& inner_most_dim,
        Tensor& output)
{
    const T* p = reinterpret_cast<const T*>(in.data());
    T* q = reinterpret_cast<T*>((out.data()));

    // It's just so simple to implement the reduce operation.
    for (int64_t o_idx = 0; o_idx < output.NumElements(); o_idx++) {
        T sum = 0;
        for (int64_t i_idx = o_idx * inner_most_dim; i_idx < inner_most_dim + o_idx * inner_most_dim; i_idx++) {
            sum += p[i_idx];
        }
        q[o_idx] = sum;
    }
}

} // namespace utils
} // namespace container