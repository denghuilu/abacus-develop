#include "math_kernel_op.h"

#include <vector>

namespace container {
namespace op {

template <typename T>
static std::vector<T> ComputeStride(const std::vector<int>& shape) {
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
    auto in_strides = ComputeStride<int64_t>(input.shape().dims());
    auto out_strides = ComputeStride<int64_t>(output.shape().dims());

    const T* p = reinterpret_cast<const T*>(input.data());
    T* q = reinterpret_cast<T*>((output.data()));

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
            for (int ii = 0; ii < ndim; ++ii) {
                // Calculate the ratio of the current output Tensor index 't' in the current dimension.
                const int64_t ratio = t / out_strides[ii];
                // Update the output Tensor index 't' by removing the offset in the current dimension.
                t -= ratio * out_strides[ii];
                // Calculate the offset for the corresponding index position in the input Tensor and accumulate it in 'i_idx'.
                i_idx += ratio * in_strides[perm[ii]];
            }
            // Check if conjugation is needed.
            if (Conjugate) {
                // Assign the conjugate value of the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
                q[o_idx] = std::conj(p[i_idx]);
            } else {
                // Assign the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
                q[o_idx] = p[i_idx];
            }
        }
    };
    // Perform transpose operation on the output Tensor.
    transpose_fn(0, output.shape().NumElements());
}


// TODO implement the stride operation within the Tensor class.
template <typename T, typename Device>
void stride_op<T, Device>::operator()(
        const Tensor& input,
        const TensorShape& stride,
        Tensor& output)
{
    const int ndim = input.shape().ndim();
    auto in_strides = ComputeStride<int64_t>(input.shape().dims());
    auto out_strides = ComputeStride<int64_t>(output.shape().dims());

    const T* p = reinterpret_cast<const T*>(input.data());
    T* q = reinterpret_cast<T*>((output.data()));

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
                i_idx += ratio * (in_strides[ii] + stride.dim_size(ii));
            }
            // Assign the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
            q[o_idx] = p[i_idx];
        }
    };
    // Perform stride operation on the output Tensor.
    stride_fn(0, output.shape().NumElements());
}

// TODO implement the stride operation within the Tensor class.
template <typename T, typename Device>
void inflate_op<T, Device>::operator()(
        const Tensor& input,
        const TensorShape& stride,
        Tensor& output)
{
    const int ndim = input.shape().ndim();
    auto in_strides = ComputeStride<int64_t>(input.shape().dims());
    auto out_strides = ComputeStride<int64_t>(output.shape().dims());

    const T* p = reinterpret_cast<const T*>(input.data());
    T* q = reinterpret_cast<T*>((output.data()));

    // Perform stride operation for the specified range [begin, end) in the output Tensor.
    for (int64_t o_idx = 0; o_idx < output.shape().NumElements(); o_idx++) {
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
            if (ratio % stride.dim_size(ii) == 0) {
                i_idx += ratio * (in_strides[ii] / stride.dim_size(ii));
            } else {
                valid = false;
                break;
            }
        }
        // Assign the input Tensor element at index 'i_idx' to the output Tensor element at index 'o_idx'.
        q[o_idx] = p[i_idx] * static_cast<T>(valid);
    }
}

template <typename T, typename Device>
void reduce_op<T, Device>::operator()(
        const Tensor& input,
        const int64_t& inner_most_dim,
        Tensor& output)
{
    const T* p = reinterpret_cast<const T*>(input.data());
    T* q = reinterpret_cast<T*>((output.data()));

    // It's just so simple to implement the reduce operation.
    for (int64_t o_idx = 0; o_idx < output.NumElements(); o_idx++) {
        T sum = 0;
        for (int64_t i_idx = o_idx * inner_most_dim; i_idx < inner_most_dim + o_idx * inner_most_dim; i_idx++) {
            sum += p[i_idx];
        }
        q[o_idx] = sum;
    }
}

template <typename T, typename Device>
void contract_op<T, Device>::operator()(
        const Tensor& in_x,
        const Tensor& in_y,
        const bool& trans_x,
        const bool& trans_y,
        const utils::BCast& bcast,
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

} // namespace op
} // namespace container