#include <ATen/ops/linalg_op.h>

#include <ATen/kernels/linalg.h>
#include <base/macros/macros.h>

namespace container {
namespace op {

template<bool Conjugate>
void transpose_op<Conjugate>::operator()(
    const Tensor& input,
    const std::vector<int>& perm,
    Tensor& output)
{
    TEMPLATE_ALL_2(input.data_type(), input.device_type(),
        kernels::transpose<T_, DEVICE_, Conjugate>()(
            perm, input.shape().dims(), output.shape().dims(), input.data<T_>(), output.data<T_>()))
}


void stride_op::operator()(
        const Tensor& input,
        const std::vector<int64_t>& stride,
        Tensor& output)
{
    TEMPLATE_ALL_2(input.data_type(), input.device_type(),
        kernels::stride<T_, DEVICE_>()(
            stride, input.shape().dims(), output.shape().dims(), input.data<T_>(), output.data<T_>()))
}


void inflate_op::operator()(
        const Tensor& input,
        const std::vector<int64_t>& inflate,
        Tensor& output)
{
    TEMPLATE_ALL_2(input.data_type(), input.device_type(),
        kernels::inflate<T_, DEVICE_>()(
            inflate, input.shape().dims(), output.shape().dims(), input.data<T_>(), output.data<T_>()))
}


void reduce_op::operator()(
    const Tensor &input,
    const int64_t &inner_most_dim,
    Tensor &output)
{
    TEMPLATE_ALL_2(input.data_type(), input.device_type(),
        kernels::reduce<T_, DEVICE_>()(
            output.NumElements(), inner_most_dim, input.data<T_>(), output.data<T_>()))
}

template struct transpose_op<false>;

} // namespace kernels
} // namespace container