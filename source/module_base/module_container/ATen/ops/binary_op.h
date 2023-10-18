#ifndef ATEN_OPS_BINARY_OP_H_
#define ATEN_OPS_BINARY_OP_H_

#include <ATen/core/tensor.h>
#include <base/macros/macros.h>

namespace container {
namespace op {

template <typename Device, typename Func>
Tensor binary_op(const Tensor& self, const Tensor& other, Func op) {
    return kernels::apply<Device, Func>()(self, other, op);
}

template <typename Device, typename Func>
Tensor binary_inplace_op(const Tensor& self, const Tensor& other, Tensor& out, Func op) {
    return kernels::apply<Device, Func>()(self, other, out, op);
}

Tensor operator+(const Tensor& self, const Tensor& other);
Tensor operator-(const Tensor& self, const Tensor& other);
Tensor operator*(const Tensor& self, const Tensor& other);
Tensor operator/(const Tensor& self, const Tensor& other);

Tensor operator+=(const Tensor& self, const Tensor& other);
Tensor operator-=(const Tensor& self, const Tensor& other);
Tensor operator*=(const Tensor& self, const Tensor& other);
Tensor operator/=(const Tensor& self, const Tensor& other);

} // namespace op
} // namespace container

#endif // ATEN_OPS_BINARY_OP_H_