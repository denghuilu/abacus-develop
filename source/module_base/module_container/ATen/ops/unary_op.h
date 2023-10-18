#ifndef ATEN_OPS_UNARY_OP_H_
#define ATEN_OPS_UNARY_OP_H_

#include <ATen/core/tensor.h>
#include <base/macros/macros.h>

namespace container {
namespace op {

template <typename Device, typename Func>
Tensor unary_op(const Tensor& self, Func op) {
    return kernels::apply<Device, Func>()(self, op);
}

Tensor sin(const Tensor& self);
Tensor cos(const Tensor& self);
Tensor conj(const Tensor& self);

Tensor max(const Tensor& self);

Tensor operator-(const Tensor& self);

} // namespace op
} // namespace container

#endif // ATEN_OPS_UNARY_OP_H_