#include <ATen/ops/unary_op.h>
#include <base/func/unary_functor.h>

namespace container {
namespace op {

Tensor operator-(const Tensor& self) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return unary_op<DEVICE_>(self, ::base::func::negnate_func<T_>());
    })
}

Tensor sin(const Tensor& self) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return unary_op<DEVICE_>(self, ::base::func::sin_func<T_>());
    })
}

Tensor cos(const Tensor& self) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return unary_op<DEVICE_>(self, ::base::func::cos_func<T_>());
    })
}

Tensor conj(const Tensor& self) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return unary_op<DEVICE_>(self, ::base::func::conj_func<T_>());
    })
}

} // namespace op
} // namespace container