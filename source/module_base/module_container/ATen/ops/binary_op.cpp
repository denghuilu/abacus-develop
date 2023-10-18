#include <ATen/ops/binary_op.h>
#include <base/func/binary_functor.h>

namespace container {
namespace op {

Tensor operator+(const Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_op<DEVICE_>(self, other, ::base::func::add_func<T_>());
    })
}

Tensor operator-(const Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_op<DEVICE_>(self, other, ::base::func::sub_func<T_>());
    })
}

Tensor operator*(const Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_op<DEVICE_>(self, other, ::base::func::mul_func<T_, DEVICE_>());
    })
}

Tensor operator/(const Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_op<DEVICE_>(self, other, ::base::func::div_func<T_>());
    })
}

Tensor operator+=(Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_inplace_op<DEVICE_>(self, other, self, ::base::func::add_func<T_>());
    })
}

Tensor operator-=(Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_inplace_op<DEVICE_>(self, other, self, ::base::func::sub_func<T_>());
    })
}

Tensor operator*=(Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_inplace_op<DEVICE_>(self, other, self, ::base::func::mul_func<T_, DEVICE_>());
    })
}

Tensor operator/=(Tensor& self, const Tensor& other) {
    TEMPLATE_ALL_2(
        self.data_type(), self.device_type(), {
        return binary_inplace_op<DEVICE_>(self, other, self, ::base::func::div_func<T_>());
    })
}

} // namespace op
} // namespace container