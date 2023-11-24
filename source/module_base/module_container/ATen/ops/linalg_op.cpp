#include <ATen/ops/linalg_op.h>

#include <ATen/kernels/linalg.h>
#include <base/macros/macros.h>

#include <complex>

namespace container {
namespace op {

static bool check_shape(const TensorShape& x, const TensorShape& y) {
    return (x == y || y == TensorShape({1}));
}

static bool check_data_type(const DataType& x, const DataType& y) {
    return (x == y || 
        x == DataType::DT_COMPLEX_DOUBLE && y == DataType::DT_DOUBLE ||
        x == DataType::DT_COMPLEX && y == DataType::DT_FLOAT);
}

void add_op::operator()(const Tensor &x, const Tensor &y, Tensor &z) {
    // check the shape
    REQUIRES_OK(x.shape() == y.shape() && x.shape() == z.shape(),
                "add: the shape of the two input Tensors must be the same")
    REQUIRES_OK(x.data_type() == y.data_type() && x.data_type() == z.data_type(),
                "add: the data type of the two input Tensors must be the same")
    REQUIRES_OK(x.device_type() == y.device_type() && x.device_type() == z.device_type(),
                "add: the device type of the two input Tensors must be the same")
    // allocate memory for the result
    TEMPLATE_ALL_LAMBDA_2(x.data_type(), x.device_type(), [&](){
        T_ alpha = static_cast<T_>(1);
        T_ beta  = static_cast<T_>(1);
        kernels::add<T_, T_, DEVICE_>()(
            x.NumElements(), alpha, x.data<T_>(), beta, y.data<T_>(), z.data<T_>());
    })
}

template<typename T>
void add_op::operator()(const T& alpha, const Tensor &x, const T& beta, const Tensor &y, Tensor &z) {
    // check the shape
    REQUIRES_OK(x.shape() == y.shape() && x.shape() == z.shape(),
                "add: the shape of the two input Tensors must be the same")
    REQUIRES_OK(check_data_type(x.data_type(), y.data_type()) && x.data_type() == z.data_type(),
                "add: the data type of the two input Tensors must be the same")
    REQUIRES_OK(x.device_type() == y.device_type() && x.device_type() == z.device_type() ,
                "add: the device type of the two input Tensors must be the same")
    // allocate memory for the result
    TEMPLATE_ALL_LAMBDA_2(x.data_type(), x.device_type(), [&](){
        auto x_ = x;
        auto y_ = y;
        kernels::add<T, T, DEVICE_>()(
            x_.NumElements(), alpha, x_.data<T>(), beta, y_.data<T>(), z.data<T>());
    })
}

void mul_op::operator()(const container::Tensor &x, const container::Tensor &y, container::Tensor &z) {
    // check the shape
    REQUIRES_OK(check_shape(x.shape(), y.shape()) && x.shape() == z.shape(),
                "mul: the shape of the two input Tensors must be the same")
    REQUIRES_OK(check_data_type(x.data_type(), y.data_type()) && x.data_type() == z.data_type(),
                "mul: the data type of the two input Tensors must be the same")
    REQUIRES_OK(x.device_type() == y.device_type() || (y.shape().NumElements() == 1 && y.device_type() == DeviceType::CpuDevice),
                "mul: the device type of the two input Tensors must be the same")
    // allocate memory for the result
    TEMPLATE_ALL_LAMBDA_3(x.data_type(), y.data_type(), x.device_type(), [&](){
        T_1 alpha = static_cast<T_1>(1);
        kernels::mul<T_1, T_2, DEVICE_>()(
            x.NumElements(), alpha, x.data<T_1>(), y.data<T_2>(), z.data<T_1>());
    })
}

template<typename T>
void mul_op::operator()(const T& alpha, const container::Tensor &x, container::Tensor &y) {
    // check the shape
    REQUIRES_OK(x.shape() == y.shape(),
                "mul: the shape of the two input Tensors must be the same")
    REQUIRES_OK(x.data_type() == y.data_type(),
                "mul: the data type of the two input Tensors must be the same")
    REQUIRES_OK(x.device_type() == y.device_type(),
                "mul: the device type of the two input Tensors must be the same")
    // allocate memory for the result
    TEMPLATE_ALL_LAMBDA_2(y.data_type(), y.device_type(), [&](){
        kernels::mul<T, T, DEVICE_>()(
            x.NumElements(), alpha, x.data<T>(), y.data<T>());
    })
}

// z = x - y
void div_op::operator()(const container::Tensor &x, const container::Tensor &y, container::Tensor &z) {
    // check the shape
    REQUIRES_OK(check_shape(x.shape(), y.shape()) && x.shape() == z.shape(),
                "div: the shape of the two input Tensors must be the same")
    REQUIRES_OK(check_data_type(x.data_type(), y.data_type()) && x.data_type() == z.data_type(),
                "div: the data type of the two input Tensors must be the same")
    REQUIRES_OK(x.device_type() == y.device_type() || (y.shape().NumElements() == 1 && y.device_type() == DeviceType::CpuDevice),
                "div: the device type of the two input Tensors must be the same")
    // allocate memory for the result
    if (y.NumElements() == 1) {
        TEMPLATE_ALL_LAMBDA_3(x.data_type(), y.data_type(), x.device_type(), [&](){
            T_1 alpha = static_cast<T_1>(1);
            auto x_ = x;
            kernels::div<T_1, T_2, DEVICE_>()(
                x.NumElements(), alpha, x_.data<T_1>(), y.data<T_2>()[0], z.data<T_1>());
        })
    }
    else {
        TEMPLATE_ALL_LAMBDA_3(x.data_type(), y.data_type(), x.device_type(), [&](){
            T_1 alpha = static_cast<T_1>(1);
            kernels::div<T_1, T_2, DEVICE_>()(
                x.NumElements(), alpha, x.data<T_1>(), y.data<T_2>(), z.data<T_1>());
        })
    }
}


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

template void add_op::operator()<float>(const float&, const container::Tensor&, const float&, const container::Tensor&, container::Tensor&);
template void add_op::operator()<double>(const double&, const container::Tensor&, const double&, const container::Tensor&, container::Tensor&);
template void add_op::operator()<std::complex<float> >(const std::complex<float>&, const container::Tensor&, const std::complex<float>&, const container::Tensor&, container::Tensor&);
template void add_op::operator()<std::complex<double>>(const std::complex<double>&, const container::Tensor&, const std::complex<double>&, const container::Tensor&, container::Tensor&);

template void mul_op::operator()<float>(const float&, const container::Tensor&, container::Tensor&);
template void mul_op::operator()<double>(const double&, const container::Tensor&, container::Tensor&);
template void mul_op::operator()<std::complex<float> >(const std::complex<float>&, const container::Tensor&, container::Tensor&);
template void mul_op::operator()<std::complex<double>>(const std::complex<double>&, const container::Tensor&, container::Tensor&);

} // namespace kernels
} // namespace container

ct::Tensor operator+(const ct::Tensor& self, const ct::Tensor& other) {
    // check the shape
    REQUIRES_OK(self.shape() == other.shape(),
                "add: the shape of the two input Tensors must be the same")
    // allocate memory for the result
    ct::Tensor result = ct::Tensor(self.data_type(), self.device_type(), self.shape());
    ct::op::add_op()(self, other, result);
    return result;
}

ct::Tensor operator-(const ct::Tensor& self, const ct::Tensor& other) {
    // check the shape
    REQUIRES_OK(self.shape() == other.shape(),
                "add: the shape of the two input Tensors must be the same")
    REQUIRES_OK(self.data_type() == other.data_type(),
                "add: the data type of the two input Tensors must be the same")
    REQUIRES_OK(self.device_type() == other.device_type(),
                "add: the device type of the two input Tensors must be the same")
    ct::Tensor result = ct::Tensor(self.data_type(), self.device_type(), self.shape());
    // allocate memory for the result
    TEMPLATE_ALL_LAMBDA_2(self.data_type(), self.device_type(), [&](){
        T_ alpha = static_cast<T_>(1.0);
        T_ beta  = static_cast<T_>(-1.0);
        ct::kernels::add<T_, T_, DEVICE_>()(
            self.NumElements(), alpha, self.data<T_>(), beta, other.data<T_>(), result.data<T_>());
    })
    return result;
}

ct::Tensor operator*(const ct::Tensor& self, const ct::Tensor& other) {
    // check the shape
    REQUIRES_OK(self.shape() == other.shape(),
                "mul: the shape of the two input Tensors must be the same")
    // allocate memory for the result
    ct::Tensor result = ct::Tensor(self.data_type(), self.device_type(), self.shape());
    ct::op::mul_op()(self, other, result);
    return result;
}

ct::Tensor operator/(const ct::Tensor& self, const ct::Tensor& other) {
    // allocate memory for the result
    ct::Tensor result = ct::Tensor(self.data_type(), self.device_type(), self.shape());
    ct::op::div_op()(self, other, result);
    return result;
}

ct::Tensor& operator+=(ct::Tensor& self, const ct::Tensor& other) {
    // check the shape
    REQUIRES_OK(self.shape() == other.shape(),
                "add: the shape of the two input Tensors must be the same")
    ct::op::add_op()(self, other, self);
    return self;
}

ct::Tensor& operator-=(ct::Tensor& self, const ct::Tensor& other) {
    // check the shape
    REQUIRES_OK(self.shape() == other.shape(),
                "add: the shape of the two input Tensors must be the same")
    REQUIRES_OK(self.data_type() == other.data_type(),
                "add: the data type of the two input Tensors must be the same")
    REQUIRES_OK(self.device_type() == other.device_type(),
                "add: the device type of the two input Tensors must be the same")
    // allocate memory for the result
    TEMPLATE_ALL_LAMBDA_2(self.data_type(), self.device_type(), [&](){
        T_ alpha = static_cast<T_>(1.0);
        T_ beta  = static_cast<T_>(-1.0);
        ct::kernels::add<T_, T_, DEVICE_>()(
            self.NumElements(), alpha, self.data<T_>(), beta, other.data<T_>(), self.data<T_>());
    })

    return self;
}

ct::Tensor& operator*=(ct::Tensor& self, const ct::Tensor& other) {
    // check the shape
    REQUIRES_OK(self.shape() == other.shape(),
                "mul: the shape of the two input Tensors must be the same")
    ct::op::mul_op()(self, other, self);
    return self;
}

ct::Tensor& operator/=(ct::Tensor& self, const ct::Tensor& other) {
    // check the shape
    REQUIRES_OK(self.shape() == other.shape(),
                "div: the shape of the two input Tensors must be the same")
    ct::op::div_op()(self, other, self);
    return self;
}