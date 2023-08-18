#include <gtest/gtest.h>

#include <ATen/core/tensor.h>
#include <ATen/kernels/lapack_op.h>
#include <ATen/kernels/einsum_op.h>
#include <ATen/kernels/test/op_test_utils.h>

namespace container {
namespace op {

template <typename T>
class EinsumOpTest : public testing::Test {
public:
    EinsumOpTest() {
        tests_utils::init_blas_handle();
        tests_utils::init_cusolver_handle();
    }
    ~EinsumOpTest() override {
        tests_utils::delete_blas_handle();
        tests_utils::delete_cusolver_handle();
    }
};

TYPED_TEST_SUITE(EinsumOpTest, tests_utils::Types);

TYPED_TEST(EinsumOpTest, Transform) {
    using Type = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using Device = typename std::tuple_element<1, decltype(TypeParam())>::type;

    const int dim = 3;
    Tensor A = std::move(Tensor({static_cast<Type>(1.0), static_cast<Type>(2.0), static_cast<Type>(3.0),
                                 static_cast<Type>(0.0), static_cast<Type>(4.0), static_cast<Type>(5.0),
                                 static_cast<Type>(0.0), static_cast<Type>(0.0), static_cast<Type>(6.0)}).to_device<Device>());
    A.reshape({-1, dim});
    Tensor expected = std::move(Tensor(
                                {static_cast<Type>(1.0), static_cast<Type>(0.0), static_cast<Type>(0.0),
                                 static_cast<Type>(2.0), static_cast<Type>(4.0), static_cast<Type>(0.0),
                                 static_cast<Type>(3.0), static_cast<Type>(5.0), static_cast<Type>(6.0)}).to_device<Device>());
    expected.reshape({-1, dim});
    // const Tensor expected = std::move(Tensor({static_cast<Type>(21.0)}).to_device<Device>());

    Tensor A_transformed = op::einsum("ij->ji", A);
    EXPECT_EQ(A_transformed, expected);
}

TYPED_TEST(EinsumOpTest, Reduce) {
    using Type = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using Device = typename std::tuple_element<1, decltype(TypeParam())>::type;

    const int dim = 3;
    Tensor A = std::move(Tensor({static_cast<Type>(1.0), static_cast<Type>(2.0), static_cast<Type>(3.0),
                                 static_cast<Type>(0.0), static_cast<Type>(4.0), static_cast<Type>(5.0),
                                 static_cast<Type>(0.0), static_cast<Type>(0.0), static_cast<Type>(6.0)}).to_device<Device>());
    A.reshape({-1, dim});
    Tensor expected_1 = std::move(Tensor(
                                {static_cast<Type>(6.0), static_cast<Type>(9.0), static_cast<Type>(6.0)}).to_device<Device>());
    Tensor expected_2 = std::move(Tensor(
                                {static_cast<Type>(1.0), static_cast<Type>(6.0), static_cast<Type>(14.0)}).to_device<Device>());
    // const Tensor expected = std::move(Tensor({static_cast<Type>(21.0)}).to_device<Device>());

    // Case 1: Normal reduction
    Tensor A_reduced = op::einsum("ij->i", A);
    EXPECT_EQ(A_reduced, expected_1);

    // Case 2: Transpose reduction
    A_reduced = op::einsum("ij->j", A);
    EXPECT_EQ(A_reduced, expected_2);

    // Case 3: All reduction
    A_reduced = op::einsum("ij->", A);
    EXPECT_EQ(A_reduced, Tensor({static_cast<Type>(21.0)}));
}

TYPED_TEST(EinsumOpTest, Stride) {
    using Type = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using Device = typename std::tuple_element<1, decltype(TypeParam())>::type;

    const int dim = 3;
    Tensor A = std::move(Tensor({static_cast<Type>(1.0), static_cast<Type>(2.0), static_cast<Type>(3.0),
                                 static_cast<Type>(0.0), static_cast<Type>(4.0), static_cast<Type>(5.0),
                                 static_cast<Type>(0.0), static_cast<Type>(0.0), static_cast<Type>(6.0)}).to_device<Device>());
    A.reshape({-1, dim});
    Tensor expected = std::move(Tensor(
                                {static_cast<Type>(1.0), static_cast<Type>(4.0), static_cast<Type>(6.0)}).to_device<Device>());
    // const Tensor expected = std::move(Tensor({static_cast<Type>(21.0)}).to_device<Device>());

    // Case 1: Normal reduction
    Tensor A_strided = op::einsum("ii->i", A);
    EXPECT_EQ(A_strided, expected);
}

TYPED_TEST(EinsumOpTest, Inflate) {
    using Type = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using Device = typename std::tuple_element<1, decltype(TypeParam())>::type;

    const int dim = 3;
    Tensor A = std::move(Tensor({static_cast<Type>(1.0), static_cast<Type>(4.0), static_cast<Type>(6.0)}).to_device<Device>());
    Tensor expected = std::move(Tensor(
                                {static_cast<Type>(1.0), static_cast<Type>(0.0), static_cast<Type>(0.0),
                                 static_cast<Type>(0.0), static_cast<Type>(4.0), static_cast<Type>(0.0),
                                 static_cast<Type>(0.0), static_cast<Type>(0.0), static_cast<Type>(6.0)}).to_device<Device>());
    expected.reshape({-1, dim});
    // const Tensor expected = std::move(Tensor({static_cast<Type>(21.0)}).to_device<Device>());

    // Case 1: Normal reduction
    Tensor A_inflated = op::einsum("i->ii", A);
    EXPECT_EQ(A_inflated, expected);
}

} // namespace op
} // namespace container
