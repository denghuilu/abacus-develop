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

TYPED_TEST(EinsumOpTest, Trtri) {
    using Type = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using Device = typename std::tuple_element<1, decltype(TypeParam())>::type;

    Tensor empty_1 = {};

    Tensor empty_2 = empty_1;

    const int dim = 3;
    Tensor A = std::move(Tensor({static_cast<Type>(1.0), static_cast<Type>(2.0), static_cast<Type>(3.0),
                                 static_cast<Type>(0.0), static_cast<Type>(4.0), static_cast<Type>(5.0),
                                 static_cast<Type>(0.0), static_cast<Type>(0.0), static_cast<Type>(6.0)}).to_device<Device>());
    A.reshape({-1, dim});
    Tensor B = std::move(Tensor({static_cast<Type>(1.0), static_cast<Type>(0.0), static_cast<Type>(0.0),
                                 static_cast<Type>(0.0), static_cast<Type>(1.0), static_cast<Type>(0.0),
                                 static_cast<Type>(0.0), static_cast<Type>(0.0), static_cast<Type>(1.0)}).to_device<Device>());
    B.reshape({-1, dim});
    Tensor C = op::einsum("ij->i", A);
    
    // std::cerr << "A = \t" << A << std::endl;
    // std::cerr << "B = \t" << B << std::endl;
    // std::cerr << "C = \t" << C << std::endl;

    // Tensor B = A;
    // Tensor C = B;
    // C.zero();
    
    // const char trans = 'N';
    // const int m = 3;
    // const int n = 3;
    // const int k = 3;
    // const Type alpha = static_cast<Type>(1.0);
    // const Type beta  = static_cast<Type>(0.0);
    // Note all blas and lapack operators within container are column major!
    // For this reason, we should employ 'L' instead of 'U' in the subsequent line.
    // gemmCalculator(trans, trans, m, n, k, &alpha, A.data<Type>(), k, I.data<Type>(), n, &beta, C.data<Type>(), n);
    
    // EXPECT_EQ(C, I);
}

} // namespace op
} // namespace container
