#include <gtest/gtest.h>
#include "../blas_op.h"  // Include the code snippet
#include "module_base/module_container/tensor.h"

namespace container {
namespace op {

template <typename T>
class BlasOpTest : public testing::Test {};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(BlasOpTest, Types);

TYPED_TEST(BlasOpTest, DotCpu) {
    blas_dot<TypeParam, DEVICE_CPU> dotCalculator;

    const int n = 3;
    const Tensor x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), static_cast<TypeParam>(3.0)};
    const Tensor y = {static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0)};
    
    TypeParam result = {};
    dotCalculator(n, x.data<TypeParam>(), 1, y.data<TypeParam>(), 1, &result);
    const TypeParam expected = static_cast<TypeParam>(32.0);

    EXPECT_EQ(result, expected);
}

TYPED_TEST(BlasOpTest, ScalCpu) {
    blas_scal<TypeParam, DEVICE_CPU> scalCalculator;

    const int n = 3;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    Tensor x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), static_cast<TypeParam>(3.0)};
    
    scalCalculator(n, &alpha, x.data<TypeParam>(), 1);
    const Tensor expected = {static_cast<TypeParam>(2.0), static_cast<TypeParam>(4.0), static_cast<TypeParam>(6.0)};

    EXPECT_EQ(x, expected);
}

TYPED_TEST(BlasOpTest, AxpyCpu) {
    blas_axpy<TypeParam, DEVICE_CPU> axpyCalculator;

    const int n = 3;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    const Tensor x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), static_cast<TypeParam>(3.0)};
    Tensor y = {static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0)};
    
    axpyCalculator(n, &alpha, x.data<TypeParam>(), 1, y.data<TypeParam>(), 1);
    const Tensor expected = {static_cast<TypeParam>(6.0), static_cast<TypeParam>(9.0), static_cast<TypeParam>(12.0)};

    EXPECT_EQ(y, expected);
}

TYPED_TEST(BlasOpTest, GemvCpu) {
    blas_gemv<TypeParam, DEVICE_CPU> gemvCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    const TypeParam beta = static_cast<TypeParam>(3.0);
    const Tensor A = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), static_cast<TypeParam>(3.0), 
                      static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0)};
    const Tensor x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0)};
    Tensor y = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), static_cast<TypeParam>(3.0)};
    
    gemvCalculator(trans, m, n, &alpha, A.data<TypeParam>(), m, x.data<TypeParam>(), 1, &beta, y.data<TypeParam>(), 1);
    const Tensor expected = {static_cast<TypeParam>(21.0), static_cast<TypeParam>(30.0), static_cast<TypeParam>(39.0)};

    EXPECT_EQ(y, expected);
}

TYPED_TEST(BlasOpTest, GemvBatchedCpu) {
    blas_gemv<TypeParam, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched<TypeParam, DEVICE_CPU> gemvBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    const TypeParam beta = static_cast<TypeParam>(3.0);

    std::vector<TypeParam*> A = {};
    std::vector<TypeParam*> x = {};
    std::vector<TypeParam*> y = {};

    const Tensor _A = {
        static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), 
        static_cast<TypeParam>(3.0), static_cast<TypeParam>(4.0), 
        static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
        
        static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0),
        static_cast<TypeParam>(9.0), static_cast<TypeParam>(10.0),
        static_cast<TypeParam>(11.0), static_cast<TypeParam>(12.0)};
    
    A.push_back(_A.data<TypeParam>());
    A.push_back(_A.data<TypeParam>() + m * n);

    const Tensor _x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0)};
    x.push_back(_x.data<TypeParam>());
    x.push_back(_x.data<TypeParam>());

    Tensor _y1 = {static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
                 static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0), static_cast<TypeParam>(9.0)};
    Tensor _y2 = _y1;
    y.push_back(_y1.data<TypeParam>());
    y.push_back(_y1.data<TypeParam>() + m);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y.data(), 1, batch_size);

    for (int ii = 0; ii < batch_size; ++ii) {
        gemvCalculator(trans, m, n, &alpha, A[ii], m, x[ii], 1, &beta, _y2.data<TypeParam>() + ii * m, 1);
    }

    EXPECT_EQ(_y1, _y2);
}

TYPED_TEST(BlasOpTest, GemvBatchedStridedCpu) {
    blas_gemv<TypeParam, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched_strided<TypeParam, DEVICE_CPU> gemvBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    const TypeParam beta = static_cast<TypeParam>(3.0);

    std::vector<TypeParam*> A = {};
    std::vector<TypeParam*> x = {};
    std::vector<TypeParam*> y = {};

    const Tensor _A = {
        static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), 
        static_cast<TypeParam>(3.0), static_cast<TypeParam>(4.0), 
        static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
        
        static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0),
        static_cast<TypeParam>(9.0), static_cast<TypeParam>(10.0),
        static_cast<TypeParam>(11.0), static_cast<TypeParam>(12.0)};
    
    A.push_back(_A.data<TypeParam>());
    A.push_back(_A.data<TypeParam>() + m * n);

    const Tensor _x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0)};
    x.push_back(_x.data<TypeParam>());
    x.push_back(_x.data<TypeParam>());

    Tensor _y1 = {static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
                 static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0), static_cast<TypeParam>(9.0)};
    Tensor _y2 = _y1;
    y.push_back(_y1.data<TypeParam>());
    y.push_back(_y1.data<TypeParam>() + m);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y[0], 1, m, batch_size);

    for (int ii = 0; ii < batch_size; ++ii) {
        gemvCalculator(trans, m, n, &alpha, A[ii], m, x[ii], 1, &beta, _y2.data<TypeParam>() + ii * m, 1);
    }
    EXPECT_EQ(_y1, _y2);
}


TYPED_TEST(BlasOpTest, GemmCpu) {
    blas_gemm<TypeParam, DEVICE_CPU> gemmCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    const TypeParam beta = static_cast<TypeParam>(3.0);
    const Tensor A = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), static_cast<TypeParam>(3.0), 
                      static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0)};
    const Tensor x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0)};
    Tensor y = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), static_cast<TypeParam>(3.0)};
    
    gemmCalculator(trans, trans, m, 1, n, &alpha, A.data<TypeParam>(), m, x.data<TypeParam>(), n, &beta, y.data<TypeParam>(), m);
    const Tensor expected = {static_cast<TypeParam>(21.0), static_cast<TypeParam>(30.0), static_cast<TypeParam>(39.0)};

    EXPECT_EQ(y, expected);
}

TYPED_TEST(BlasOpTest, GemmBatchedCpu) {
    blas_gemv_batched<TypeParam, DEVICE_CPU> gemvBatchedCalculator;
    blas_gemm_batched<TypeParam, DEVICE_CPU> gemmBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    const TypeParam beta = static_cast<TypeParam>(3.0);

    std::vector<TypeParam*> A = {};
    std::vector<TypeParam*> x = {};
    std::vector<TypeParam*> y1 = {};
    std::vector<TypeParam*> y2 = {};

    const Tensor _A = {
        static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), 
        static_cast<TypeParam>(3.0), static_cast<TypeParam>(4.0), 
        static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
        
        static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0),
        static_cast<TypeParam>(9.0), static_cast<TypeParam>(10.0),
        static_cast<TypeParam>(11.0), static_cast<TypeParam>(12.0)};
    
    A.push_back(_A.data<TypeParam>());
    A.push_back(_A.data<TypeParam>() + m * n);

    const Tensor _x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0)};
    x.push_back(_x.data<TypeParam>());
    x.push_back(_x.data<TypeParam>());

    Tensor _y1 = {static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
                 static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0), static_cast<TypeParam>(9.0)};
    Tensor _y2 = _y1;
    y1.push_back(_y1.data<TypeParam>());
    y1.push_back(_y1.data<TypeParam>() + m);
    y2.push_back(_y2.data<TypeParam>());
    y2.push_back(_y2.data<TypeParam>() + m);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y1.data(), 1, batch_size);
    gemmBatchedCalculator(trans, trans, m, 1, n, &alpha, A.data(), m, x.data(), n, &beta, y2.data(), m, batch_size);

    EXPECT_EQ(_y1, _y2);
}

TYPED_TEST(BlasOpTest, GemmBatchedStridedCpu) {
    blas_gemv_batched_strided<TypeParam, DEVICE_CPU> gemvBatchedStridedCalculator;
    blas_gemm_batched_strided<TypeParam, DEVICE_CPU> gemmBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const TypeParam alpha = static_cast<TypeParam>(2.0);
    const TypeParam beta = static_cast<TypeParam>(3.0);

    std::vector<TypeParam*> A = {};
    std::vector<TypeParam*> x = {};
    std::vector<TypeParam*> y1 = {};
    std::vector<TypeParam*> y2 = {};

    const Tensor _A = {
        static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0), 
        static_cast<TypeParam>(3.0), static_cast<TypeParam>(4.0), 
        static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
        
        static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0),
        static_cast<TypeParam>(9.0), static_cast<TypeParam>(10.0),
        static_cast<TypeParam>(11.0), static_cast<TypeParam>(12.0)};
    
    A.push_back(_A.data<TypeParam>());
    A.push_back(_A.data<TypeParam>() + m * n);

    const Tensor _x = {static_cast<TypeParam>(1.0), static_cast<TypeParam>(2.0)};
    x.push_back(_x.data<TypeParam>());
    x.push_back(_x.data<TypeParam>());

    Tensor _y1 = {static_cast<TypeParam>(4.0), static_cast<TypeParam>(5.0), static_cast<TypeParam>(6.0),
                 static_cast<TypeParam>(7.0), static_cast<TypeParam>(8.0), static_cast<TypeParam>(9.0)};
    Tensor _y2 = _y1;
    y1.push_back(_y1.data<TypeParam>());
    y1.push_back(_y1.data<TypeParam>() + m);
    y2.push_back(_y2.data<TypeParam>());
    y2.push_back(_y2.data<TypeParam>() + m);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y1[0], 1, m, batch_size);
    gemmBatchedStridedCalculator(trans, trans, m, 1, n, &alpha, A[0], m, m * n, x[0], n, 0, &beta, y2[0], m, m, batch_size);

    EXPECT_EQ(_y1, _y2);
}

} // namespace op
} // namespace container

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
