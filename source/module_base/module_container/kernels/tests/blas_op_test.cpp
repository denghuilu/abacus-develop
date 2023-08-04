#include <gtest/gtest.h>
#include "../blas_op.h"  // Include the code snippet

namespace container {
namespace op {

// Dot test cases
TEST(BlasOpTest, DotProductFloat) {
    blas_dot<float, DEVICE_CPU> dotCalculator;

    const int n = 3;
    float x[] = {1.0, 2.0, 3.0};
    float y[] = {4.0, 5.0, 6.0};
    float result;

    dotCalculator(n, x, 1, y, 1, &result);

    float expected = 32.0;

    EXPECT_DOUBLE_EQ(result, expected);
}

TEST(BlasOpTest, DotProductDouble) {
    blas_dot<double, DEVICE_CPU> dotCalculator;

    const int n = 3;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    double result;

    dotCalculator(n, x, 1, y, 1, &result);

    double expected = 32.0;

    EXPECT_DOUBLE_EQ(result, expected);
}

TEST(BlasOpTest, DotProductComplex) {
    blas_dot<std::complex<float>, DEVICE_CPU> dotCalculator;

    const int n = 3;
    std::complex<float> x[] = {std::complex<float>(1.0, 1.0), std::complex<float>(2.0, 2.0), std::complex<float>(3.0, 3.0)};
    std::complex<float> y[] = {std::complex<float>(4.0, 4.0), std::complex<float>(5.0, 5.0), std::complex<float>(6.0, 6.0)};
    std::complex<float> result;

    dotCalculator(n, x, 1, y, 1, &result);

    std::complex<float> expected = std::complex<float>(64.0, 0.0);

    EXPECT_DOUBLE_EQ(std::norm(result - expected), 0);
}

TEST(BlasOpTest, DotProductComplexDouble) {
    blas_dot<std::complex<double>, DEVICE_CPU> dotCalculator;

    const int n = 3;
    std::complex<double> x[] = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0), std::complex<double>(3.0, 3.0)};
    std::complex<double> y[] = {std::complex<double>(4.0, 4.0), std::complex<double>(5.0, 5.0), std::complex<double>(6.0, 6.0)};
    std::complex<double> result;

    dotCalculator(n, x, 1, y, 1, &result);

    std::complex<double> expected = std::complex<double>(64.0, 0.0);

    EXPECT_DOUBLE_EQ(std::norm(result - expected), 0);
}

// Scal test cases
TEST(BlasOpTest, ScalFloat) {
    blas_scal<float, DEVICE_CPU> scalCalculator;

    const int n = 3;
    float alpha = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f};
    float expected[] = {2.0f, 4.0f, 6.0f};

    scalCalculator(n, &alpha, x, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(x[i], expected[i]);
    }
}

TEST(BlasOpTest, ScalDouble) {
    blas_scal<double, DEVICE_CPU> scalCalculator;

    const int n = 3;
    double alpha = 2.0;
    double x[] = {1.0, 2.0, 3.0};
    double expected[] = {2.0, 4.0, 6.0};

    scalCalculator(n, &alpha, x, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(x[i], expected[i]);
    }
}

TEST(BlasOpTest, ScalComplex) {
    blas_scal<std::complex<float>, DEVICE_CPU> scalCalculator;

    const int n = 3;
    std::complex<float> alpha(2.0f, 0.0f);
    std::complex<float> x[] = {std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, 2.0f), std::complex<float>(3.0f, 3.0f)};
    std::complex<float> expected[] = {std::complex<float>(2.0f, 2.0f), std::complex<float>(4.0f, 4.0f), std::complex<float>(6.0f, 6.0f)};

    scalCalculator(n, &alpha, x, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(x[i].real(), expected[i].real());
        EXPECT_FLOAT_EQ(x[i].imag(), expected[i].imag());
    }
}

TEST(BlasOpTest, ScalComplexDouble) {
    blas_scal<std::complex<double>, DEVICE_CPU> scalCalculator;

    const int n = 3;
    std::complex<double> alpha(2.0, 0.0);
    std::complex<double> x[] = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0), std::complex<double>(3.0, 3.0)};
    std::complex<double> expected[] = {std::complex<double>(2.0, 2.0), std::complex<double>(4.0, 4.0), std::complex<double>(6.0, 6.0)};

    scalCalculator(n, &alpha, x, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(x[i].real(), expected[i].real());
        EXPECT_DOUBLE_EQ(x[i].imag(), expected[i].imag());
    }
}


// axpy test cases
TEST(BlasOpTest, AxpyFloat) {
    blas_axpy<float, DEVICE_CPU> axpyCalculator;

    const int n = 3;
    float alpha = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    float expected[] = {6.0f, 9.0f, 12.0f};

    axpyCalculator(n, &alpha, x, 1, y, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }
}

TEST(BlasOpTest, AxpyDouble) {
    blas_axpy<double, DEVICE_CPU> axpyCalculator;

    const int n = 3;
    double alpha = 2.0;
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    double expected[] = {6.0, 9.0, 12.0};

    axpyCalculator(n, &alpha, x, 1, y, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(y[i], expected[i]);
    }
}

TEST(BlasOpTest, AxpyComplex) {
    blas_axpy<std::complex<float>, DEVICE_CPU> axpyCalculator;

    const int n = 3;
    std::complex<float> alpha(2.0f, 0.0f);
    std::complex<float> x[] = {std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, 2.0f), std::complex<float>(3.0f, 3.0f)};
    std::complex<float> y[] = {std::complex<float>(4.0f, 4.0f), std::complex<float>(5.0f, 5.0f), std::complex<float>(6.0f, 6.0f)};
    std::complex<float> expected[] = {std::complex<float>(6.0f, 6.0f), std::complex<float>(9.0f, 9.0f), std::complex<float>(12.0f, 12.0f)};

    axpyCalculator(n, &alpha, x, 1, y, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y[i].real(), expected[i].real());
        EXPECT_FLOAT_EQ(y[i].imag(), expected[i].imag());
    }
}

TEST(BlasOpTest, AxpyComplexDouble) {
    blas_axpy<std::complex<double>, DEVICE_CPU> axpyCalculator;

    const int n = 3;
    std::complex<double> alpha(2.0, 0.0);
    std::complex<double> x[] = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0), std::complex<double>(3.0, 3.0)};
    std::complex<double> y[] = {std::complex<double>(4.0, 4.0), std::complex<double>(5.0, 5.0), std::complex<double>(6.0, 6.0)};
    std::complex<double> expected[] = {std::complex<double>(6.0, 6.0), std::complex<double>(9.0, 9.0), std::complex<double>(12.0, 12.0)};

    axpyCalculator(n, &alpha, x, 1, y, 1);

    for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(y[i].real(), expected[i].real());
        EXPECT_DOUBLE_EQ(y[i].imag(), expected[i].imag());
    }
}

// Gemv test cases
TEST(BlasOpTest, GemvFloat) {
    blas_gemv<float, DEVICE_CPU> gemvCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const float alpha = 2.0f;
    const float beta = 3.0f;
    const float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float x[] = {1.0f, 2.0f};
    float y[] = {1.0f, 2.0f, 3.0f};
    float expected[] = {21.0f, 30.0f, 39.0f};

    gemvCalculator(trans, m, n, &alpha, A, m, x, 1, &beta, y, 1);

    for (int i = 0; i < m; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }
}

TEST(BlasOpTest, GemvDouble) {
    blas_gemv<double, DEVICE_CPU> gemvCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const double alpha = 2.0;
    const double beta = 3.0;
    const double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double x[] = {1.0, 2.0};
    double y[] = {1.0, 2.0, 3.0};
    double expected[] = {21.0, 30.0, 39.0};

    gemvCalculator(trans, m, n, &alpha, A, m, x, 1, &beta, y, 1);

    for (int i = 0; i < m; ++i) {
        EXPECT_DOUBLE_EQ(y[i], expected[i]);
    }
}

TEST(BlasOpTest, GemvComplex) {
    blas_dot<std::complex<float>, DEVICE_CPU> dotCalculator;
    blas_gemv<std::complex<float>, DEVICE_CPU> gemvCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const std::complex<float> alpha(1.0f, 0.0f);
    const std::complex<float> beta(0.0f, 0.0f);
    const std::complex<float> A[] = {
        std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, 2.0f),
        std::complex<float>(3.0f, 3.0f), std::complex<float>(4.0f, 4.0f),
        std::complex<float>(5.0f, 5.0f), std::complex<float>(6.0f, 6.0f)
    };
    const std::complex<float> x[] = {std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, 2.0f)};
    std::complex<float> y[] = {std::complex<float>(4.0f, 4.0f), std::complex<float>(5.0f, 5.0f), std::complex<float>(6.0f, 6.0f)};

    gemvCalculator(trans, m, n, &alpha, A, m, x, 1, &beta, y, 1);

    for (int i = 0; i < m; ++i) {
        std::complex<float> result = {0.0f, 0.0f};
        dotCalculator(n, A + i, m, x, 1, &result);
        EXPECT_FLOAT_EQ(y[i].real(), result.imag());
        EXPECT_FLOAT_EQ(y[i].imag(), result.real());
    }
}

TEST(BlasOpTest, GemvComplexDouble) {
    blas_dot<std::complex<double>, DEVICE_CPU> dotCalculator;
    blas_gemv<std::complex<double>, DEVICE_CPU> gemvCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);
    const std::complex<double> A[] = {
        std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0),
        std::complex<double>(3.0, 3.0), std::complex<double>(4.0, 4.0),
        std::complex<double>(5.0, 5.0), std::complex<double>(6.0, 6.0)
    };
    const std::complex<double> x[] = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0)};
    std::complex<double> y[] = {std::complex<double>(4.0, 4.0), std::complex<double>(5.0, 5.0), std::complex<double>(6.0, 6.0)};

    gemvCalculator(trans, m, n, &alpha, A, m, x, 1, &beta, y, 1);

    for (int i = 0; i < m; ++i) {
        std::complex<double> result = {0.0, 0.0};
        dotCalculator(n, A + i, m, x, 1, &result);
        EXPECT_DOUBLE_EQ(y[i].real(), result.imag());
        EXPECT_DOUBLE_EQ(y[i].imag(), result.real());
    }
}

// Gemv batched test cases
TEST(BlasOpTest, GemvBatchedFloat) {
    blas_gemv<float, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched<float, DEVICE_CPU> gemvBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const float alpha = 2.0f;
    const float beta = 3.0f;
    
    std::vector<float*> A = {};
    std::vector<float*> x = {};
    std::vector<float*> y = {};
    float _A[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    float _x[] = {1.0f, 2.0f};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    float _y1[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float _y2[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y.data(), 1, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        gemvCalculator(trans, m, n, &alpha, A[i], m, x[i], 1, &beta, _y2 + i * m, 1);
    }
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemvBatchedDouble) {
    blas_gemv<double, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched<double, DEVICE_CPU> gemvBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const double alpha = 2.0;
    const double beta = 3.0;
    
    std::vector<double*> A = {};
    std::vector<double*> x = {};
    std::vector<double*> y = {};
    double _A[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,

        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    double _x[] = {1.0, 2.0};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    double _y1[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double _y2[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y.data(), 1, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        gemvCalculator(trans, m, n, &alpha, A[i], m, x[i], 1, &beta, _y2 + i * m, 1);
    }
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemvBatchedComplex) {
    blas_gemv<std::complex<float>, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched<std::complex<float>, DEVICE_CPU> gemvBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<float> alpha(2.0f, 0.0f);
    const std::complex<float> beta(3.0f, 0.0f);

    std::vector<std::complex<float>*> A = {};
    std::vector<std::complex<float>*> x = {};
    std::vector<std::complex<float>*> y = {};

    std::complex<float> _A[] = {
        {1.0f, 2.0f}, {3.0f, 4.0f},
        {5.0f, 6.0f}, {7.0f, 8.0f},
        {9.0f, 10.0f}, {11.0f, 12.0f},

        {13.0f, 14.0f}, {15.0f, 16.0f},
        {17.0f, 18.0f}, {19.0f, 20.0f},
        {21.0f, 22.0f}, {23.0f, 24.0f}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<float> _x[] = {{1.0f, 0.0f}, {2.0f, 0.0f}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<float> _y1[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    std::complex<float> _y2[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y.data(), 1, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        gemvCalculator(trans, m, n, &alpha, A[i], m, x[i], 1, &beta, _y2 + i * m, 1);
    }
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}

TEST(BlasOpTest, GemvBatchedComplexDouble) {
    blas_gemv<std::complex<double>, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched<std::complex<double>, DEVICE_CPU> gemvBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<double> alpha(2.0, 0.0);
    const std::complex<double> beta(3.0, 0.0);

    std::vector<std::complex<double>*> A = {};
    std::vector<std::complex<double>*> x = {};
    std::vector<std::complex<double>*> y = {};

    std::complex<double> _A[] = {
        {1.0, 2.0}, {3.0, 4.0},
        {5.0, 6.0}, {7.0, 8.0},
        {9.0, 10.0}, {11.0, 12.0},

        {13.0, 14.0}, {15.0, 16.0},
        {17.0, 18.0}, {19.0, 20.0},
        {21.0, 22.0}, {23.0, 24.0}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<double> _x[] = {{1.0, 0.0}, {2.0, 0.0}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<double> _y1[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    std::complex<double> _y2[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y.data(), 1, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        gemvCalculator(trans, m, n, &alpha, A[i], m, x[i], 1, &beta, _y2 + i * m, 1);
    }
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}

// Gemv batched test cases
TEST(BlasOpTest, GemvBatchedStridedFloat) {
    blas_gemv<float, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched_strided<float, DEVICE_CPU> gemvBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const float alpha = 2.0f;
    const float beta = 3.0f;
    
    std::vector<float*> A = {};
    std::vector<float*> x = {};
    std::vector<float*> y = {};
    float _A[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    float _x[] = {1.0f, 2.0f};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    float _y1[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float _y2[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y[0], 1, m, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        gemvCalculator(trans, m, n, &alpha, A[i], m, x[i], 1, &beta, _y2 + i * m, 1);
    }
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemvBatchedStridedDouble) {
    blas_gemv<double, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched_strided<double, DEVICE_CPU> gemvBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const double alpha = 2.0;
    const double beta = 3.0;
    
    std::vector<double*> A = {};
    std::vector<double*> x = {};
    std::vector<double*> y = {};
    double _A[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,

        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    double _x[] = {1.0, 2.0};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    double _y1[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double _y2[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y[0], 1, m, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        gemvCalculator(trans, m, n, &alpha, A[i], m, x[i], 1, &beta, _y2 + i * m, 1);
    }
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemvBatchedStridedComplex) {
    blas_gemv<std::complex<float>, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched_strided<std::complex<float>, DEVICE_CPU> gemvBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<float> alpha(2.0, 0.0);
    const std::complex<float> beta(3.0, 0.0);

    std::vector<std::complex<float>*> A = {};
    std::vector<std::complex<float>*> x = {};
    std::vector<std::complex<float>*> y = {};

    std::complex<float> _A[] = {
        {1.0, 2.0}, {3.0, 4.0},
        {5.0, 6.0}, {7.0, 8.0},
        {9.0, 10.0}, {11.0, 12.0},

        {13.0, 14.0}, {15.0, 16.0},
        {17.0, 18.0}, {19.0, 20.0},
        {21.0, 22.0}, {23.0, 24.0}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<float> _x[] = {{1.0, 0.0}, {2.0, 0.0}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<float> _y1[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    std::complex<float> _y2[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y[0], 1, m, batch_size);

    for (int ii = 0; ii < batch_size; ii++) {
            // Call the single GEMV for each pair of matrix A[ii] and vector x[ii]
            gemvCalculator(trans, m, n, &alpha, A[0] + ii * m * n, m, x[0] + ii * 0, 1, &beta, _y2 + ii * m, 1);
    } 
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}


TEST(BlasOpTest, GemvBatchedStridedComplexDouble) {
    blas_gemv<std::complex<double>, DEVICE_CPU> gemvCalculator;
    blas_gemv_batched_strided<std::complex<double>, DEVICE_CPU> gemvBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<double> alpha(2.0f, 0.0f);
    const std::complex<double> beta(3.0f, 0.0f);

    std::vector<std::complex<double>*> A = {};
    std::vector<std::complex<double>*> x = {};
    std::vector<std::complex<double>*> y = {};

    std::complex<double> _A[] = {
        {1.0f, 2.0f}, {3.0f, 4.0f},
        {5.0f, 6.0f}, {7.0f, 8.0f},
        {9.0f, 10.0f}, {11.0f, 12.0f},

        {13.0f, 14.0f}, {15.0f, 16.0f},
        {17.0f, 18.0f}, {19.0f, 20.0f},
        {21.0f, 22.0f}, {23.0f, 24.0f}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<double> _x[] = {{1.0f, 0.0f}, {2.0f, 0.0f}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<double> _y1[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    std::complex<double> _y2[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    y.push_back(&_y1[0]);
    y.push_back(&_y1[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y[0], 1, m, batch_size);

    for (int ii = 0; ii < batch_size; ii++) {
            // Call the single GEMV for each pair of matrix A[ii] and vector x[ii]
            gemvCalculator(trans, m, n, &alpha, A[0] + ii * m * n, m, x[0] + ii * 0, 1, &beta, _y2 + ii * m, 1);
    } 
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}



TEST(BlasOpTest, GemmFloat) {
    blas_gemm<float, DEVICE_CPU> gemmCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const float alpha = 2.0f;
    const float beta = 3.0f;
    const float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float x[] = {1.0f, 2.0f};
    float y[] = {1.0f, 2.0f, 3.0f};
    float expected[] = {21.0f, 30.0f, 39.0f};

    gemmCalculator(trans, trans, m, 1, n, &alpha, A, m, x, n, &beta, y, m);

    for (int i = 0; i < m; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }
}

TEST(BlasOpTest, GemmDouble) {
    blas_gemm<double, DEVICE_CPU> gemmCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const double alpha = 2.0;
    const double beta = 3.0;
    const double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double x[] = {1.0, 2.0};
    double y[] = {1.0, 2.0, 3.0};
    double expected[] = {21.0, 30.0, 39.0};

    gemmCalculator(trans, trans, m, 1, n, &alpha, A, m, x, n, &beta, y, m);

    for (int i = 0; i < m; ++i) {
        EXPECT_DOUBLE_EQ(y[i], expected[i]);
    }
}


TEST(BlasOpTest, GemmComplex) {
    blas_gemv<std::complex<float>, DEVICE_CPU> gemvCalculator;
    blas_gemm<std::complex<float>, DEVICE_CPU> gemmCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const std::complex<float> alpha(1.0f, 0.0f);
    const std::complex<float> beta(0.0f, 0.0f);
    const std::complex<float> A[] = {
        std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, 2.0f),
        std::complex<float>(3.0f, 3.0f), std::complex<float>(4.0f, 4.0f),
        std::complex<float>(5.0f, 5.0f), std::complex<float>(6.0f, 6.0f)
    };
    const std::complex<float> x[] = {std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, 2.0f)};
    std::complex<float> y[] = {std::complex<float>(4.0f, 4.0f), std::complex<float>(5.0f, 5.0f), std::complex<float>(6.0f, 6.0f)};
    std::complex<float> expected[] = {std::complex<float>(4.0f, 4.0f), std::complex<float>(5.0f, 5.0f), std::complex<float>(6.0f, 6.0f)};

    gemvCalculator(trans, m, n, &alpha, A, m, x, 1, &beta, expected, 1);
    gemmCalculator(trans, trans, m, 1, n, &alpha, A, m, x, n, &beta, y, m);

    for (int i = 0; i < m; ++i) {
        EXPECT_FLOAT_EQ(y[i].real(), expected[i].real());
        EXPECT_FLOAT_EQ(y[i].imag(), expected[i].imag());
    }
}

TEST(BlasOpTest, GemmComplexDouble) {
    blas_gemv<std::complex<double>, DEVICE_CPU> gemvCalculator;
    blas_gemm<std::complex<double>, DEVICE_CPU> gemmCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);
    const std::complex<double> A[] = {
        std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0),
        std::complex<double>(3.0, 3.0), std::complex<double>(4.0, 4.0),
        std::complex<double>(5.0, 5.0), std::complex<double>(6.0, 6.0)
    };
    const std::complex<double> x[] = {std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0)};
    std::complex<double> y[] = {std::complex<double>(4.0, 4.0), std::complex<double>(5.0, 5.0), std::complex<double>(6.0, 6.0)};
    std::complex<double> expected[] = {std::complex<double>(4.0, 4.0), std::complex<double>(5.0, 5.0), std::complex<double>(6.0, 6.0)};

    gemvCalculator(trans, m, n, &alpha, A, m, x, 1, &beta, expected, 1);
    gemmCalculator(trans, trans, m, 1, n, &alpha, A, m, x, n, &beta, y, m);

    for (int i = 0; i < m; ++i) {
        EXPECT_DOUBLE_EQ(y[i].real(), expected[i].real());
        EXPECT_DOUBLE_EQ(y[i].imag(), expected[i].imag());
    }
}

// Gemm batched test cases
TEST(BlasOpTest, GemmBatchedFloat) {
    blas_gemv_batched<float, DEVICE_CPU> gemvBatchedCalculator;
    blas_gemm_batched<float, DEVICE_CPU> gemmBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const float alpha = 2.0f;
    const float beta = 3.0f;
    
    std::vector<float*> A = {};
    std::vector<float*> x = {};
    std::vector<float*> y1 = {};
    std::vector<float*> y2 = {};
    float _A[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    float _x[] = {1.0f, 2.0f};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    float _y1[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float _y2[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y1.data(), 1, batch_size);
    gemmBatchedCalculator(trans, trans, m, 1, n, &alpha, A.data(), m, x.data(), n, &beta, y2.data(), m, batch_size);
    
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemmBatchedDouble) {
    blas_gemv_batched<double, DEVICE_CPU> gemvBatchedCalculator;
    blas_gemm_batched<double, DEVICE_CPU> gemmBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const double alpha = 2.0;
    const double beta = 3.0;
    
    std::vector<double*> A = {};
    std::vector<double*> x = {};
    std::vector<double*> y1 = {};
    std::vector<double*> y2 = {};
    double _A[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,

        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    double _x[] = {1.0, 2.0};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    double _y1[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double _y2[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y1.data(), 1, batch_size);
    gemmBatchedCalculator(trans, trans, m, 1, n, &alpha, A.data(), m, x.data(), n, &beta, y2.data(), m, batch_size);
    
    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_DOUBLE_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemmBatchedComplex) {
    blas_gemv_batched<std::complex<float>, DEVICE_CPU> gemvBatchedCalculator;
    blas_gemm_batched<std::complex<float>, DEVICE_CPU> gemmBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<float> alpha(2.0f, 0.0f);
    const std::complex<float> beta(3.0f, 0.0f);

    std::vector<std::complex<float>*> A = {};
    std::vector<std::complex<float>*> x = {};
    std::vector<std::complex<float>*> y1 = {};
    std::vector<std::complex<float>*> y2 = {};

    std::complex<float> _A[] = {
        {1.0f, 2.0f}, {3.0f, 4.0f},
        {5.0f, 6.0f}, {7.0f, 8.0f},
        {9.0f, 10.0f}, {11.0f, 12.0f},

        {13.0f, 14.0f}, {15.0f, 16.0f},
        {17.0f, 18.0f}, {19.0f, 20.0f},
        {21.0f, 22.0f}, {23.0f, 24.0f}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<float> _x[] = {{1.0f, 0.0f}, {2.0f, 0.0f}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<float> _y1[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    std::complex<float> _y2[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y1.data(), 1, batch_size);
    gemmBatchedCalculator(trans, trans, m, 1, n, &alpha, A.data(), m, x.data(), n, &beta, y2.data(), m, batch_size);

    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}

TEST(BlasOpTest, GemmBatchedComplexDouble) {
    blas_gemv_batched<std::complex<double>, DEVICE_CPU> gemvBatchedCalculator;
    blas_gemm_batched<std::complex<double>, DEVICE_CPU> gemmBatchedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<double> alpha(2.0, 0.0);
    const std::complex<double> beta(3.0, 0.0);

    std::vector<std::complex<double>*> A = {};
    std::vector<std::complex<double>*> x = {};
    std::vector<std::complex<double>*> y1 = {};
    std::vector<std::complex<double>*> y2 = {};

    std::complex<double> _A[] = {
        {1.0, 2.0}, {3.0, 4.0},
        {5.0, 6.0}, {7.0, 8.0},
        {9.0, 10.0}, {11.0, 12.0},

        {13.0, 14.0}, {15.0, 16.0},
        {17.0, 18.0}, {19.0, 20.0},
        {21.0, 22.0}, {23.0, 24.0}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<double> _x[] = {{1.0, 0.0}, {2.0, 0.0}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<double> _y1[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    std::complex<double> _y2[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedCalculator(trans, m, n, &alpha, A.data(), m, x.data(), 1, &beta, y1.data(), 1, batch_size);
    gemmBatchedCalculator(trans, trans, m, 1, n, &alpha, A.data(), m, x.data(), n, &beta, y2.data(), m, batch_size);

    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}


// Gemm batched test cases
TEST(BlasOpTest, GemmBatchedStridedFloat) {
    blas_gemv_batched_strided<float, DEVICE_CPU> gemvBatchedStridedCalculator;
    blas_gemm_batched_strided<float, DEVICE_CPU> gemmBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const float alpha = 2.0f;
    const float beta = 3.0f;
    
    std::vector<float*> A = {};
    std::vector<float*> x = {};
    std::vector<float*> y1 = {};
    std::vector<float*> y2 = {};
    float _A[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    float _x[] = {1.0f, 2.0f};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    float _y1[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float _y2[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y1[0], 1, m, batch_size);
    gemmBatchedStridedCalculator(trans, trans, m, 1, n, &alpha, A[0], m, m * n, x[0], n, 0, &beta, y2[0], m, m, batch_size);

    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemmBatchedStridedDouble) {
    blas_gemv_batched_strided<double, DEVICE_CPU> gemvBatchedStridedCalculator;
    blas_gemm_batched_strided<double, DEVICE_CPU> gemmBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const double alpha = 2.0;
    const double beta = 3.0;
    
    std::vector<double*> A = {};
    std::vector<double*> x = {};
    std::vector<double*> y1 = {};
    std::vector<double*> y2 = {};
    double _A[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,

        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);
    
    double _x[] = {1.0, 2.0};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);
    
    double _y1[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double _y2[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y1[0], 1, m, batch_size);
    gemmBatchedStridedCalculator(trans, trans, m, 1, n, &alpha, A[0], m, m * n, x[0], n, 0, &beta, y2[0], m, m, batch_size);

    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(_y1[i], _y2[i]);
    }
}

TEST(BlasOpTest, GemmBatchedStridedComplex) {
    blas_gemv_batched_strided<std::complex<float>, DEVICE_CPU> gemvBatchedStridedCalculator;
    blas_gemm_batched_strided<std::complex<float>, DEVICE_CPU> gemmBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<float> alpha(2.0f, 0.0f);
    const std::complex<float> beta(3.0f, 0.0f);

    std::vector<std::complex<float>*> A = {};
    std::vector<std::complex<float>*> x = {};
    std::vector<std::complex<float>*> y1 = {};
    std::vector<std::complex<float>*> y2 = {};

    std::complex<float> _A[] = {
        {1.0f, 2.0f}, {3.0f, 4.0f},
        {5.0f, 6.0f}, {7.0f, 8.0f},
        {9.0f, 10.0f}, {11.0f, 12.0f},

        {13.0f, 14.0f}, {15.0f, 16.0f},
        {17.0f, 18.0f}, {19.0f, 20.0f},
        {21.0f, 22.0f}, {23.0f, 24.0f}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<float> _x[] = {{1.0f, 0.0f}, {2.0f, 0.0f}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<float> _y1[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    std::complex<float> _y2[6] = {{4.0f, 0.0f}, {5.0f, 0.0f}, {6.0f, 0.0f},
                                  {7.0f, 0.0f}, {8.0f, 0.0f}, {9.0f, 0.0f}};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y1[0], 1, m, batch_size);
    gemmBatchedStridedCalculator(trans, trans, m, 1, n, &alpha, A[0], m, m * n, x[0], n, 0, &beta, y2[0], m, m, batch_size);

    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}


TEST(BlasOpTest, GemmBatchedStridedComplexDouble) {
    blas_gemv_batched_strided<std::complex<double>, DEVICE_CPU> gemvBatchedStridedCalculator;
    blas_gemm_batched_strided<std::complex<double>, DEVICE_CPU> gemmBatchedStridedCalculator;

    const char trans = 'N';
    const int m = 3;
    const int n = 2;
    const int batch_size = 2;
    const std::complex<double> alpha(2.0, 0.0);
    const std::complex<double> beta(3.0, 0.0);

    std::vector<std::complex<double>*> A = {};
    std::vector<std::complex<double>*> x = {};
    std::vector<std::complex<double>*> y1 = {};
    std::vector<std::complex<double>*> y2 = {};

    std::complex<double> _A[] = {
        {1.0, 2.0}, {3.0, 4.0},
        {5.0, 6.0}, {7.0, 8.0},
        {9.0, 10.0}, {11.0, 12.0},

        {13.0, 14.0}, {15.0, 16.0},
        {17.0, 18.0}, {19.0, 20.0},
        {21.0, 22.0}, {23.0, 24.0}
    };
    A.push_back(&_A[0]);
    A.push_back(&_A[m * n]);

    std::complex<double> _x[] = {{1.0, 0.0}, {2.0, 0.0}};
    x.push_back(&_x[0]);
    x.push_back(&_x[0]);

    std::complex<double> _y1[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    std::complex<double> _y2[6] = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0},
                                  {7.0, 0.0}, {8.0, 0.0}, {9.0, 0.0}};
    y1.push_back(&_y1[0]);
    y1.push_back(&_y1[m]);
    y2.push_back(&_y2[0]);
    y2.push_back(&_y2[m]);

    gemvBatchedStridedCalculator(trans, m, n, &alpha, A[0], m, m * n, x[0], 1, 0, &beta, y1[0], 1, m, batch_size);
    gemmBatchedStridedCalculator(trans, trans, m, 1, n, &alpha, A[0], m, m * n, x[0], n, 0, &beta, y2[0], m, m, batch_size);

    for (int i = 0; i < m * batch_size; ++i) {
        EXPECT_FLOAT_EQ(std::real(_y1[i]), std::real(_y2[i]));
        EXPECT_FLOAT_EQ(std::imag(_y1[i]), std::imag(_y2[i]));
    }
}

} // namespace op
} // namespace container

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
