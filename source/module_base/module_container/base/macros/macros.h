#ifndef BASE_MACROS_MACROS_H_
#define BASE_MACROS_MACROS_H_

#include <stdint.h>
#include <base/utils/logging.h>

#if __CUDA
#include <base/macros/cuda.h> 
#elif __ROCM
#include <base/macros/rocm.h>
#endif

#include <ATen/core/tensor.h>
#include <ATen/core/tensor_types.h>

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete

#define DISALLOW_MOVE_AND_ASSIGN(TypeName) \
  TypeName(TypeName&&) = delete;           \
  void operator=(TypeName&&) = delete

#define DISALLOW_COPY_MOVE_AND_ASSIGN(TypeName) \
  DISALLOW_COPY_AND_ASSIGN(TypeName);           \
  DISALLOW_MOVE_AND_ASSIGN(TypeName)

#define DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName) \
  TypeName() = delete;                           \
  DISALLOW_COPY_MOVE_AND_ASSIGN(TypeName)

#define MAX_SIZE_T UINT64_MAX

#define AT_HOST
#define AT_DEVICE
#define AT_HOST_DEVICE

#if defined(__CUDACC__) || defined(__HIPCC__)
#define AT_HOST __host__
#define AT_DEVICE __device__
#define AT_HOST_DEVICE __host__ __device__
#endif // defined(__CUDACC__) || defined(__HIPCC__)

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define PREDICT_TRUE(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define PREDICT_FALSE(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define PREDICT_TRUE(expr) (expr)
#define PREDICT_FALSE(expr) (expr)
#endif

#define CHECK_MSG(expr, ...)                                  \
    (::base::utils::check_msg_impl(                           \
        "Expected " #expr                                     \
        " to be true, but got false.  "                       \
        __VA_ARGS__ " "                                       \
        "(Could this error message be improved?  If so, "     \
        "please report an enhancement request to Container)"  \
        ))

#define REQUIRES_OK(expr, ...)                  \
    if(PREDICT_FALSE(!(expr))) {                \
        ::base::utils::check_exit_impl(         \
            __func__,                           \
            __FILE__,                           \
            static_cast<uint32_t>(__LINE__),    \
            CHECK_MSG(expr, ##__VA_ARGS__));    \
    }

// The macro TEMPLATE_1() expands to a switch statement conditioned on
// TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__

#define CASE_2(TYPE, DEVICE, STMTS)               \
  case (int(DataTypeToEnum<TYPE>::value) * 10 +   \
        int(DeviceTypeToEnum<DEVICE>::value)): {  \
    typedef TYPE T_;                              \
    typedef DEVICE DEVICE_;                       \
    STMTS;                                        \
    break;                                        \
  }

#define CASES_ALL_WITH_DEFAULT_2(TYPE_ENUM, DEVICE_ENUM, STMTS, DEFAULT) \
  switch (int(TYPE_ENUM) * 10 + int(DEVICE_ENUM)) {                      \
    CASE_2(float, DEVICE_CPU, SINGLE_ARG(STMTS))                         \
    CASE_2(double, DEVICE_CPU, SINGLE_ARG(STMTS))                        \
    CASE_2(int, DEVICE_CPU, SINGLE_ARG(STMTS))                           \
    CASE_2(int64_t, DEVICE_CPU, SINGLE_ARG(STMTS))                       \
    CASE_2(std::complex<float>, DEVICE_CPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<double>, DEVICE_CPU, SINGLE_ARG(STMTS))          \
    default:                                                             \
      DEFAULT;                                                           \
      break;                                                             \
  }

#define CASES_BLAS_WITH_DEFAULT_2(TYPE_ENUM, DEVICE_ENUM, STMTS, DEFAULT) \
  switch (int(TYPE_ENUM) * 10 + int(DEVICE_ENUM)) {                      \
    CASE_2(float, DEVICE_CPU, SINGLE_ARG(STMTS))                         \
    CASE_2(double, DEVICE_CPU, SINGLE_ARG(STMTS))                        \
    CASE_2(std::complex<float>, DEVICE_CPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<double>, DEVICE_CPU, SINGLE_ARG(STMTS))          \
    default:                                                             \
      DEFAULT;                                                           \
      break;                                                             \
  }

#define CASES_ALL_WITH_DEFAULT_2_GPU(TYPE_ENUM, DEVICE_ENUM, STMTS, DEFAULT) \
  switch (int(TYPE_ENUM) * 10 + int(DEVICE_ENUM)) {                      \
    CASE_2(float, DEVICE_CPU, SINGLE_ARG(STMTS))                         \
    CASE_2(float, DEVICE_GPU, SINGLE_ARG(STMTS))                         \
    CASE_2(double, DEVICE_CPU, SINGLE_ARG(STMTS))                        \
    CASE_2(double, DEVICE_GPU, SINGLE_ARG(STMTS))                        \
    CASE_2(int, DEVICE_CPU, SINGLE_ARG(STMTS))                           \
    CASE_2(int, DEVICE_GPU, SINGLE_ARG(STMTS))                           \
    CASE_2(int64_t, DEVICE_CPU, SINGLE_ARG(STMTS))                       \
    CASE_2(int64_t, DEVICE_GPU, SINGLE_ARG(STMTS))                       \
    CASE_2(std::complex<float>, DEVICE_CPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<float>, DEVICE_GPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<double>, DEVICE_CPU, SINGLE_ARG(STMTS))          \
    CASE_2(std::complex<double>, DEVICE_GPU, SINGLE_ARG(STMTS))          \
    default:                                                             \
      DEFAULT;                                                           \
      break;                                                             \
  }

#define CASES_BLAS_WITH_DEFAULT_2_GPU(TYPE_ENUM, DEVICE_ENUM, STMTS, DEFAULT) \
  switch (int(TYPE_ENUM) * 10 + int(DEVICE_ENUM)) {                      \
    CASE_2(float, DEVICE_CPU, SINGLE_ARG(STMTS))                         \
    CASE_2(float, DEVICE_GPU, SINGLE_ARG(STMTS))                         \
    CASE_2(double, DEVICE_CPU, SINGLE_ARG(STMTS))                        \
    CASE_2(double, DEVICE_GPU, SINGLE_ARG(STMTS))                        \
    CASE_2(std::complex<float>, DEVICE_CPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<float>, DEVICE_GPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<double>, DEVICE_CPU, SINGLE_ARG(STMTS))          \
    CASE_2(std::complex<double>, DEVICE_GPU, SINGLE_ARG(STMTS))          \
    default:                                                             \
      DEFAULT;                                                           \
      break;                                                             \
  }

#define CASES_CZ_WITH_DEFAULT_2(TYPE_ENUM, DEVICE_ENUM, STMTS, DEFAULT)  \
  switch (int(TYPE_ENUM) * 10 + int(DEVICE_ENUM)) {                      \
    CASE_2(std::complex<float>, DEVICE_CPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<float>, DEVICE_GPU, SINGLE_ARG(STMTS))           \
    CASE_2(std::complex<double>, DEVICE_CPU, SINGLE_ARG(STMTS))          \
    CASE_2(std::complex<double>, DEVICE_GPU, SINGLE_ARG(STMTS))          \
    default:                                                             \
      DEFAULT;                                                           \
      break;                                                             \
  }


#if __CUDA || __ROCM
#define TEMPLATE_ALL_2(TYPE_ENUM, DEVICE_ENUM, ...)                         \
CASES_ALL_WITH_DEFAULT_2_GPU(TYPE_ENUM, DEVICE_ENUM, (__VA_ARGS__),         \
                       std::cerr << "Unexpected type: " << TYPE_ENUM; exit(EXIT_FAILURE));
#define TEMPLATE_BLAS_2(TYPE_ENUM, DEVICE_ENUM, ...)                        \
CASES_BLAS_WITH_DEFAULT_2_GPU(TYPE_ENUM, DEVICE_ENUM, (__VA_ARGS__),        \
                       std::cerr << "Unexpected type: " << TYPE_ENUM; exit(EXIT_FAILURE));
#define TEMPLATE_ALL_CALC_2(TYPE_ENUM, DEVICE_ENUM, ...)                    \
CASES_ALL_CALC_WITH_DEFAULT_2_GPU(TYPE_ENUM, DEVICE_ENUM, (__VA_ARGS__),    \
                       std::cerr << "Unexpected type: " << TYPE_ENUM; exit(EXIT_FAILURE));
#else 
#define TEMPLATE_ALL_2(TYPE_ENUM, DEVICE_ENUM, ...)                         \
CASES_ALL_WITH_DEFAULT_2(TYPE_ENUM, DEVICE_ENUM, (__VA_ARGS__),             \
                       std::cerr << "Unexpected type: " << TYPE_ENUM; exit(EXIT_FAILURE));
#define TEMPLATE_BLAS_2(TYPE_ENUM, DEVICE_ENUM, ...)                        \
CASES_BLAS_WITH_DEFAULT_2(TYPE_ENUM, DEVICE_ENUM, (__VA_ARGS__),            \
                       std::cerr << "Unexpected type: " << TYPE_ENUM; exit(EXIT_FAILURE));
#define TEMPLATE_ALL_CALC_2(TYPE_ENUM, DEVICE_ENUM, ...)                    \
CASES_ALL_WITH_DEFAULT_2(TYPE_ENUM, DEVICE_ENUM, (__VA_ARGS__),             \
                       std::cerr << "Unexpected type: " << TYPE_ENUM; exit(EXIT_FAILURE));
#endif 

#define TEMPLATE_CZ_2(TYPE_ENUM, DEVICE_ENUM, ...)                          \
CASES_CZ_WITH_DEFAULT_2(TYPE_ENUM, DEVICE_ENUM, (__VA_ARGS__),              \
                       std::cerr << "Unexpected type: " << TYPE_ENUM; exit(EXIT_FAILURE));


#if defined(__CUDACC__) || defined(__HIPCC__)
#define ATEN_HOST __host__
#define ATEN_DEVICE __device__
#define ATEN_HOST_DEVICE __host__ __device__
#else
#define ATEN_HOST
#define ATEN_DEVICE
#define ATEN_HOST_DEVICE
#endif

#if defined(_MSC_VER)
#define ATEN_ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ATEN_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ATEN_ALWAYS_INLINE inline
#endif

#endif // BASE_MACROS_MACROS_H_