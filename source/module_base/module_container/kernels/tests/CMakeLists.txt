AddTest(
  TARGET Container_Kernels_UTs
  LIBS ${math_libs} base device container kernels
  SOURCES blas_op_test.cpp lapack_op_test.cpp ../lapack_op.cpp
)