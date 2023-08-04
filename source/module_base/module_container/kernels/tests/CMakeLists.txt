AddTest(
  TARGET Container_Kernels_UTs
  LIBS ${math_libs} base device
  SOURCES blas_op_test.cpp
)

target_link_libraries(Container_Kernels_UTs container kernels)