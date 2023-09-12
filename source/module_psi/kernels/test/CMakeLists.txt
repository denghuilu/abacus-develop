AddTest(
  TARGET Module_Psi_UTs
  LIBS ${math_libs} base device container
  SOURCES memory_test.cpp device_test.cpp
)