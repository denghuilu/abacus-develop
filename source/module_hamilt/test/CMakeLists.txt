if (USE_CUDA)
AddTest(
  TARGET Hamilt_UTs
  LIBS ${math_libs} ELPA::ELPA
  SOURCES ekinetic_test.cpp 
          ../src/ekinetic.cpp
          ../src/cuda/ekinetic.cu
          ../../module_psi/src/memory.cpp
          ../../module_psi/src/cuda/memory.cu
)
else()
AddTest(
  TARGET Hamilt_UTs
  LIBS ${math_libs} ELPA::ELPA
  SOURCES ekinetic_test.cpp 
          ../src/ekinetic.cpp
          ../../module_psi/src/memory.cpp
)
endif()