list(APPEND hsolver_srcs
    diago_cg.cpp
    diago_david.cpp
    hsolver_pw.cpp
    hsolver_pw_sdft.cpp
    diago_iter_assist.cpp
    hsolver_lcao.cpp
    diago_blas.cpp
    diago_elpa.cpp
    diago_lapack.cpp
    src/math_kernel.cpp
)

if (USE_CUDA) 
    list(APPEND hsolver_srcs src/cuda/math_kernel.cu)
elseif(USE_ROCM)
    list(APPEND hsolver_srcs src/rocm/math_kernel.cu)
endif()

add_library(
    hsolver
    OBJECT
    ${hsolver_srcs}
)

add_subdirectory(genelpa)

IF (BUILD_TESTING)
  add_subdirectory(test)
endif()
