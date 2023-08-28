list(APPEND kernels_srcs
    blas_op.cpp
    lapack_op.cpp
    memory_op.cpp
    linalg_op.cpp
    einsum_op.cpp
)

if(USE_CUDA)
    list(APPEND kernels_srcs
        cuda/memory_op.cu
        cuda/lapack_op.cu
        cuda/blas_op.cu
        cuda/linalg_op.cu
    )
endif()

if(USE_ROCM)
    hip_add_library(kernels_rocm STATIC
        rocm/memory_op.hip.cu
    )
endif()

add_library(kernels OBJECT ${kernels_srcs})

if(BUILD_TESTING)
  add_subdirectory(test)
endif()