list(APPEND container_srcs
    ATen/core/tensor.cpp
    ATen/core/tensor_map.cpp
    ATen/core/cpu_allocator.cpp
    ATen/core/tensor_buffer.cpp
    ATen/core/tensor_shape.cpp
    ATen/core/tensor_types.cpp
)

if(USE_CUDA)
    list(APPEND container_srcs ATen/core/gpu_allocator.cpp)
endif()

add_library(container OBJECT ${container_srcs})

add_subdirectory(ATen/kernels)

target_link_libraries(container kernels)

if(ENABLE_COVERAGE)
  add_coverage(container)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
