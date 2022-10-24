list(APPEND planewave_srcs
    fft.cpp
    pw_basis.cpp
    pw_basis_k.cpp
    pw_distributeg.cpp
    pw_distributeg_method1.cpp
    pw_distributeg_method2.cpp
    pw_distributer.cpp
    pw_init.cpp
    pw_transform.cpp
    pw_transform_k.cpp
    src/fft_multi_device.cpp
)

if (USE_CUDA)
  list(APPEND planewave_srcs src/cuda/fft_multi_device.cu)
elseif(USE_ROCM)
  list(APPEND planewave_srcs src/rocm/fft_multi_device.hip)
endif()

add_library(
    planewave
    OBJECT
    ${planewave_srcs}
)

IF (BUILD_TESTING)
  add_subdirectory(test)
endif()
