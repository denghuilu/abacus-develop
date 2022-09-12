#ifndef ABACUS_MEMORY_H_
#define ABACUS_MEMORY_H_

#include <string.h>
#include "module_psi/include/types.h"

#if __CUDA
#include <module_psi/include/gpu_cuda.h>
#elif __ROCM
#include <module_psi/include/gpu_cuda.h>
#endif

template<typename FPTYPE>
void abacus_resize_memory(FPTYPE * &arr, const int& size, const std::string& device) {
  if (device == "CPU") {
    if (arr != nullptr) {
      free(arr);
    }
    arr = (FPTYPE*) malloc(sizeof(FPTYPE) * size);
  }
  else if (device == "GPU") {
    #if __CUDA
      gpu_cuda::abacus_resize_memory(arr, size);
    #elif __ROCM
      gpu_rocm::abacus_resize_memory(arr, size);
    #endif
  }
}

template<typename FPTYPE>
void abacus_memset(FPTYPE * &arr, const int & val, const int& size, const std::string& device) {
  if (device == "CPU") {
    memset(arr, val, sizeof(FPTYPE) * size);
  }
  else if (device == "GPU") {
    #if __CUDA
      gpu_cuda::abacus_memset(arr, val, size);
    #elif __ROCM
      gpu_rocm::abacus_memset(arr, val, size);
    #endif
  }
}

template<typename FPTYPE>
void abacus_memory_sync(
    FPTYPE * &arr1, 
    const FPTYPE * &arr2, 
    const int& size, 
    const std::string& device1, 
    const std::string& device2) 
{
  if (device1 == "CPU" and device2 == "CPU") {
    memcpy(arr1, arr2, size);
  }
  if (device1 == "GPU" and device2 == "GPU") {
    #if __CUDA
      gpu_cuda::abacus_memcpy_device_to_device(arr1, arr2, size);
    #elif __ROCM
      gpu_rocm::abacus_memcpy_device_to_device(arr1, arr2, size);
    #endif
  }
  else if (device1 == "CPU" and device2 == "GPU") {
    #if __CUDA
      gpu_cuda::abacus_memcpy_device_to_host(arr1, arr2, size);
    #elif __ROCM
      gpu_rocm::abacus_memcpy_device_to_host(arr1, arr2, size);
    #endif
  }
  else if (device1 == "GPU" and device2 == "CPU") {
    #if __CUDA
      gpu_cuda::abacus_memcpy_host_to_device(arr1, arr2, size);
    #elif __ROCM
      gpu_rocm::abacus_memcpy_host_to_device(arr1, arr2, size);
    #endif
  }
}

#endif // ABACUS_MEMORY_H_