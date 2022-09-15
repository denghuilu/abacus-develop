#include <iostream>
#include <complex>
#include <string.h>
#include "module_psi/include/types.h"
#include "module_psi/include/memory.h"

namespace psi{
namespace memory{

template<typename T>
void abacus_resize_memory(T * &arr, const int size, const AbacusDevice_t device) {
  if (device == CpuDevice) {
    if (arr != nullptr) {
      free(arr);
    }
    arr = (T*) malloc(sizeof(T) * size);
  }
  else if (device == GpuDevice) {
    #if __CUDA
      abacus_resize_memory_gpu_cuda(arr, size);
    #elif __ROCM
      abacus_resize_memory_gpu_rocm(arr, size);
    #endif
  }
}

template<typename T>
void abacus_memset(T* arr, const int val, const int size, const AbacusDevice_t device) {
  if (device == CpuDevice) {
    memset(arr, val, sizeof(T) * size);
  }
  else if (device == GpuDevice) {
    #if __CUDA
      abacus_memset_gpu_cuda(arr, val, size);
    #elif __ROCM
      abacus_memset_gpu_rocm(arr, val, size);
    #endif
  }
}

template<typename T>
void abacus_sync_memory(
    T* arr1,
    const T* arr2,
    const size_t size, 
    const AbacusDevice_t dev1,
    const AbacusDevice_t dev2) 
{
  if (dev1 == dev2) {
    if (dev1 == CpuDevice) {
      memcpy(arr1, arr2, sizeof(T) * size);
    }
    else {
      #if __CUDA
      abacus_memcpy_device_to_device_gpu_cuda(arr1, arr2, size);
      #elif __ROCM
      abacus_memcpy_device_to_device_gpu_rocm(arr1, arr2, size);
      #endif
    }
  }
  else {
    if (dev1 == CpuDevice) {
      #if __CUDA
      abacus_memcpy_device_to_host_gpu_cuda(arr1, arr2, size);
      #elif __ROCM
      abacus_memcpy_device_to_host_gpu_rocm(arr1, arr2, size);
      #endif
    }
    else {
      #if __CUDA
      abacus_memcpy_host_to_device_gpu_cuda(arr1, arr2, size);
      #elif __ROCM
      abacus_memcpy_host_to_device_gpu_rocm(arr1, arr2, size);
      #endif
    }
  }
}

template<typename T>
void abacus_delete_memory(
    T* arr,
    const AbacusDevice_t device) 
{
  if (device == CpuDevice) {
    free(arr);
  }
  else{
    #if __CUDA
      abacus_delete_memory_gpu_cuda(arr);
    #elif __ROCM
      abacus_delete_memory_gpu_rocm(arr);
    #endif
  }
}

template void abacus_resize_memory<double>(double*&, const int, const AbacusDevice_t);
template void abacus_resize_memory<std::complex<double>>(std::complex<double>*&, const int, const AbacusDevice_t);

template void abacus_memset<double>(double*, const int, const int, const AbacusDevice_t);
template void abacus_memset<std::complex<double>>(std::complex<double>*, const int, const int, const AbacusDevice_t);

template void abacus_sync_memory<double>(double*, const double*, const size_t,  const AbacusDevice_t, const AbacusDevice_t);
template void abacus_sync_memory<std::complex<double>>(std::complex<double>*, const std::complex<double>*, const size_t,  const AbacusDevice_t, const AbacusDevice_t);

template void abacus_delete_memory<double>(double*, const AbacusDevice_t);
template void abacus_delete_memory<std::complex<double>>(std::complex<double>*, const AbacusDevice_t);

}
}