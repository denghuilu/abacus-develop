#ifndef MODULE_PSI_GPU_CUDA_H_
#define MODULE_PSI_GPU_CUDA_H_

#include <vector>
#include <stdio.h>
#include <complex>
#include <assert.h>
#include <cuda_runtime.h>

namespace psi {
namespace memory {

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  cudaMemcpy(device, host, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice);  
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  cudaMalloc((void **)&device, sizeof(FPTYPE) * size);
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device,
    const int size)
{
  cudaMalloc((void **)&device, sizeof(FPTYPE) * size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  cudaFree(device);
}

template <typename FPTYPE>
void memset_device_memory(
    FPTYPE * device, 
    const int var,
    const int size) 
{
  cudaMemset(device, var, sizeof(FPTYPE) * size);  
}

template <typename T>
void abacus_resize_memory_gpu_cuda(
    T*& arr, 
    const int size) 
{
  if (arr != nullptr) {
    delete_device_memory(arr);
  }
  malloc_device_memory(arr, size);  
}


template <typename T>
void abacus_memset_gpu_cuda(
    T* arr, 
    const int var,
    const int size) 
{
  memset_device_memory(arr, var, size);  
}

template <typename T>
void abacus_memcpy_device_to_device_gpu_cuda(
    T* arr1, 
    const T* arr2, 
    const int size) 
{
  cudaMemcpy(arr1, arr2, sizeof(T) * size, cudaMemcpyDeviceToDevice);  
}


template <typename T>
void abacus_memcpy_device_to_host_gpu_cuda(
    T* arr1, 
    const T* arr2, 
    const int size) 
{
  cudaMemcpy(arr1, arr2, sizeof(T) * size, cudaMemcpyDeviceToHost);  
}

template <typename T>
void abacus_memcpy_host_to_device_gpu_cuda(
    T* arr1, 
    const T* arr2, 
    const int size) 
{
  cudaMemcpy(arr1, arr2, sizeof(T) * size, cudaMemcpyHostToDevice);  
}

template void abacus_resize_memory_gpu_cuda<double>(double*&, const int);
template void abacus_resize_memory_gpu_cuda<std::complex<double>>(std::complex<double>*&, const int);

template void abacus_memset_gpu_cuda<double>(double*, const int, const int);
template void abacus_memset_gpu_cuda<std::complex<double>>(std::complex<double>*, const int, const int);

template void abacus_memcpy_device_to_device_gpu_cuda<double>(double*, const double *, const int);
template void abacus_memcpy_device_to_device_gpu_cuda<std::complex<double>>(std::complex<double>*, const std::complex<double>*, const int);

template void abacus_memcpy_device_to_host_gpu_cuda<double>(double*, const double *, const int);
template void abacus_memcpy_device_to_host_gpu_cuda<std::complex<double>>(std::complex<double>*, const std::complex<double>*, const int);

template void abacus_memcpy_host_to_device_gpu_cuda<double>(double*, const double *, const int);
template void abacus_memcpy_host_to_device_gpu_cuda<std::complex<double>>(std::complex<double>*, const std::complex<double>*, const int);

} // end of namespace gpu_cuda
} // end of namespace psi

#endif  // MODULE_PSI_GPU_CUDA_H_