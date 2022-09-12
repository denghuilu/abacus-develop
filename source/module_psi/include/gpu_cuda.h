#pragma once
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

namespace gpu_cuda {

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  cudaMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), cudaMemcpyDeviceToHost);  
}

template <typename FPTYPE>
void malloc_host_memory_sync(
    FPTYPE * &host,
    const FPTYPE * device,
    const int size)
{
  host = (FPTYPE *)malloc(sizeof(FPTYPE) * size);
  memcpy_device_to_host(device, host, size);
}

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
  if (device != NULL) {
    cudaFree(device);
  }
}

template <typename FPTYPE>
void memset_device_memory(
    FPTYPE * device, 
    const int var,
    const int size) 
{
  cudaMemset(device, var, sizeof(FPTYPE) * size);  
}

template <typename FPTYPE>
void abacus_resize_memory(
    FPTYPE * arr, 
    const int& size) 
{
  if (arr != nullptr) {
    delete_device_memory(arr);
  }
  malloc_device_memory(arr, size);  
}


template <typename FPTYPE>
void abacus_memset(
    FPTYPE * arr, 
    const int var,
    const int& size) 
{
  memset_device_memory(arr, var, size);  
}

template <typename FPTYPE>
void abacus_memcpy_device_to_device(
    FPTYPE * arr1, 
    const FPTYPE * arr2, 
    const int size) 
{
  cudaMemcpy(arr1, arr2, sizeof(FPTYPE) * size, cudaMemcpyDeviceToDevice);  
}

template <typename FPTYPE>
void abacus_memcpy_device_to_host(
    FPTYPE * arr1, 
    const FPTYPE * arr2, 
    const int size) 
{
  cudaMemcpy(arr1, arr2, sizeof(FPTYPE) * size, cudaMemcpyDeviceToHost);  
}

template <typename FPTYPE>
void abacus_memcpy_host_to_device(
    FPTYPE * arr1, 
    const FPTYPE * arr2, 
    const int size) 
{
  cudaMemcpy(arr1, arr2, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice);  
}


} // end of namespace deepmd