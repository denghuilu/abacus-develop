#ifndef MODULE_PSI_MEMORY_H_
#define MODULE_PSI_MEMORY_H_

#include "module_psi/include/types.h"

namespace psi {
namespace memory {

template<typename T>
void abacus_resize_memory(T*&, const int, const AbacusDevice_t);

template<typename T>
void abacus_memset(T*, const int, const int, const AbacusDevice_t);

template<typename T>
void abacus_sync_memory(T*, const T*, const size_t,  const AbacusDevice_t, const AbacusDevice_t);

#if __CUDA
template<typename T>
void abacus_resize_memory_gpu_cuda(T*&, const int);

template<typename T>
void abacus_memset_gpu_cuda(T*, const int, const int);

template <typename T>
void abacus_memcpy_device_to_device_gpu_cuda(T* , const T*, const int);

template <typename T>
void abacus_memcpy_host_to_device_gpu_cuda(T* , const T*, const int);

template <typename T>
void abacus_memcpy_device_to_host_gpu_cuda(T* , const T*, const int);

#elif __ROCM
template<typename T>
void abacus_resize_memory_gpu_rocm(T*&, const int, const AbacusDevice_t);

template<typename T>
void abacus_memset_gpu_rocm(T*&, const int, const int, const AbacusDevice_t);

template <typename T>
void abacus_memcpy_device_to_device_gpu_rocm(T* , const T*, const int);

template <typename T>
void abacus_memcpy_host_to_device_gpu_rocm(T* , const T*, const int);

template <typename T>
void abacus_memcpy_device_to_host_gpu_rocm(T* , const T*, const int);
#endif

} // end of namespace memory
} // end of namespace psi

#endif // MODULE_PSI_MEMORY_H_