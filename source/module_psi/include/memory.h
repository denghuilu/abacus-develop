#ifndef MODULE_PSI_MEMORY_H_
#define MODULE_PSI_MEMORY_H_

#include <string.h>
#include "module_psi/include/types.h"

#if __CUDA
#include <module_psi/include/gpu_cuda.h>
#elif __ROCM
#include <module_psi/include/gpu_cuda.h>
#endif

namespace psi {
namespace memory {

template<typename FPTYPE>
void abacus_resize_memory(FPTYPE * &arr, const int& size, const AbacusDevice_t& device) {
  if (device == CpuDevice) {
    if (arr != nullptr) {
      free(arr);
    }
    arr = (FPTYPE*) malloc(sizeof(FPTYPE) * size);
  }
  else if (device == GpuDevice) {
    #if __CUDA
      gpu_cuda::abacus_resize_memory(arr, size);
    #elif __ROCM
      gpu_rocm::abacus_resize_memory(arr, size);
    #endif
  }
}

template<typename FPTYPE>
void abacus_memset(FPTYPE * &arr, const int & val, const int& size, const AbacusDevice_t& device) {
  if (device == CpuDevice) {
    memset(arr, val, sizeof(FPTYPE) * size);
  }
  else if (device == GpuDevice) {
    #if __CUDA
      gpu_cuda::abacus_memset(arr, val, size);
    #elif __ROCM
      gpu_rocm::abacus_memset(arr, val, size);
    #endif
  }
}

} // end of namespace memory
} // end of namespace psi

#endif // MODULE_PSI_MEMORY_H_