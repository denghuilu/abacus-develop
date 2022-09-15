
#include <complex>
#include <iostream>
#include "module_psi/psi.h"
#include "module_psi/include/types.h"
#include "module_psi/include/device.h"

namespace psi{

namespace device{

void test_something(const Psi<std::complex<double>>& psi_in) {
    Psi<std::complex<double>, DEVICE_GPU>* psi_cuda = new Psi<std::complex<double>, DEVICE_GPU>(psi_in);
    std::cout << "oooo" << std::endl;
}

// functions used in custom ops
template<> AbacusDevice_t get_device_type <DEVICE_CPU> (DEVICE_CPU* dev) {
    return CpuDevice;
}

#if ((defined __CUDA) || (defined __ROCM))
template<> AbacusDevice_t get_device_type <DEVICE_GPU> (DEVICE_GPU* dev) {
    return GpuDevice;
}
#endif

} // end of namespace device
} // end of namespace psi