
#include <iostream>
#include "module_psi/include/types.h"
#include "module_psi/include/device.h"

namespace psi{


// functions used in custom ops
template<> std::string get_device_type <ABACUS::DEVICE_CPU> (ABACUS::DEVICE_CPU* dev) {
    return "CPU";
}

#if ((defined __CUDA) || (defined __ROCM))
template<> std::string get_device_type <ABACUS::DEVICE_GPU> (ABACUS::DEVICE_GPU* dev) {
    return "GPU";
}
#endif

}