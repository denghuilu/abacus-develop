#ifndef ABACUS_DEVICE_H_
#define ABACUS_DEVICE_H_

#include <iostream>
#include "module_psi/include/types.h"

namespace psi{

template<typename Device> std::string get_device_type (Device* dev);

}

#endif  // ABACUS_DEVICE_H_