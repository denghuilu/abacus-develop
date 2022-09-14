#ifndef MODULE_PSI_TYPES_H_
#define MODULE_PSI_TYPES_H_

#include <map>
#include <set>
#include <string>

namespace psi {

struct DEVICE_CPU;
struct DEVICE_GPU;
struct DEVICE_SYCL;

enum AbacusDevice_t {UnKnown, CpuDevice, GpuDevice, SyclDevice};

}  // end namespace psi

#endif  // MODULE_PSI_TYPES_H_