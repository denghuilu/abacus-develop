#ifndef ABACUS_TYPES_H_
#define ABACUS_TYPES_H_

#include <map>
#include <set>
#include <string>

namespace ABACUS {
struct DEVICE_CPU;
struct DEVICE_GPU;
struct DEVICE_SYCL;

extern DEVICE_CPU CpuDevice;
extern DEVICE_GPU GpuDevice;
extern DEVICE_SYCL SyclDevice;
}  // end namespace Eigen



#endif  // ABACUS_TYPES_H_