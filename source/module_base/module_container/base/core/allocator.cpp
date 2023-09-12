#include <base/core/allocator.h>
#include <base/core/cpu_allocator.h>
#include <base/core/bfc_allocator.h>

#if defined(__CUDA) || defined(__ROCM)
#include <base/core/gpu_allocator.h>
#endif

namespace container {
namespace base {

Allocator* Allocator::get_singleton_instance(DeviceType device) {
    base::Allocator* alloc = nullptr;
    if (device == DeviceType::CpuDevice) {
        alloc = base::CPUAllocator::get_singleton_instance();
    }
#if defined(__CUDA) || defined(__ROCM)
    else if (device == DeviceType::GpuDevice) {
        alloc = base::BFCAllocator::get_singleton_instance();
    }
#endif // __CUDA || __ROCM
    REQUIRES_OK(alloc != nullptr, "Failed to get the allocator instance.");
    return alloc;
}

} // namespace base
} // namespace container