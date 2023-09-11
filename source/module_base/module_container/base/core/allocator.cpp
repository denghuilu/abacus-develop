#include <base/core/allocator.h>
#include <base/core/cpu_allocator.h>
#include <base/core/bfc_allocator.h>

#if defined(__CUDA) || defined(__ROCM)
#include <base/core/gpu_allocator.h>
#endif

namespace container {
namespace base {

static base::Allocator* get_default_bfc_allocator() {
    static base::Allocator* bfc_allocator = new base::BFCAllocator(DeviceType::GpuDevice, static_cast<size_t>(10240));
    return bfc_allocator;
}

Allocator* Allocator::GetAllocator(DeviceType device) {
    base::Allocator* allocator = nullptr;
    if (device == DeviceType::CpuDevice) {
        allocator = new base::CPUAllocator();
    }
#if defined(__CUDA) || defined(__ROCM)
    else if (device == DeviceType::GpuDevice) {
        // allocator = new base::GPUAllocator();
        return get_default_bfc_allocator();
    }
#endif // __CUDA || __ROCM
    else {
        std::cerr << "Tensor device type " << device << " does not match requested type." << std::endl;
        exit(EXIT_FAILURE);
    }
    return allocator;   
}

} // namespace base
} // namespace container