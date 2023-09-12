#include <base/core/cpu_allocator.h>

namespace container {
namespace base {

// Get the singleton instance of the CPUAllocator.
Allocator* CPUAllocator::get_singleton_instance() {
    static CPUAllocator instance_{};
    return &instance_;
}

// Allocate a block of CPU memory with the given size and default alignment.
void* CPUAllocator::allocate(size_t size) {
    this->allocated_size_ = size;
    return ::operator new(size);
}

// Allocate a block of CPU memory with the given size and alignment.
void* CPUAllocator::allocate(size_t size, size_t alignment) {
    this->allocated_size_ = size;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
    return ptr;
}

// Free a block of CPU memory that was previously allocated by this allocator.
void CPUAllocator::free(void* ptr) {
    this->allocated_size_ = 0;
    ::operator delete(ptr);
}

//  Get the type of device used by the TensorBuffer.
DeviceType CPUAllocator::GetDeviceType() {
    return DeviceType::CpuDevice;
}

AllocatorType CPUAllocator::GetAllocatorType() {
    return AllocatorType::CPU;
}

size_t CPUAllocator::get_available_memory() {
    // TODO: Insert some warnings message here.
    return 0;
}

} // namespace base
} // namespace container
