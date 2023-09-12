#include <base/core/gpu_allocator.h>

namespace container {
namespace base {

// Get the singleton instance of the GPUAllocator.
Allocator* GPUAllocator::get_singleton_instance() {
    static GPUAllocator instance_{};
    return &instance_;
}

// Allocate a block of memory with the given size and default alignment on GPU.
void* GPUAllocator::allocate(size_t size) {
    void* ptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    if (result != cudaSuccess) {
        return nullptr;
    }
    this->allocated_size_ = size;
    return ptr;
}

// Allocate a block of CPU memory with the given size and alignment.
void* GPUAllocator::allocate(size_t size, size_t alignment) {
    void* ptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    if (result != cudaSuccess) {
        return nullptr;
    }
    this->allocated_size_ = size;
    return ptr;
}

// Free a block of CPU memory that was previously allocated by this allocator.
void GPUAllocator::free(void* ptr) {
    cudaFree(ptr);
    this->allocated_size_ = 0;
}

// Get the type of device used by the TensorBuffer.
DeviceType GPUAllocator::GetDeviceType() {
    return DeviceType::GpuDevice;
}

AllocatorType GPUAllocator::GetAllocatorType() {
    return AllocatorType::GPU;
}

size_t GPUAllocator::get_available_memory() {
    size_t free_memory = 0, max_memory = 0;
    cudaError_t result = cudaMemGetInfo(&free_memory, &max_memory);
    REQUIRES_OK(result == cudaSuccess, "Failed to get the available GPU memory.");
    return free_memory;
}

} // namespace base
} // namespace container
