#ifndef BASE_CORE_GPU_ALLOCATOR_H_
#define BASE_CORE_GPU_ALLOCATOR_H_

#include <mutex>
#include <base/core/allocator.h>

namespace container {
namespace base {
/**
 * @brief An allocator that allocates memory on a GPU device.
 *
 * This class provides an implementation of the Allocator interface that allocates memory
 * on a GPU device using CUDA APIs.
 */
class GPUAllocator : public Allocator {
public:
    /**
     * @brief Allocate a block of memory with the given size and default alignment on GPU.
     *
     * @param size The size of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    void* allocate(size_t size) override;

    /**
     * @brief Allocate a block of memory with the given size and alignment on GPU.
     *
     * @param size The size of the memory block to allocate.
     * @param alignment The alignment of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    void* allocate(size_t size, size_t alignment) override;

    /**
     * @brief Free a block of GPU memory that was previously allocated by this allocator.
     *
     * @param ptr A pointer to the memory block to free.
     */
    void free(void* ptr) override;

    /**
     * @brief Get the type of memory used by the TensorBuffer.
     *
     * @return MemoryType The type of memory used by the TensorBuffer.
     */
    DeviceType GetDeviceType() override;

    /**
     * @brief Get the type of allocator.
     *
     * This function returns an enum or identifier representing the type of allocator.
     *
     * @return AllocatorType The type of allocator.
     */
    AllocatorType GetAllocatorType() override;

    /**
     * @brief Get the singleton instance of the Allocator.
     *
     * This static member function returns a pointer to the singleton instance of the Allocator class.
     *
     * @return A pointer to the singleton instance of the Allocator.
     */
    static Allocator* get_singleton_instance();

    /**
     * @brief Gets the available memory size.
     *
     * This function returns the size of available memory as a size_t value.
     *
     * @return The size of available memory.
     */
    size_t get_available_memory() override;
  
  private:

    DEFAULT_CONSTRUCTORS(GPUAllocator);
    DISALLOW_COPY_AND_ASSIGN(GPUAllocator);
};

} // namespace base
} // namespace container

#endif // BASE_CORE_GPU_ALLOCATOR_H_
