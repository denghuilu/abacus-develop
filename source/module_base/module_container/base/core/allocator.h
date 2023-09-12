#ifndef BASE_CORE_ALLOCATOR_H_
#define BASE_CORE_ALLOCATOR_H_

#include <ATen/core/tensor_types.h>
#include <base/macros/macros.h>

namespace container {

namespace base {

enum AllocatorType {
    CPU,
    GPU,
    BFC,
    UNKNOWN
};

struct AllocatorStats {
  int64_t num_allocs;          // Number of allocations.
  int64_t bytes_in_use;        // Number of bytes in use.
  int64_t peak_bytes_in_use;   // The peak bytes in use.
  int64_t largest_alloc_size;  // The largest single allocation seen.

  int64_t bytes_limit;

  // Stats for reserved memory usage.
  int64_t bytes_reserved;       // Number of bytes reserved.
  int64_t peak_bytes_reserved;  // The peak number of bytes reserved.
  // The upper limit on the number bytes of reservable memory,
  int64_t bytes_reservable_limit;

  int64_t largest_free_block_bytes;  // Largest free block's size in heap.

  // Number of bytes of memory held by the allocator.  This may be higher than
  // bytes_in_use if the allocator holds a pool of memory (e.g. BFCAllocator).
  int64_t pool_bytes;
  int64_t peak_pool_bytes;

  AllocatorStats()
      : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_limit(0),
        bytes_reserved(0),
        peak_bytes_reserved(0),
        bytes_reservable_limit(0),
        largest_free_block_bytes(0),
        pool_bytes(0),
        peak_pool_bytes(0) {}
};

/**
 * @brief An abstract base class for memory allocators.
 *
 * This class defines an interface for memory allocators. Subclasses of this class
 * can provide different implementations of memory allocation/deallocation strategies.
 *
 * All memory allocated by an Allocator must be freed using the same allocator that
 * allocated it.
 */
class Allocator {
  public:
    /**
     * @brief Allocate a block of memory with the given size and default alignment.
     *
     * @param size The size of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Allocate a block of memory with the given size and alignment.
     *
     * @param size The size of the memory block to allocate.
     * @param alignment The alignment of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    virtual void* allocate(size_t size, size_t alignment) = 0;

    /**
     * @brief Free a block of memory that was previously allocated by this allocator.
     *
     * @param ptr A pointer to the memory block to free.
     */
    virtual void free(void* ptr) = 0;

    /**
     * @brief Get the allocated size of a given pointer.
     *
     * @param ptr The pointer to get the allocated size of.
     * @return size_t The size of the allocated block of memory, in bytes.
     */
    virtual size_t AllocatedSize(void* ptr) {
        return allocated_size_;
    }

    /**
     * @brief Get the type of memory used by the TensorBuffer.
     *
     * @return MemoryType The type of memory used by the TensorBuffer.
     */
    virtual DeviceType GetDeviceType() = 0;

    /**
     * @brief Get the type of allocator.
     *
     * This function returns an enum or identifier representing the type of allocator.
     * Subclasses should provide an implementation that specifies the specific
     * allocator type using their own custom enum or identifier.
     *
     * @return AllocatorType The type of allocator.
     */
    virtual AllocatorType GetAllocatorType() = 0;

    virtual size_t get_available_memory() = 0;
    
    /**
     * @brief Retrieves a singleton instance of the Allocator.
     *
     * This function returns a singleton instance of the Allocator based on the specified
     * AllocatorType. If an instance does not exist for the given type, it will create one.
     * Note: This function is thread-compatible. and will return the same instance for the same
     * AllocatorType. It means that the returned instance is shared by all callers.
     *
     * @param type The DeviceType to specify the type of Allocator instance to retrieve.
     * @return A pointer to the singleton instance of the Allocator.
     */
    static Allocator* get_singleton_instance(DeviceType type);

  protected:
    /**
     * @brief The total number of bytes allocated by this allocator.
     */
    size_t allocated_size_ = 0;

    DEFAULT_CONSTRUCTORS(Allocator);
    DISALLOW_COPY_AND_ASSIGN(Allocator);
};

} // namespace base
} // namespace container

#endif // BASE_CORE_ALLOCATOR_H_
