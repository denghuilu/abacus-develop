#ifndef CONTAINER_TENSOR_BUFFER_H_
#define CONTAINER_TENSOR_BUFFER_H_

#include <cstddef>

#include "allocator.h"
#include "tensor_types.h"

namespace container {

/**
 * @brief Interface to access the raw ref-counted data buffer.
 */
 class TensorBuffer {
   public:
    /**
     * @brief Construct a new TensorBuffer object.
     *
     * @param alloc Pointer to the allocator to use for memory allocation.
     * @param data_ptr Pointer to the underlying data buffer.
     */
    TensorBuffer(Allocator* alloc, void* data_ptr);

    /**
      * @brief Construct a new TensorBuffer object.
      *
      * This is a reference TensorBuffer, does not owns memory itself.
      *
      * @param data_ptr Pointer to the given data buffer.
      */
    explicit TensorBuffer(void* data_ptr);
    
    /**
     * @brief Destroy the TensorBuffer object.
     */
    ~TensorBuffer();

    /**
     * @brief Get the raw data pointer.
     *
     * @return void* Pointer to the underlying data buffer.
     */
    void* data() const;

    /**
     * @brief resize the tensor buffer
     */
    void resize(size_t size);

    /**
     * @brief Get the size of the buffer.
     *
     * @return size_t The size of the buffer in bytes.
     */
    size_t GetAllocatedBytes() const;

    /**
     * @brief Get the root TensorBuffer object.
     *
     * If this TensorBuffer is a sub-buffer of another TensorBuffer, returns that
     * TensorBuffer. Otherwise, returns this.
     *
     * @return TensorBuffer* Pointer to the root TensorBuffer object.
     */
    TensorBuffer* root_buffer();

    /**
     * @brief Get the Allocator object used in this class.
     *
     * @return Allocator* Pointer to the Allocator object.
     */
    Allocator* allocator() const;

    /**
     * @brief Reinterpret the buffer as an array of type T.
     *
     * @tparam T The type to reinterpret the buffer as.
     * @return T* Pointer to the underlying buffer reinterpreted as type T.
     */
    template <typename T>
    T* base() const {
        return reinterpret_cast<T*>(data());
    }

    /**
     * @brief Check whether this TensorBuffer owns the underlying memory.
     *
     * @return true If the TensorBuffer owns the underlying memory.
     * @return false If the TensorBuffer does not own the underlying memory.
     */
    virtual bool OwnsMemory() const;

    /**
     * @brief Get the type of device used by the TensorBuffer.
     *
     * @return MemoryType The type of memory used by the TensorBuffer.
     */
    DeviceType GetDeviceType() const;

    /**
     * @brief Assignment operator overload for the TensorBuffer class.
     *
     * This operator is used to assign the values of another TensorBuffer object to the current object.
     * It performs a deep copy of the data from the source TensorBuffer to the destination TensorBuffer.
     *
     * @param other The source TensorBuffer object whose values will be assigned.
     * @return A reference to the current TensorBuffer object after the assignment.
     */
    TensorBuffer& operator=(const TensorBuffer& other);


  private:
    void *data_;       ///< Pointer to the underlying data buffer.
    Allocator* alloc_; ///< Pointer to the allocator used for memory allocation.
    bool owns_memory;  ///< Bool to indicate whether this tensor owns it's memory.
};

}  // namespace container

#endif  // CONTAINER_TENSOR_BUFFER_H_
