#include "tensor_buffer.h"
#include "cpu_allocator.h"

#if defined(__CUDA) || defined(__ROCM)
#include "gpu_allocator.h"
#endif

namespace container {

// Construct a new TensorBuffer object.
TensorBuffer::TensorBuffer(Allocator* alloc, void* data_ptr) : data_(data_ptr), alloc_(alloc), owns_memory(true) {}

// Construct a new TensorBuffer object.
// Note, this is a reference TensorBuffer, does not owns memory itself.
TensorBuffer::TensorBuffer(void* data_ptr) : data_(data_ptr), alloc_(), owns_memory(false) {}

// Move constructor.
TensorBuffer::TensorBuffer(TensorBuffer&& other) noexcept
        : data_(other.data_), 
          alloc_(other.alloc_), 
          owns_memory(other.owns_memory) 
{
    // Reset the other TensorBuffer.
    other.data_ = nullptr;
    other.owns_memory = false;
}

// Destroy the TensorBuffer object.
TensorBuffer::~TensorBuffer() {
    if (this->OwnsMemory() && data_ != nullptr) {
        alloc_->free(data_);
    }
}

// Get the raw data pointer.
void* TensorBuffer::data() const { return data_; }

// Get the total number of bytes allocated for the buffer.
// This method returns the total number of bytes allocated for the buffer by the allocator
// associated with the TensorBuffer. If the buffer is not yet allocated, the function returns 0.
size_t TensorBuffer::GetAllocatedBytes() const {
    return alloc_ == nullptr ?
           0 :
           alloc_->AllocatedSize(data());
}

// Get the root TensorBuffer object.
// If this TensorBuffer is a sub-buffer of another TensorBuffer, returns that
// TensorBuffer. Otherwise, returns this.
TensorBuffer* TensorBuffer::root_buffer() { return this; } // Implementation goes here.

// Get the Allocator object used in this class.
Allocator* TensorBuffer::allocator() const {
    return alloc_;
}

// Check whether this TensorBuffer owns the underlying memory.
bool TensorBuffer::OwnsMemory() const { return this->owns_memory; }

// Get the type of device used by the TensorBuffer.
DeviceType TensorBuffer::GetDeviceType() const {
    if (alloc_ != nullptr) {
        return alloc_->GetDeviceType();
    }
    return DeviceType::UnKnown;
}

void TensorBuffer::resize(size_t size) {
    // Allocate a new buffer.
    void* new_data = this->alloc_->allocate(size);

    // Free the old buffer.
    if (this->OwnsMemory()) {
        this->alloc_->free(data_);
    }

    // Update the internal state.
    this->data_ = new_data;
    this->owns_memory = true;
}

TensorBuffer& TensorBuffer::operator=(const TensorBuffer& other) {
    if (this->OwnsMemory()) {
        this->alloc_->free(data_);
    }

    delete this->alloc_;
    if (other.GetDeviceType() == DeviceType::CpuDevice) {
        this->alloc_ = new CPUAllocator();
    }
    #if defined(__CUDA) || defined(__ROCM)
    else if (other.GetDeviceType() == DeviceType::GpuDevice) {
        this->alloc_ = new GPUAllocator();
    }
    #endif // __CUDA || __ROCM


    this->data_ = this->alloc_->allocate(other.GetAllocatedBytes());
    this->owns_memory = true;
    return *this;
}

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
    if (this->OwnsMemory()) {
        this->alloc_->free(data_);
    }
    delete this->alloc_;
    this->alloc_ = other.alloc_;
    this->data_ = other.data_;
    this->owns_memory = other.owns_memory;

    // Reset the other TensorBuffer.
    other.data_ = nullptr;
    other.owns_memory = false;
    return *this;
}

}  // namespace container
