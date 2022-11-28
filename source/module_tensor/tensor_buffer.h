/// Interface to access the raw ref-counted data buffer.
class TensorBuffer : public core::RefCounted {
public:
    explicit TensorBuffer(void* data_ptr) : data_(data_ptr) {}
    ~TensorBuffer() override {}

    /// \brief data() points to a memory region of size() bytes.
    ///
    /// NOTE(mrry): The `data()` method is not virtual for performance reasons.
    /// It can be called multiple times when the contents of a `Tensor` are
    /// accessed, and so making it non-virtual allows the body to be inlined.
    void* data() const { return data_; }

    /// \brief Size (in bytes) of the buffer.
    virtual size_t size() const = 0;

    /// \brief If this TensorBuffer is sub-buffer of another TensorBuffer,
    /// returns that TensorBuffer. Otherwise, returns this.
    virtual TensorBuffer* root_buffer() = 0;

    /// \brief Fills metadata about the allocation into the proto.
    virtual void FillAllocationDescription(
            AllocationDescription* proto) const = 0;

    virtual bool GetAllocatedBytes(size_t* out_bytes) const;

    /// \brief Helper method to reinterpret the buffer as an array of `T`.
    template <typename T>
    T* base() const {
        return reinterpret_cast<T*>(data());
    }

    /// \brief Whether this TensorBuffer owns the underlying memory.
    virtual bool OwnsMemory() const { return true; }

    /// \brief The type of the underlying memory.
    virtual AllocatorMemoryType GetMemoryType() const {
        return AllocatorMemoryType::kUnknown;
    }

private:
    void* const data_;
};