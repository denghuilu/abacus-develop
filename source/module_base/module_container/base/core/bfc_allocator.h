#ifndef BASE_CORE_BFC_ALLOCATOR_H_
#define BASE_CORE_BFC_ALLOCATOR_H_

#include <set>
#include <mutex>

#include <base/macros/macros.h>
#include <base/core/allocator.h>
#include <base/core/gpu_allocator.h>

namespace container {
namespace base {
/**
 * @brief An allocator that allocates memory on a GPU device.
 *
 * This class provides an implementation of the Allocator interface that allocates memory
 * on a GPU device using CUDA APIs.
 */
class BFCAllocator : public Allocator {
  public:
    struct Options {
        bool allow_growth = true;
        float init_allocation_fraction = 0.6;
        float fragment_fraction = 0.0;
        Options() : allow_growth(true), fragment_fraction(0.0) {}
    };

  private:
    // Singleton pattern: hide the constructor and copy constructor
    BFCAllocator(const Options& options = Options());
    virtual ~BFCAllocator();

  public:
    static Allocator* get_singleton_instance();
    /**
     * @brief Allocate a block of memory with the given size and default alignment on GPU.
     *
     * @param size The size of the memory block to allocate.
     *
     * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
     */
    void* allocate(size_t size) override {return allocate(size, 256);}

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

    size_t get_available_memory() override;

  private:


    struct bin;
    //
    mutable std::mutex mtx_;

    // A chunk_handle is an index into the chunks_ vector in BFCAllocator
    // kInvalidChunkHandle means an invalid chunk index.
    typedef size_t chunk_handle_t;
    static constexpr chunk_handle_t kInvalidChunkHandle = UINT64_MAX;

    typedef int bin_index_t;
    static constexpr int kInvalidBinNum = -1;
    // The following means that the largest bin'd chunk size is 256 << 21 = 512MB.
    static constexpr int kNumBins = 21;

    struct chunk {
        // The size of the chunk in bytes.
        size_t size = 0;
        // The bin index of the chunk.
        bin_index_t bin_index = kInvalidBinNum;
        // We sometimes give chunks that are larger than needed to reduce
        // fragmentation.  requested_size keeps track of what the client
        // actually wanted so we can understand whether our splitting
        // strategy is efficient.
        size_t requested_size = 0;
        // allocation_id is set to -1 when the chunk is not in use. It is assigned a
        // value greater than zero before the chunk is returned from
        // AllocateRaw, and this value is unique among values assigned by
        // the parent allocator.
        int64_t allocation_id = -1;
        // pointer to granted subbuffer.
        void* ptr = nullptr;  
        chunk_handle_t next_chunk_handle = kInvalidChunkHandle;
        // The handle of the previous chunk in the bin.
        chunk_handle_t prev_chunk_handle = kInvalidChunkHandle;
        // The freed count of the chunk.
        uint64_t freed_at_count = 0;
        // Whether the chunk is allocated.
        bool is_allocated() const { return allocation_id > 0; }
    };

    struct bin {
        // The size of the chunks in this bin.
        size_t bin_size = 0;
        // The number of chunks in this bin.
        size_t num_chunks = 0;
        // The handle of the first chunk in the bin.
        chunk_handle_t first_chunk_handle = kInvalidChunkHandle;
        // The handle of the last chunk in the bin.
        chunk_handle_t last_chunk_handle = kInvalidChunkHandle;

        class chunk_comparator {
          public:
            explicit chunk_comparator(BFCAllocator* allocator) : allocator_(allocator) {}
            // Sort first by size and then use pointer address as a tie breaker.
            bool operator()(const chunk_handle_t ha,
                            const chunk_handle_t hb) const {
                const chunk* a = allocator_->chunk_from_handle(ha);
                const chunk* b = allocator_->chunk_from_handle(hb);
                if (a->size != b->size) {
                    return a->size < b->size;
                }
                return a->ptr < b->ptr;
            }

          private:
            BFCAllocator* allocator_;  // The parent allocator
        };

        using free_chunk_set_t = std::set<chunk_handle_t, chunk_comparator>;

        free_chunk_set_t free_chunks;
        bin(BFCAllocator* allocator, size_t bs)
            : bin_size(bs), free_chunks(chunk_comparator(allocator)) {}
    };

    static constexpr size_t kMinAllocationBits = 8;
    static constexpr size_t kMinAllocationSize = 1 << kMinAllocationBits;

    // BFCAllocator allocates memory into a collection of disjoint
    // AllocationRegions.  Each AllocationRegion corresponds to one call to
    // SubAllocator::Alloc().  (Actually, if a subsequent call to
    // SubAllocator::Alloc() returns another region immediately adjacent to the
    // last, it will be used to extend the first AllocationRegion, not create a
    // separate one.)
    //
    // An AllocationRegion contains one or more Chunks, covering all of its
    // memory.  Its primary job is to map pointers to ChunkHandles.
    //
    // This class is thread-compatible.
    class allocation_region {
      public:
        allocation_region(void* ptr, size_t memory_size)
        : ptr_(ptr),
          memory_size_(memory_size),
          end_ptr_(
              static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)) 
        {
            // DCHECK_EQ(0, memory_size % kMinAllocationSize);
            // TODO: Implement a check macro for container tensors.
            const size_t n_handles =
                (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
            handles_.resize(n_handles, kInvalidChunkHandle);
        }

        allocation_region() = default;
        allocation_region(allocation_region&& other) {this->swap(&other);}
        allocation_region& operator=(allocation_region&& other) {
            this->swap(&other);
            return *this;
        }

        void* ptr() const { return ptr_; }
        void* end_ptr() const { return end_ptr_; }
        size_t memory_size() const { return memory_size_; }

        void extend(size_t new_memory_size) {
            memory_size_ += new_memory_size;
            // DCHECK_EQ(0, memory_size_ % kMinAllocationSize);

            end_ptr_ = static_cast<void*>(static_cast<char*>(ptr_) + memory_size_);
            const int n_handles =
                (memory_size_ + kMinAllocationSize - 1) / kMinAllocationSize;
            handles_.resize(n_handles, kInvalidChunkHandle);

        }

        chunk_handle_t handle_for_ptr(const void* ptr) const {
            return handles_[index_for_handle(ptr)];
        }

        void set_handle_for_ptr(const void* ptr, chunk_handle_t handle) {
            handles_[index_for_handle(ptr)] = handle;
        }

        void erase (const void* ptr) {
            set_handle_for_ptr(ptr, kInvalidChunkHandle);
        }

      private:
        void swap(allocation_region* other) {
            std::swap(ptr_, other->ptr_);
            std::swap(memory_size_, other->memory_size_);
            std::swap(end_ptr_, other->end_ptr_);
            std::swap(handles_, other->handles_);
        }

        size_t index_for_handle(const void* ptr) const {
            const size_t offset = reinterpret_cast<std::uintptr_t>(ptr) - reinterpret_cast<std::uintptr_t>(ptr_);
            // Do the checks
            return static_cast<size_t>(offset >> kMinAllocationBits);
        }

        // The pointer to the start of the memory region.
        void* ptr_;
        // The size of the memory region.
        size_t memory_size_;
        // The pointer to the end of the memory region.
        void* end_ptr_;
        // The vector of chunk handles.
        std::vector<chunk_handle_t> handles_;

        DISALLOW_COPY_AND_ASSIGN(allocation_region);
    };
    
    // RegionManager aggregates one or more "AllocationRegions" and provides
    // a layer of indirection from pointers to the underlying ChunkHandle,
    // allowing allocation across multiple discontiguous memory regions.
    //
    // This class is thread-compatible.

    class region_manager {
      public:
        region_manager() {}
        ~region_manager() {}

        void add_allocation_region(void* ptr, size_t memory_size) {
            auto entry =
                std::upper_bound(regions_.begin(), regions_.end(), ptr, &comparator);
            regions_.insert(entry, allocation_region(ptr, memory_size));
        }

        // Adds an alloation region for the given ptr and size, potentially
        // extending a region if ptr matches the end_ptr of an existing region.
        // If a region is extended, returns a pointer to the extended region so that
        // the BFC allocator can reason about chunkification.
        allocation_region* add_allocation_region_or_extend(
            void* ptr, 
            size_t memory_size) 
        {
            auto entry =
                std::upper_bound(regions_.begin(), regions_.end(), ptr, &comparator);
            if (entry != regions_.begin()) {
                auto prev = entry - 1;
                prev->extend(memory_size);
                return &(*prev);
            }
            regions_.insert(entry, allocation_region(ptr, memory_size));
            return nullptr;
        }

        std::vector<allocation_region>::iterator remove_allocation_region(
            std::vector<allocation_region>::iterator it) {
            return regions_.erase(it);
        }

        chunk_handle_t handle_for_ptr(const void* ptr) const {
            return index_for_region(ptr)->handle_for_ptr(ptr);
        }

        void set_handle_for_ptr(const void* ptr, chunk_handle_t handle) {
            mutable_index_for_region(ptr)->set_handle_for_ptr(ptr, handle);
        }

        void erase(const void* ptr) {
            mutable_index_for_region(ptr)->erase(ptr);
        }

        const std::vector<allocation_region>& regions() const { return regions_; }

      private:
        static bool comparator(const void* ptr, const allocation_region& other) {
            return ptr < other.end_ptr();
        }

        const allocation_region* index_for_region(const void* ptr) const {
            auto entry =
                std::upper_bound(regions_.begin(), regions_.end(), ptr, &comparator);
            if (entry != regions_.end()) {
                return &(*entry);
            }
            // LOG(FATAL) << "Could not find Region for " << p;
            return nullptr;
        }

        allocation_region* mutable_index_for_region(const void* ptr) {
            return const_cast<allocation_region*>(index_for_region(ptr));
        }

        std::vector<allocation_region> regions_;
    };

    // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
    static size_t rounded_bytes(size_t bytes);

    // Returns floor(log2(n)).
    inline int log2_floor_non_zero(uint64_t n) {
        int r = 0;
        while (n > 0) {
            r++;
            n >>= 1;
        }
        return r - 1;
    }

    size_t bin_index_to_size(bin_index_t index) {
        return static_cast<size_t>(256) << index;
    }

    // This is actually a reference of the bin.
    bin* bin_from_index(bin_index_t index) {
        return reinterpret_cast<bin*>(&(bins_space_[index * sizeof(bin)]));
    }

    bin_index_t bin_index_for_size(size_t bytes) {
        uint64_t v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
        int b = std::min(kNumBins - 1, log2_floor_non_zero(v));
        return b;
    }

    bin* bin_for_size(size_t bytes) { return bin_from_index(bin_index_for_size(bytes)); }

    chunk* chunk_from_handle(chunk_handle_t h);
    const chunk* chunk_from_handle(chunk_handle_t h) const;

    void merge(chunk_handle_t handle_1, chunk_handle_t handle_2);

    void split_chunk(chunk_handle_t handle, size_t size);

    bool extend(size_t alignment, size_t rounded_bytes);

    chunk_handle_t allocate_chunk();

    void deallocate_chunk(chunk_handle_t handle);

    void insert_free_chunk_into_bin(chunk_handle_t handle);

    void remove_free_chunk_from_bin(chunk_handle_t handle);

    void remove_free_chunk_iter_from_bin(bin::free_chunk_set_t* free_chunks, bin::free_chunk_set_t::iterator iter);

    chunk_handle_t try_to_coalesce_with_nbor_chunk(chunk_handle_t handle, bool ignore_freed_at);

    void* find_chunk_ptr(bin_index_t bin_index, size_t rounded_bytes, size_t size, uint64_t freed_before);

    void mark_free(chunk_handle_t handle);
    // record the allocated memory;
    region_manager region_manager_;

    char bins_space_[sizeof(bin) * kNumBins];
    // Params for BFCAllocator 
    Options options_ = {};
    // Sub allocator for BFCAllocator 
    Allocator* sub_alloc_ = nullptr;
    // The size of the current region allocation.
    size_t curr_region_allocation_bytes_ = 0;
    // Record the memory allocate operations
    AllocatorStats stats_ = {};
    // Allocate the requested amount of memory.
    size_t memory_limit_ = 0;
    // empty chunk handle
    std::vector<chunk> chunks_ = {};
    chunk_handle_t free_chunks_list_ = kInvalidChunkHandle;
    // mark the allocation counter 
    int64_t next_allocation_id_ = 0;

    DISALLOW_COPY_AND_ASSIGN(BFCAllocator);
};

} // namespace base
} // namespace container

#endif // BASE_CORE_BFC_ALLOCATOR_H_
