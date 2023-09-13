#include <ATen/core/tensor.h>
#include <base/macros/macros.h>
#include <base/core/bfc_allocator.h>

namespace container {
namespace base {

// Mark as a constant value
constexpr BFCAllocator::chunk_handle_t BFCAllocator::kInvalidChunkHandle;

BFCAllocator::BFCAllocator(const Options& options) 
    : options_(options), 
    next_allocation_id_(1),
    free_chunks_list_(kInvalidChunkHandle)
{   
    sub_alloc_ = GPUAllocator::get_singleton_instance();
    memory_limit_ = sub_alloc_->get_available_memory();

    auto init_memory = static_cast<size_t>(memory_limit_ * options_.init_allocation_fraction);
    if (options_.allow_growth) {
        // Allow growth, so start with a small region and grow as needed.
        // Note the minimum region size is 2MiB.
        curr_region_allocation_bytes_ = rounded_bytes(std::max(init_memory, size_t{2 << 20}));
    }
    else {
        curr_region_allocation_bytes_ = rounded_bytes(memory_limit_);
    }
    stats_.pool_bytes = 0;
    stats_.peak_pool_bytes = 0;
    stats_.bytes_limit = static_cast<int64_t>(memory_limit_);

    // Create Bins 
    for (bin_index_t b = 0; b < kNumBins; b++) {
        size_t bin_size = bin_index_to_size(b);
        new (bin_from_index(b)) bin(this, bin_size);
        // TODO: Check the following code
        // Error handling is too weak now!
        REQUIRES_OK(bin_for_size(bin_size) == bin_from_index(b));
        REQUIRES_OK(bin_for_size(bin_size + 255) == bin_from_index(b));
        REQUIRES_OK(bin_for_size(bin_size * 2 - 1) == bin_from_index(b));
        if (b + 1 < kNumBins) {
            REQUIRES_OK(bin_for_size(bin_size * 2) != bin_from_index(b));
        }
    }
}
// noexcept: Indicates that the function does not throw any exceptions.
// https://en.cppreference.com/w/cpp/language/noexcept
BFCAllocator::~BFCAllocator() noexcept {
    // Return memory back.
    for (const auto& region : region_manager_.regions()) {
        sub_alloc_->free(region.ptr());
    }

    for (bin_index_t b = 0; b < kNumBins; b++) {
        bin_from_index(b)->~bin();
    }
}

Allocator* BFCAllocator::get_singleton_instance()
{
    // guranteed to be freed when the program exits
    static BFCAllocator instance_{};
    return &instance_;
}


AllocatorType BFCAllocator::GetAllocatorType() {
    return AllocatorType::BFC;
}

BFCAllocator::chunk* BFCAllocator::chunk_from_handle(chunk_handle_t h) {
    REQUIRES_OK(h >= 0);
    REQUIRES_OK(h <  static_cast<int>(chunks_.size()));
    return &chunks_[h];
}

const BFCAllocator::chunk* BFCAllocator::chunk_from_handle(chunk_handle_t h) const {
    REQUIRES_OK(h >= 0);
    REQUIRES_OK(h <  static_cast<int>(chunks_.size()));
    return &chunks_[h];
}

BFCAllocator::chunk_handle_t BFCAllocator::allocate_chunk() {
    if (free_chunks_list_ != kInvalidChunkHandle) {
        chunk_handle_t handle = free_chunks_list_;
        chunk* c = chunk_from_handle(handle);
        free_chunks_list_ = c->next_chunk_handle;
        return handle;
    }
    else {
        chunk_handle_t handle = chunks_.size();
        chunks_.resize(handle + 1);
        return handle;
    }
}

void BFCAllocator::deallocate_chunk(chunk_handle_t handle) {
    chunk* c = chunk_from_handle(handle);
    c->allocation_id = -1;
    c->bin_index = kInvalidBinNum;
    c->next_chunk_handle = free_chunks_list_;
    free_chunks_list_ = handle;
}

void BFCAllocator::insert_free_chunk_into_bin(chunk_handle_t handle) {
    chunk* c = chunk_from_handle(handle);
    REQUIRES_OK(!c->is_allocated() && c->bin_index == kInvalidBinNum);
    bin_index_t bin_index = bin_index_for_size(c->size);
    bin* new_bin = bin_for_size(c->size);
    c->bin_index = bin_index;
    new_bin->free_chunks.insert(handle);
}

void BFCAllocator::remove_free_chunk_from_bin(chunk_handle_t handle) {
    chunk* c = chunk_from_handle(handle);
    REQUIRES_OK(!c->is_allocated() && c->bin_index != kInvalidBinNum);
    REQUIRES_OK(bin_from_index(c->bin_index)->free_chunks.erase(handle) > 0);
    c->bin_index = kInvalidBinNum;
}

void BFCAllocator::remove_free_chunk_iter_from_bin(
    bin::free_chunk_set_t* free_chunks, 
    bin::free_chunk_set_t::iterator iter)
{
    chunk_handle_t handle = *iter;
    chunk* c = chunk_from_handle(handle);
    REQUIRES_OK(!c->is_allocated() && c->bin_index != kInvalidBinNum);
    free_chunks->erase(iter);
    c->bin_index = kInvalidBinNum;
}

void BFCAllocator::merge(chunk_handle_t handle_1, chunk_handle_t handle_2) {
    chunk* c1 = chunk_from_handle(handle_1);
    chunk* c2 = chunk_from_handle(handle_2);
    // We can only merge chunks that are not in use.
    REQUIRES_OK(!c1->is_allocated() && !c2->is_allocated());
    // c1's prev doesn't change, still points to the same ptr, and is
    // still not in use.

    // Fix up neighbor pointers
    //
    // c1 <-> c2 <-> c3 should become
    // c1 <-> c3
    chunk_handle_t h3 = c2->next_chunk_handle;
    c1->next_chunk_handle = h3;
    REQUIRES_OK(c2->prev_chunk_handle == handle_1);
    if (h3 != kInvalidChunkHandle) {
        chunk* c3 = chunk_from_handle(h3);
        c3->prev_chunk_handle = handle_1;
    }
    // Set the new size
    c1->size += c2->size;
    // Pick latest free time
    c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

    // Deallocate c2
    deallocate_chunk(handle_2);
}

void BFCAllocator::split_chunk(chunk_handle_t handle, size_t size) {
    // Allocate the new chunk before we do any ChunkFromHandle
    chunk_handle_t h_new = allocate_chunk(); 

    chunk* c = chunk_from_handle(handle);
    REQUIRES_OK(!c->is_allocated() && c->bin_index == kInvalidBinNum);

    // Create a new chunk starting size after c
    chunk* new_c = chunk_from_handle(h_new);
    // What does static_cast exactly do?
    new_c->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + size);
    region_manager_.set_handle_for_ptr(new_c->ptr, h_new);

    // Set the new sizes of the chunks
    new_c->size = c->size - size;
    c->size = size;

    // The new chunk is not in use!
    new_c->allocation_id = -1;

    // It inherits the freed time.
    new_c->freed_at_count = c->freed_at_count;

    // Maintain the pointers.
    // c <-> c_neighbor becomes
    // c <-> new_chunk <-> c_neighbor
    chunk_handle_t h_nbor = c->next_chunk_handle;
    new_c->prev_chunk_handle = handle;
    new_c->next_chunk_handle = h_nbor;
    c->next_chunk_handle = h_new;
    if (h_nbor != kInvalidChunkHandle) {
        chunk* c_nbor = chunk_from_handle(h_nbor);
        c_nbor->prev_chunk_handle = h_new;
    }

    // Add the newly free chunk to the free bin.
    insert_free_chunk_into_bin(h_new);
}

BFCAllocator::chunk_handle_t BFCAllocator::try_to_coalesce_with_nbor_chunk(
    chunk_handle_t handle, 
    bool ignore_freed_at) 
{
    chunk* c = chunk_from_handle(handle);
    if ((!ignore_freed_at) && c->freed_at_count > 0) {
        return handle;
    }

    chunk_handle_t coalesced_chunk = handle;

    // If the next chunk is free, merge it into c and delete it.
    if (c->next_chunk_handle != kInvalidChunkHandle && !chunk_from_handle(c->next_chunk_handle)->is_allocated()) {
        chunk* next_c = chunk_from_handle(c->next_chunk_handle);
        if (next_c->freed_at_count == 0 || ignore_freed_at) {
            remove_free_chunk_from_bin(c->next_chunk_handle);
            merge(handle, c->next_chunk_handle);
        }
    }

    // If the previous chunk is free, merge c into it and delete c.
    if (c->prev_chunk_handle != kInvalidChunkHandle && !chunk_from_handle(c->prev_chunk_handle)->is_allocated()) {
        chunk* prev_c = chunk_from_handle(c->prev_chunk_handle);
        if (prev_c->freed_at_count == 0 || ignore_freed_at) {
            coalesced_chunk = c->prev_chunk_handle;
            remove_free_chunk_from_bin(c->prev_chunk_handle);
            merge(c->prev_chunk_handle, handle);
        }
    }

    return coalesced_chunk;
}

// static function
size_t BFCAllocator::rounded_bytes(size_t size) {
    size_t rounded_bytes = (kMinAllocationSize *
            ((size + kMinAllocationSize - 1) / kMinAllocationSize));

    REQUIRES_OK(size_t(0) == rounded_bytes % kMinAllocationSize);
    return rounded_bytes;
}

bool BFCAllocator::extend(size_t alignment, size_t rounded_bytes) {
    size_t available_bytes = memory_limit_ - stats_.pool_bytes;
    // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
    size_t rounded_available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

    if (rounded_available_bytes < rounded_bytes) {
        return false;
    }

    // If curr_region_allocation_bytes_ is not enough to satisfy the
    // allocation, keep multiplying by a power of two until that is
    // sufficient.
    bool increased_allocation = false;
    while (rounded_bytes > curr_region_allocation_bytes_) {
        curr_region_allocation_bytes_ *= 2;
        increased_allocation = true;
    }

    // Try allocating a new memory
    size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
    void* mem_addr = sub_alloc_->allocate(bytes, alignment);

    // If the allocation fails, fail immediately.
    if (mem_addr == nullptr) {
        return false;
    }

    if (!increased_allocation) {
        curr_region_allocation_bytes_ *= 2;
    }

    stats_.pool_bytes += bytes;
    stats_.peak_pool_bytes = std::max(stats_.peak_pool_bytes, stats_.pool_bytes);

    region_manager_.add_allocation_region(mem_addr, bytes);

    // Create one large chunk for the whole memory space that will
    // be chunked later.
    chunk_handle_t handle = allocate_chunk();
    chunk* c = chunk_from_handle(handle);
    c->ptr = mem_addr;
    c->size = bytes;
    c->allocation_id = -1;
    c->prev_chunk_handle = kInvalidChunkHandle;
    c->next_chunk_handle = kInvalidChunkHandle;
    c->freed_at_count = 0;

    // Be careful here: using region_manager to handle the memory space.
    region_manager_.set_handle_for_ptr(c->ptr, handle);
    // Maybe merge adjacent chunks and insert the chunk into the right bin.
    
    insert_free_chunk_into_bin(try_to_coalesce_with_nbor_chunk(handle, /*ignore_freed_at=*/false));

    return true;
}

void* BFCAllocator::find_chunk_ptr(
    bin_index_t bin_index, 
    size_t rounded_bytes, 
    size_t size, 
    uint64_t freed_before) 
{
    while (bin_index < kNumBins) {
        bin* b = bin_from_index(bin_index);
        for (auto chunk_iter = b->free_chunks.begin(); chunk_iter != b->free_chunks.end(); chunk_iter++) {
            const chunk_handle_t handle = *chunk_iter;
            chunk* c = chunk_from_handle(handle);
            REQUIRES_OK(!c->is_allocated());
            if (freed_before > 0 && c->freed_at_count > freed_before) {
                continue;
            }
            if (c->size >= rounded_bytes) {
                // We found an existing chunk that fits us that wasn't in use, so remove
                // it from the free bin structure prior to using.
                remove_free_chunk_iter_from_bin(&b->free_chunks, chunk_iter);

                // If we can break the size of the chunk into two reasonably large
                // pieces, do don't waste more than max_internal_fragmentation_bytes on
                // padding. If this threshold is not set by the user, then use 128MB as
                // the default.
                const int64_t max_internal_fragmentation_bytes =
                      options_.fragment_fraction > 0.0
                    ? options_.fragment_fraction * memory_limit_
                    : 128 << 20;
                
                if (c->size >= rounded_bytes * 2 || 
                    static_cast<int64_t>(c->size) - rounded_bytes >= max_internal_fragmentation_bytes)
                {
                    split_chunk(handle, rounded_bytes);
                    c = chunk_from_handle(handle); // Update chunk pointer in case it moved
                }

                // The requested size of the returned chunk is what the user
                // has allocated.
                c->requested_size = size;
                // Assign a unique id and increment the id counter, marking the
                // chunk as being in use.
                c->allocation_id = next_allocation_id_++;

                // Update stats
                stats_.num_allocs++;
                stats_.bytes_in_use += c->size;
                stats_.peak_bytes_in_use =
                    std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
                stats_.largest_alloc_size =
                    std::max<std::size_t>(stats_.largest_alloc_size, c->size);

                return c->ptr;
            }
        }
        bin_index++;
    }
    return nullptr;
}

void* BFCAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        // TODO: Log some warning message here
        return nullptr;
    }
    // First, always allocate memory of at least kMinAllocationSize
    // bytes, and always allocate multiples of kMinAllocationSize bytes
    // so all memory addresses are nicely byte aligned.
    size_t rounded_bytes_ = rounded_bytes(size);
    bin_index_t bin_index = bin_index_for_size(rounded_bytes_);

    // std::lock_guard is a class template that implements the RAII for mutex
    // https://en.cppreference.com/w/cpp/thread/lock_guard
    // https://en.cppreference.com/w/cpp/thread/mutex
    std::lock_guard<std::mutex> lock(mtx_);

    void *mem_addr = find_chunk_ptr(bin_index, rounded_bytes_, size, 0);

    if (mem_addr != nullptr) {
        return mem_addr;
    }

    // Try to extend the memory pool.
    if (extend(alignment, rounded_bytes_)) {
        mem_addr = find_chunk_ptr(bin_index, rounded_bytes_, size, 0);
        if (mem_addr != nullptr) {
            return mem_addr;
        }
    }

    // TODO: implement a deallocate_free_regions func to handle the gabarge collection
    // Reaching this point means that no chunks can satisfy the request. Also,
    // the unallocated bytes cannot satisfy the request. Before giving up, let's
    // try deallocating free regions so that suballocator can combine them with
    // the unallocated bytes and form a larger region.
    // if (deallocate_free_regions(rounded_bytes_) && extend(alignment, rounded_bytes_)) {
    //     mem_addr = find_chunk_ptr(bin_index, rounded_bytes_, size, 0);
    //     if (mem_addr != nullptr) {
    //         return mem_addr;
    //     }
    // }

    // fail to allocate memory
    REQUIRES_OK(mem_addr != nullptr);
    return mem_addr;
}

void BFCAllocator::mark_free(chunk_handle_t handle) {
    chunk* c = chunk_from_handle(handle);
    REQUIRES_OK(c->is_allocated() && c->bin_index == kInvalidBinNum);

    // Mark the chunk as no longer in use.
    c->allocation_id = -1;

    // Updates the stats.
    stats_.bytes_in_use -= c->size;
}

void BFCAllocator::free(void* mem_addr) {
    if (mem_addr == nullptr) {
        // TODO: Log some warning message here
        return;
    }
    std::lock_guard<std::mutex> lock(mtx_);

    // Find the chunk from the memory address
    chunk_handle_t handle = region_manager_.handle_for_ptr(mem_addr);
    REQUIRES_OK(handle != kInvalidChunkHandle);

    mark_free(handle);

    // Insert the chunk into the free bin.
    insert_free_chunk_into_bin(try_to_coalesce_with_nbor_chunk(handle, /*ignore_freed_at=*/false));
}

DeviceType BFCAllocator::GetDeviceType() {
    return sub_alloc_->GetDeviceType();
}

/**
 * @brief Gets the available memory size.
 *
 * This function returns the size of available memory as a size_t value.
 *
 * @return The size of available memory.
 */
size_t BFCAllocator::get_available_memory() {
    return sub_alloc_->get_available_memory();
}

} // namespace base
} // namespace container