#include <base/core/allocator.h>
#include <base/core/bfc_allocator.h>
#include <base/macros/macros.h>

namespace container {
namespace base {

// Mark as a constant value
constexpr BFCAllocator::chunk_handle_t BFCAllocator::kInvalidChunkHandle;

BFCAllocator::BFCAllocator(
    std::unique_ptr<Allocator> sub_alloc, 
    const size_t& total_memory, 
    const Options& options) 
    : options_(options), sub_alloc_(std::move(sub_alloc)) 
{
    if (options_.allow_growth) {
        // Allow growth, so start with a small region and grow as needed.
        // Note the minimum region size is 2MiB.
        curr_region_allocation_bytes_ = rounded_bytes(std::min(total_memory, size_t{2 << 20}));
    }
    else {
        curr_region_allocation_bytes_ = rounded_bytes(total_memory);
    }

    memory_limit_ = total_memory;
    stats_.bytes_limit = static_cast<int64_t>(total_memory);

    // Create Bins 
    for (bin_index_t b = 0; b < kNumBins; b++) {
        size_t bin_size = bin_num_to_size(b);
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

bool BFCAllocator::extend(size_t alignment, size_t rounded_bytes) {
    
}

} // namespace base
} // namespace container