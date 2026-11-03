#include "naive_allocator.hpp"

#include "../runtime/runtime.hpp"

namespace llaisys::core::allocators {
NaiveAllocator::NaiveAllocator(const LlaisysRuntimeAPI *runtime_api)
    : MemoryAllocator(runtime_api) {}

std::byte *NaiveAllocator::allocate(size_t size) {
    auto *memory = static_cast<std::byte *>(_api->malloc_device(size));
    _live[memory] = size;
    _stats.current_bytes += size;
    _stats.peak_bytes = std::max(_stats.peak_bytes, _stats.current_bytes);
    ++_stats.allocations;
    return memory;
}

void NaiveAllocator::release(std::byte *memory) {
    _api->free_device(memory);
    auto entry = _live.find(memory);
    if (entry != _live.end()) {
        _stats.current_bytes -= entry->second;
        _live.erase(entry);
    }
}
} // namespace llaisys::core::allocators
