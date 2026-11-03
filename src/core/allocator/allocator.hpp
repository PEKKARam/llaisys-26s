#pragma once

#include "llaisys/runtime.h"

#include "../storage/storage.hpp"
#include <algorithm>
#include <unordered_map>

namespace llaisys::core {
class MemoryAllocator {
protected:
    const LlaisysRuntimeAPI *_api;
    std::unordered_map<std::byte *, size_t> _live;
    LlaisysAllocatorStats _stats{};
    MemoryAllocator(const LlaisysRuntimeAPI *runtime_api) : _api(runtime_api){};

public:
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;
    virtual void release(std::byte *memory) = 0;
    LlaisysAllocatorStats stats() const { return _stats; }
    void resetPeak() { _stats.peak_bytes = _stats.current_bytes; }
};

} // namespace llaisys::core
