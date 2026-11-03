#include "llaisys/runtime.h"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"

// Llaisys API for setting context runtime.
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type,
                                  int device_id) {
    llaisys::core::context().setDevice(device_type, device_id);
}

// Llaisys API for getting the runtime APIs
__C const LlaisysRuntimeAPI *
llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    return llaisys::device::getRuntimeAPI(device_type);
}

__C int llaisysGetAllocatorStats(llaisysDeviceType_t device, int id,
                                 LlaisysAllocatorStats *stats) {
    try {
        if (!stats) {
            return -1;
        }
        llaisys::core::context().setDevice(device, id);
        *stats = llaisys::core::context().runtime().allocatorStats();
        return 0;
    } catch (...) { return -1; }
}
__C int llaisysResetAllocatorPeak(llaisysDeviceType_t device, int id) {
    try {
        llaisys::core::context().setDevice(device, id);
        llaisys::core::context().runtime().resetAllocatorPeak();
        return 0;
    } catch (...) { return -1; }
}
