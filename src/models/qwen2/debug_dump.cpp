#include "debug_dump.hpp"

#include "../../utils/check.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace llaisys::models::qwen2::debug {

void dumpTensor(const char *directory, const char *name, tensor_t tensor) {
    CHECK_ARGUMENT(tensor->isContiguous(), "Qwen2 debug dump requires a contiguous tensor");
    const char *dtype = nullptr;
    switch (tensor->dtype()) {
    case LLAISYS_DTYPE_BF16: dtype = "bf16"; break;
    case LLAISYS_DTYPE_F16: dtype = "f16"; break;
    case LLAISYS_DTYPE_F32: dtype = "f32"; break;
    default: CHECK_ARGUMENT(false, "Unsupported Qwen2 debug dump dtype");
    }

    const size_t bytes = tensor->numel() * tensor->elementSize();
    std::vector<std::byte> host(bytes);
    if (tensor->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(host.data(), tensor->data(), bytes);
    } else {
        core::context().setDevice(tensor->deviceType(), tensor->deviceId());
        const auto *api = core::context().runtime().api();
        api->device_synchronize();
        api->memcpy_sync(host.data(), tensor->data(), bytes, LLAISYS_MEMCPY_D2H);
    }

    const auto root = std::filesystem::path(directory);
    std::filesystem::create_directories(root);
    const auto base = root / name;
    std::ofstream data(base.string() + ".bin", std::ios::binary | std::ios::trunc);
    data.exceptions(std::ios::failbit | std::ios::badbit);
    data.write(reinterpret_cast<const char *>(host.data()), static_cast<std::streamsize>(bytes));
    data.close();

    const uint16_t endian_probe = 1;
    const bool little_endian = *reinterpret_cast<const unsigned char *>(&endian_probe) == 1;
    std::ofstream meta(base.string() + ".json", std::ios::trunc);
    meta.exceptions(std::ios::failbit | std::ios::badbit);
    meta << "{\"format\":\"llaisys_raw_tensor_v1\",\"dtype\":\"" << dtype
         << "\",\"byte_order\":\"" << (little_endian ? "little" : "big")
         << "\",\"shape\":[";
    for (size_t i = 0; i < tensor->ndim(); ++i) {
        if (i > 0) meta << ',';
        meta << tensor->shape()[i];
    }
    meta << "],\"bytes\":" << bytes << "}\n";
    meta.close();
}

} // namespace llaisys::models::qwen2::debug
