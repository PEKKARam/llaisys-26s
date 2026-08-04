#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    ptrdiff_t expected_stride = 1;
    bool contig_if_nonempty = true;
    for (size_t dim = ndim(); dim-- > 0;) { // size_t 遍历注意判断条件
        if (_meta.shape[dim] == 0) {
            return true;
        }
        if (contig_if_nonempty) {
            if (_meta.shape[dim] != 1 && _meta.strides[dim] != expected_stride) {
                contig_if_nonempty = false;
            }
            expected_stride *= static_cast<ptrdiff_t>(_meta.shape[dim]);
        }
    }
    return contig_if_nonempty;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == ndim(), "permute order must contain every dimension");

    TensorMeta meta{dtype(), std::vector<size_t>(ndim()), std::vector<ptrdiff_t>(ndim())};
    std::vector<bool> used(ndim(), false);
    for (size_t dim = 0; dim < ndim(); ++dim) {
        CHECK_ARGUMENT(order[dim] < ndim(), "permute dimension is out of range");         // error: order[dim] >= ndim()
        CHECK_ARGUMENT(!used[order[dim]], "permute order contains duplicate dimensions"); // error: duplicate dimenmsions
        used[order[dim]] = true;
        meta.shape[dim] = _meta.shape[order[dim]];
        meta.strides[dim] = _meta.strides[order[dim]];
    }

    // Permuting changes only metadata, so the new tensor shares storage and offset.
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    const size_t new_numel = std::accumulate(
        shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(new_numel == numel(), "view shape must preserve the number of elements");

    std::vector<ptrdiff_t> new_strides(shape.size());

    if (numel() == 0 || ndim() == 0) { // old tensor is empty
        // Empty and scalar tensors have no layout constraints;
        ptrdiff_t stride = 1;
        for (size_t dim = shape.size(); dim-- > 0;) {
            new_strides[dim] = stride;
            stride *= static_cast<ptrdiff_t>(shape[dim]);
        }
    } else { // old tensor is not empty
        // Match each contiguous chunk of the old layout with dimensions in the view.
        ptrdiff_t view_dim = static_cast<ptrdiff_t>(shape.size()) - 1;
        // stride for each subspace in the chunk
        ptrdiff_t chunk_base_stride = _meta.strides.back();
        // numel in current chunk
        size_t tensor_elements = 1;
        size_t view_elements = 1;

        for (ptrdiff_t tensor_dim = static_cast<ptrdiff_t>(ndim()) - 1; tensor_dim >= 0; --tensor_dim) { // ptrdiff_t is signed
            tensor_elements *= _meta.shape[tensor_dim];
            // 包含了tensor_dim == 0 或 stride == 1
            const bool chunk_finished = tensor_dim == 0
                                     || (_meta.shape[tensor_dim - 1] != 1
                                         && _meta.strides[tensor_dim - 1] != static_cast<ptrdiff_t>(tensor_elements) * chunk_base_stride);

            if (!chunk_finished) {
                continue;
            }

            while (view_dim >= 0
                   && (view_elements < tensor_elements || shape[view_dim] == 1)) {
                new_strides[view_dim] = static_cast<ptrdiff_t>(view_elements) * chunk_base_stride;
                view_elements *= shape[view_dim];
                --view_dim;
            }

            CHECK_ARGUMENT(view_elements == tensor_elements,
                           "view shape is incompatible with the tensor strides");

            if (tensor_dim > 0) {
                chunk_base_stride = _meta.strides[tensor_dim - 1];
                tensor_elements = 1;
                view_elements = 1;
            }
        }

        CHECK_ARGUMENT(view_dim == -1, "view shape is incompatible with the tensor strides");
    }

    TensorMeta meta{dtype(), shape, std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const { // only support slice_step = 1 and positive slicing
    CHECK_ARGUMENT(dim < ndim(), "slice dimension is out of range");
    CHECK_ARGUMENT(start <= end, "slice start must not be greater than end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice end is out of range");
    CHECK_ARGUMENT(_meta.strides[dim] >= 0, "negative strides are not supported");

    TensorMeta meta = _meta;
    meta.shape[dim] = end - start;
    const size_t byte_offset = start * static_cast<size_t>(_meta.strides[dim]) * elementSize();

    // Slicing advances the byte offset but keeps the original storage and strides.
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset + byte_offset));
}

void Tensor::load(const void *src) { // only tensor is continuous
    CHECK_ARGUMENT(src != nullptr, "load source must not be null");
    const size_t bytes = numel() * elementSize();

    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(data(), src, bytes);
    } else {
        // The runtime API performs the host-to-device transfer on the tensor's device.
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(data(), src, bytes, LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
