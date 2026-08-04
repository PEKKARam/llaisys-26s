# llaisys-26s 作业报告 

作者：  [PEKKARam](https://github.com/PEKKARam)

## 作业一 #1 张量

### 任务-1.1 load

```C++
void Tensor::load(const void *src) { // only tensor is continuous
    CHECK_ARGUMENT(src != nullptr, "load source must not be null");  // check.hpp
    const size_t bytes = numel() * elementSize();

    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(data(), src, bytes);
    } else {
        // The runtime API performs the host-to-device transfer on the tensor's device.
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(data(), src, bytes, LLAISYS_MEMCPY_H2D);
    }
}
```

### 任务-1.2 isContiguous

按照张量各维度的逻辑顺序访问元素时，相邻元素在内存中也必须紧密排列，期望步长应该与实际步长相符合，否则不连续。

[torch _geometry_is_contiguous](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorGeometry.cpp)

```C++
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
```

### 任务-1.3 view

作业要求：

    创建一个新张量，通过拆分或合并原始维度将原始张量重塑为给定形状。不涉及数据传输。例如，通过合并最后两个维度，将形状为(2, 3, 5)的张量更改为(2, 15)。

    这个函数不是简单地改变张量的形状那么简单，尽管测试会通过。如果新视图与原始张量不兼容，它应该引发错误。想想一个形状为(2, 3, 5)、步长为(30, 10, 1)的张量。你还能在不传输数据的情况下将其重塑为(2, 15)吗？

分析：

[torch.Tensor.view Doc](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.view.html)

[torch::aten computeStride_imlp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorUtils.cpp#L327)

不能复制数据，只能修改`shape`和`strides`，如果新形状中的某个大维（由原张量的多个维度合并而成），在原始内存中不是连续排列的，就必须报错。

根据参考资料，在内存中，只要满足 `strides[i-1] == shape[i] * strides[i]`，就说明第 `i-1` 维和第 `i` 维在内存中是连在一起的，它们可被归为同一个 Chunk。

略微修改torch源码，加入创建tensor语句如下。

```C++
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
```

### 任务-1.4 permute

[torch.Tensor.permute Doc](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.permute.html)

```C++
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == ndim(), "permute order must contain every dimension");

    TensorMeta meta{dtype(), std::vector<size_t>(ndim()), std::vector<ptrdiff_t>(ndim())};
    std::vector<bool> used(ndim(), false);
    for (size_t dim = 0; dim < ndim(); ++dim) {
        CHECK_ARGUMENT(order[dim] < ndim(), "permute dimension is out of range"); // error: order[dim] >= ndim()
        CHECK_ARGUMENT(!used[order[dim]], "permute order contains duplicate dimensions"); // error: duplicate dimenmsions
        used[order[dim]] = true;
        meta.shape[dim] = _meta.shape[order[dim]];
        meta.strides[dim] = _meta.strides[order[dim]];
    }

    // Permuting changes only metadata, so the new tensor shares storage and offset.
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}
```

### 任务-1.5 slice

[torch slice](https://github.com/pytorch/pytorch/blob/84b65be2832fa711f2d5683019aae626dd334ea8/aten/src/ATen/native/TensorShape.cpp#L3007)

[torch.select](https://docs.pytorch.org/docs/2.13/generated/torch.select.html)

观察到定义的`tensor.hpp::slice()`函数不含有类似slice step参数，默认为`step == 1`，并为了简化只支持正向切片。

```C++
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
```

## 参考资料

- [ezyang's blog: Pytorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)

