#include "../../nvidia/common.cuh"
#include "argmax_nvidia.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// Reduce each row of length K; strict > preserves the first index on ties.
template <int MODE>
__global__ void argmax_k(int64_t *idx, void *val, const void *x, size_t rows, size_t K) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float best = load<MODE>(x, r * K), v;
        size_t bi = 0;
        for (size_t k = 1; k < K; k++) {
            v = load<MODE>(x, r * K + k);
            if (v > best) {
                best = v;
                bi = k;
            }
        }
        idx[r] = bi;
        store<MODE>(val, r, best);
    }
}

} // namespace

void argmax_v0(tensor_t idx, tensor_t val, tensor_t x) {
    size_t K = x->shape().back(), rows = x->numel() / K;
    if (x->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((argmax_k<0>), rows, reinterpret_cast<int64_t *>(idx->data()),
                            val->data(), x->data(), rows, K);
    } else if (x->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((argmax_k<1>), rows, reinterpret_cast<int64_t *>(idx->data()),
                            val->data(), x->data(), rows, K);
    } else {
        LLAISYS_CUDA_LAUNCH((argmax_k<2>), rows, reinterpret_cast<int64_t *>(idx->data()),
                            val->data(), x->data(), rows, K);
    }
}


namespace {
__device__ bool better(float a,unsigned long long ia,float b,unsigned long long ib) {
    if(isnan(a))return !isnan(b)||ia<ib;
    if(isnan(b))return false;
    return a>b||(a==b&&ia<ib);
}
template<int MODE> __global__ void argmax_parallel_v2(int64_t *index,void *value,const void *x,size_t k) {
    size_t row=blockIdx.x;
    float best=-INFINITY;unsigned long long bi=~0ULL;
    __shared__ float values[8];__shared__ unsigned long long indices[8];
    for(size_t j=threadIdx.x;j<k;j+=blockDim.x){float v=load<MODE>(x,row*k+j);if(better(v,j,best,bi)){best=v;bi=j;}}
    for(int offset=16;offset>0;offset>>=1){float v=__shfl_down_sync(0xffffffff,best,offset);auto i=__shfl_down_sync(0xffffffff,bi,offset);if(better(v,i,best,bi)){best=v;bi=i;}}
    if((threadIdx.x&31)==0){values[threadIdx.x/32]=best;indices[threadIdx.x/32]=bi;}
    __syncthreads();
    if(threadIdx.x<32){best=threadIdx.x<8?values[threadIdx.x]:-INFINITY;bi=threadIdx.x<8?indices[threadIdx.x]:~0ULL;
        for(int offset=16;offset>0;offset>>=1){float v=__shfl_down_sync(0xffffffff,best,offset);auto i=__shfl_down_sync(0xffffffff,bi,offset);if(better(v,i,best,bi)){best=v;bi=i;}}
        if(threadIdx.x==0){index[row]=int64_t(bi);store<MODE>(value,row,best);}}
}
template<int MODE> void argmax_run(tensor_t i,tensor_t v,tensor_t x){size_t k=x->shape().back();argmax_parallel_v2<MODE><<<x->numel()/k,256,0,detail::launchStream()>>>(reinterpret_cast<int64_t*>(i->data()),v->data(),x->data(),k);auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
}
void argmax(tensor_t i,tensor_t v,tensor_t x) {
    static const int version=detail::selectedVersion("ARGMAX");
    if(version<2)return argmax_v0(i,v,x);
    if(x->dtype()==LLAISYS_DTYPE_F32)return argmax_run<0>(i,v,x);
    if(x->dtype()==LLAISYS_DTYPE_BF16)return argmax_run<1>(i,v,x);
    return argmax_run<2>(i,v,x);
}
} // namespace llaisys::ops::nvidia
