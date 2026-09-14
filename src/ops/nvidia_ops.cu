#include "nvidia_ops.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace llaisys::ops::nvidia {
namespace {
__device__ inline float read16(const uint16_t *p, int mode) {
    if (mode == 1) return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(p));
    return __half2float(*reinterpret_cast<const __half *>(p));
}
__device__ inline void write16(uint16_t *p, float v, int mode) {
    if (mode == 1) *reinterpret_cast<__nv_bfloat16 *>(p) = __float2bfloat16(v);
    else *reinterpret_cast<__half *>(p) = __float2half(v);
}
template <int MODE> __device__ inline float load(const void *p, size_t i) {
    if constexpr (MODE != 0) return read16(reinterpret_cast<const uint16_t *>(p) + i, MODE);
    return reinterpret_cast<const float *>(p)[i];
}
template <int MODE> __device__ inline void store(void *p, size_t i, float v) {
    if constexpr (MODE != 0) write16(reinterpret_cast<uint16_t *>(p) + i, v, MODE);
    else reinterpret_cast<float *>(p)[i] = v;
}
template <int MODE> __global__ void add_k(void *c, const void *a, const void *b, size_t n) { size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) store<MODE>(c,i,load<MODE>(a,i)+load<MODE>(b,i)); }
template <int MODE> __global__ void emb_k(void *o,const int64_t *ids,const void *w,size_t n,size_t d){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*d)store<MODE>(o,i,load<MODE>(w,(size_t)ids[i/d]*d+i%d));}
template <int MODE> __global__ void linear_k(void *o,const void *x,const void *w,const void *bias,size_t M,size_t N,size_t K){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<M*N){size_t m=i/N,n=i%N;float s=bias?load<MODE>(bias,n):0.f;for(size_t k=0;k<K;k++)s+=load<MODE>(x,m*K+k)*load<MODE>(w,n*K+k);store<MODE>(o,i,s);}}
template <int MODE> __global__ void rms_k(void *o,const void *x,const void *w,size_t M,size_t K,float eps){size_t r=blockIdx.x*blockDim.x+threadIdx.x;if(r<M){float s=0;for(size_t k=0;k<K;k++){float v=load<MODE>(x,r*K+k);s+=v*v;}float inv=rsqrtf(s/K+eps);for(size_t k=0;k<K;k++)store<MODE>(o,r*K+k,load<MODE>(x,r*K+k)*load<MODE>(w,k)*inv);}}
template <int MODE> __global__ void rope_k(void *o,const void *x,const int64_t *pos,size_t S,size_t H,size_t D,float theta){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<S*H*D){size_t d=i%D,j=d<(D/2)?d:d-D/2;float a=load<MODE>(x,i);size_t base=i-d;size_t pair=base+(d<(D/2)?d+(D/2):d-(D/2));float b=load<MODE>(x,pair);float ang=(float)pos[i/(H*D)]/powf(theta,2.f*j/D);float v=d<(D/2)?a*cosf(ang)-b*sinf(ang):a*cosf(ang)+b*sinf(ang);store<MODE>(o,i,v);}}
template <int MODE> __global__ void swiglu_k(void *o,const void *g,const void *u,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float gv=load<MODE>(g,i),uv=load<MODE>(u,i);store<MODE>(o,i,uv*gv/(1.f+expf(-gv)));}}
template <int MODE> __global__ void argmax_k(int64_t *idx,void *val,const void *x,size_t rows,size_t K){size_t r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float best=load<MODE>(x,r*K),v;size_t bi=0;for(size_t k=1;k<K;k++){v=load<MODE>(x,r*K+k);if(v>best){best=v;bi=k;}}idx[r]=bi;store<MODE>(val,r,best);}}
template <int MODE> __global__ void attn_k(void *o,const void *q,const void *k,const void *v,size_t Q,size_t KL,size_t HQ,size_t HK,size_t D,float scale){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<Q*HQ*D){size_t d=i%D,h=(i/D)%HQ,qp=i/(HQ*D),kh=h/(HQ/HK),limit=KL-Q+qp+1;float mx=-1e30f;for(size_t p=0;p<limit;p++){float s=0;for(size_t z=0;z<D;z++)s+=load<MODE>(q,(qp*HQ+h)*D+z)*load<MODE>(k,(p*HK+kh)*D+z);mx=fmaxf(mx,s*scale);}float den=0,res=0;for(size_t p=0;p<limit;p++){float s=0;for(size_t z=0;z<D;z++)s+=load<MODE>(q,(qp*HQ+h)*D+z)*load<MODE>(k,(p*HK+kh)*D+z);float e=expf(s*scale-mx);den+=e;res+=e*load<MODE>(v,(p*HK+kh)*D+d);}store<MODE>(o,i,res/den);}}
#define LAUNCH(kernel, n, ...) do { kernel<<<((n)+255)/256,256>>>(__VA_ARGS__); cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
}

#define DISPATCH(T, CALL) do { if ((T)==LLAISYS_DTYPE_F32) { CALL(float); } else if ((T)==LLAISYS_DTYPE_BF16) { CALL(bf); } else if ((T)==LLAISYS_DTYPE_F16) { CALL(half); } else throw std::invalid_argument("CUDA op supports F32/F16/BF16"); } while(0)

void add(tensor_t c,tensor_t a,tensor_t b){if(c->dtype()==LLAISYS_DTYPE_F32)LAUNCH((add_k<0>),c->numel(),c->data(),a->data(),b->data(),c->numel());else if(c->dtype()==LLAISYS_DTYPE_BF16)LAUNCH((add_k<1>),c->numel(),c->data(),a->data(),b->data(),c->numel());else LAUNCH((add_k<2>),c->numel(),c->data(),a->data(),b->data(),c->numel());}
void embedding(tensor_t o,tensor_t ids,tensor_t w){int m=w->dtype()==LLAISYS_DTYPE_F32?0:w->dtype()==LLAISYS_DTYPE_BF16?1:2;if(m==0)LAUNCH((emb_k<0>),o->numel(),o->data(),reinterpret_cast<const int64_t*>(ids->data()),w->data(),ids->numel(),w->shape()[1]);else if(m==1)LAUNCH((emb_k<1>),o->numel(),o->data(),reinterpret_cast<const int64_t*>(ids->data()),w->data(),ids->numel(),w->shape()[1]);else LAUNCH((emb_k<2>),o->numel(),o->data(),reinterpret_cast<const int64_t*>(ids->data()),w->data(),ids->numel(),w->shape()[1]);}
void linear(tensor_t o,tensor_t x,tensor_t w,tensor_t bias){size_t K=x->shape().back(),M=x->numel()/K,N=w->shape()[0];if(w->dtype()==LLAISYS_DTYPE_F32)LAUNCH((linear_k<0>),o->numel(),o->data(),x->data(),w->data(),bias?bias->data():nullptr,M,N,K);else if(w->dtype()==LLAISYS_DTYPE_BF16)LAUNCH((linear_k<1>),o->numel(),o->data(),x->data(),w->data(),bias?bias->data():nullptr,M,N,K);else LAUNCH((linear_k<2>),o->numel(),o->data(),x->data(),w->data(),bias?bias->data():nullptr,M,N,K);}
void rms_norm(tensor_t o,tensor_t x,tensor_t w,float e){size_t K=x->shape().back(),M=x->numel()/K;if(x->dtype()==LLAISYS_DTYPE_F32)LAUNCH((rms_k<0>),M,o->data(),x->data(),w->data(),M,K,e);else if(x->dtype()==LLAISYS_DTYPE_BF16)LAUNCH((rms_k<1>),M,o->data(),x->data(),w->data(),M,K,e);else LAUNCH((rms_k<2>),M,o->data(),x->data(),w->data(),M,K,e);}
void rope(tensor_t o,tensor_t x,tensor_t p,float th){size_t S=x->shape()[0],H=x->shape()[1],D=x->shape()[2];if(x->dtype()==LLAISYS_DTYPE_F32)LAUNCH((rope_k<0>),o->numel(),o->data(),x->data(),reinterpret_cast<const int64_t*>(p->data()),S,H,D,th);else if(x->dtype()==LLAISYS_DTYPE_BF16)LAUNCH((rope_k<1>),o->numel(),o->data(),x->data(),reinterpret_cast<const int64_t*>(p->data()),S,H,D,th);else LAUNCH((rope_k<2>),o->numel(),o->data(),x->data(),reinterpret_cast<const int64_t*>(p->data()),S,H,D,th);}
void swiglu(tensor_t o,tensor_t g,tensor_t u){if(o->dtype()==LLAISYS_DTYPE_F32)LAUNCH((swiglu_k<0>),o->numel(),o->data(),g->data(),u->data(),o->numel());else if(o->dtype()==LLAISYS_DTYPE_BF16)LAUNCH((swiglu_k<1>),o->numel(),o->data(),g->data(),u->data(),o->numel());else LAUNCH((swiglu_k<2>),o->numel(),o->data(),g->data(),u->data(),o->numel());}
void argmax(tensor_t idx,tensor_t val,tensor_t x){size_t K=x->shape().back(),rows=x->numel()/K;if(x->dtype()==LLAISYS_DTYPE_F32)LAUNCH((argmax_k<0>),rows,reinterpret_cast<int64_t*>(idx->data()),val->data(),x->data(),rows,K);else if(x->dtype()==LLAISYS_DTYPE_BF16)LAUNCH((argmax_k<1>),rows,reinterpret_cast<int64_t*>(idx->data()),val->data(),x->data(),rows,K);else LAUNCH((argmax_k<2>),rows,reinterpret_cast<int64_t*>(idx->data()),val->data(),x->data(),rows,K);}
void self_attention(tensor_t o,tensor_t q,tensor_t k,tensor_t v,float scale){size_t Q=q->shape()[0],KL=k->shape()[0],HQ=q->shape()[1],HK=k->shape()[1],D=q->shape()[2];if(q->dtype()==LLAISYS_DTYPE_F32)LAUNCH((attn_k<0>),o->numel(),o->data(),q->data(),k->data(),v->data(),Q,KL,HQ,HK,D,scale);else if(q->dtype()==LLAISYS_DTYPE_BF16)LAUNCH((attn_k<1>),o->numel(),o->data(),q->data(),k->data(),v->data(),Q,KL,HQ,HK,D,scale);else LAUNCH((attn_k<2>),o->numel(),o->data(),q->data(),k->data(),v->data(),Q,KL,HQ,HK,D,scale);}
}
