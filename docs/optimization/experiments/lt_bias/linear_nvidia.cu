#include "../../nvidia/common.cuh"
#include "linear_nvidia.cuh"

#include <cmath>
#include "../../nvidia/resources.cuh"
#include <climits>

namespace llaisys::ops::nvidia {
namespace {
using detail::load;
using detail::store;

// X[M, K] * W[N, K]^T + bias[N] -> output[M, N].
// One thread computes one output element, accumulating across K in float.
template <int MODE>
__global__ void linear_k(void *o, const void *x, const void *w, const void *bias, size_t M,
                         size_t N, size_t K) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) {
        size_t m = i / N, n = i % N;
        float s = bias ? load<MODE>(bias, n) : 0.f;
        for (size_t k = 0; k < K; k++) {
            s += load<MODE>(x, m * K + k) * load<MODE>(w, n * K + k);
        }
        store<MODE>(o, i, s);
    }
}

} // namespace

void linear_v0(tensor_t o, tensor_t x, tensor_t w, tensor_t bias) {
    size_t K = x->shape().back(), M = x->numel() / K, N = w->shape()[0];
    if (w->dtype() == LLAISYS_DTYPE_F32) {
        LLAISYS_CUDA_LAUNCH((linear_k<0>), o->numel(), o->data(), x->data(), w->data(),
                            bias ? bias->data() : nullptr, M, N, K);
    } else if (w->dtype() == LLAISYS_DTYPE_BF16) {
        LLAISYS_CUDA_LAUNCH((linear_k<1>), o->numel(), o->data(), x->data(), w->data(),
                            bias ? bias->data() : nullptr, M, N, K);
    } else {
        LLAISYS_CUDA_LAUNCH((linear_k<2>), o->numel(), o->data(), x->data(), w->data(),
                            bias ? bias->data() : nullptr, M, N, K);
    }
}


namespace {
template<int MODE> __global__ void linear_bias(void *out,const float *product,const void *bias,size_t count,size_t n) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)store<MODE>(out,i,product[i]+load<MODE>(bias,i%n));
}

struct LtPlan {
    cublasLtMatmulDesc_t desc=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int found=0;
    LtPlan(size_t m,size_t n,size_t k,cudaDataType_t type,cublasLtHandle_t handle) {
        detail::blasCheck(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
        detail::blasCheck(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
        detail::blasCheck(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
        cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;
        detail::blasCheck(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
        detail::blasCheck(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&type,sizeof(type)));
        detail::blasCheck(cublasLtMatrixLayoutCreate(&a,type,k,n,k));
        detail::blasCheck(cublasLtMatrixLayoutCreate(&b,type,k,m,k));
        detail::blasCheck(cublasLtMatrixLayoutCreate(&c,type,n,m,n));
        cublasLtMatmulPreference_t pref;
        detail::blasCheck(cublasLtMatmulPreferenceCreate(&pref));
        size_t bytes=32*1024*1024;
        detail::blasCheck(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
        uint32_t alignment=2;
        for(auto attr:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            detail::blasCheck(cublasLtMatmulPreferenceSetAttribute(pref,attr,&alignment,sizeof(alignment)));
        detail::blasCheck(cublasLtMatmulAlgoGetHeuristic(handle,desc,a,b,c,c,pref,1,&heuristic,&found));
        cublasLtMatmulPreferenceDestroy(pref);
    }
    ~LtPlan(){if(desc)cublasLtMatmulDescDestroy(desc);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
inline bool lt_bias(tensor_t out,tensor_t x,tensor_t w,tensor_t bias,size_t m,size_t n,size_t k,cudaDataType_t dtype,detail::Resources &state) {
    static const bool use_lt=[] {const char *v=std::getenv("LLAISYS_LINEAR_BIAS_MODE");return !v||std::string(v)!="fp32";}();
    if(!use_lt)return false;
    thread_local std::unordered_map<std::string,std::unique_ptr<LtPlan>> plans;
    std::string key=std::to_string(state.device)+":"+std::to_string(m)+":"+std::to_string(n)+":"+std::to_string(k)+":"+std::to_string(int(dtype));
    auto &plan=plans[key];if(!plan)plan=std::make_unique<LtPlan>(m,n,k,dtype,state.lt);
    if(!plan->found)return false;
    const void *bias_pointer=bias->data();
    detail::blasCheck(cublasLtMatmulDescSetAttribute(plan->desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias_pointer,sizeof(bias_pointer)));
    const float alpha=1.f,beta=0.f;
    void *workspace=state.blas_workspace.reserve(32*1024*1024);
    detail::blasCheck(cublasLtMatmul(state.lt,plan->desc,&alpha,w->data(),plan->a,x->data(),plan->b,&beta,out->data(),plan->c,out->data(),plan->c,&plan->heuristic.algo,workspace,state.blas_workspace.capacity,detail::launchStream()));
    return true;
}
template<int MODE> void linear_blas(tensor_t out,tensor_t x,tensor_t w,tensor_t bias) {
    const size_t k=x->shape().back(),m=x->numel()/k,n=w->shape()[0];
    if(m>INT_MAX||n>INT_MAX||k>INT_MAX)throw std::invalid_argument("Linear dimensions exceed cuBLAS int range");
    auto &state=detail::resources();
    detail::blasCheck(cublasSetStream(state.blas,detail::launchStream()));
    const cudaDataType_t dtype=MODE==0?CUDA_R_32F:(MODE==1?CUDA_R_16BF:CUDA_R_16F);
    if(MODE!=0 && bias && lt_bias(out,x,w,bias,m,n,k,dtype,state))return;
    const float alpha=1.f,beta=0.f;
    void *target=out->data();cudaDataType_t target_dtype=dtype;
    if(bias){target=state.linear.reserve(m*n*sizeof(float));target_dtype=CUDA_R_32F;}
    // Row-major Y=XW^T is column-major Y^T=W X^T. No transpose copy is needed.
    detail::blasCheck(cublasGemmEx(state.blas,CUBLAS_OP_T,CUBLAS_OP_N,
        int(n),int(m),int(k),&alpha,w->data(),dtype,int(k),x->data(),dtype,int(k),
        &beta,target,target_dtype,int(n),MODE==0?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    if(bias)LLAISYS_CUDA_LAUNCH((linear_bias<MODE>),m*n,out->data(),static_cast<const float*>(target),bias->data(),m*n,n);
}
}
void linear(tensor_t out,tensor_t x,tensor_t w,tensor_t bias) {
    static const int version=detail::selectedVersion("LINEAR");
    if(version<2)return linear_v0(out,x,w,bias);
    if(x->dtype()==LLAISYS_DTYPE_F32)return linear_blas<0>(out,x,w,bias);
    if(x->dtype()==LLAISYS_DTYPE_BF16)return linear_blas<1>(out,x,w,bias);
    return linear_blas<2>(out,x,w,bias);
}
} // namespace llaisys::ops::nvidia
