// Synthetic Strata operators on MI50. CPU oracle is the pinned ggml dequantizer,
// NOT Strata's own GPU dequantizer. Timings are warm-cache HIP graphs, not tok/s.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "ggml.h"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace sk = strata::kernels;
static void ck(hipError_t e,const char* s){if(e!=hipSuccess)throw std::runtime_error(std::string(s)+": "+hipGetErrorString(e));}
#define HIP(x) ck((x),#x)
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t n){HIP(hipMalloc(&p,n));}
    ~Buffer(){if(p)hipFree(p);}
    Buffer(const Buffer&)=delete;
    template<class T> T* as(){return static_cast<T*>(p);}
};
static double rel2(const std::vector<float>& a,const std::vector<float>& b){
    double n=0,d=0;
    for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))return INFINITY;double e=(double)a[i]-b[i];n+=e*e;d+=(double)b[i]*b[i];}
    return std::sqrt(n/(d+1e-30));
}
static std::vector<unsigned char> weights(int type,int rows,int cols,unsigned seed){
    const auto* tr=ggml_get_type_traits((ggml_type)type);
    size_t row=ggml_row_size((ggml_type)type,cols), bs=tr->type_size;
    std::vector<unsigned char> w(row*rows);std::mt19937 rng(seed);
    for(auto& b:w)b=(unsigned char)rng();
    // Most formats start with a half scale. IQ1_M instead distributes its
    // global half across the high nibbles of four uint16 scale words (last 8B).
    // Keep random grids/local scales, but force a finite, nonzero global d.
    for(size_t i=0;i<w.size();i+=bs){
        if(type==GGML_TYPE_IQ1_M){
            for(unsigned j=0;j<4;++j){
                size_t at=i+bs-8+2*j;
                unsigned sc=(unsigned)w[at]|((unsigned)w[at+1]<<8);
                sc=(sc&0x0fffu)|(((0x1800u>>(4*j))&15u)<<12);
                w[at]=(unsigned char)sc;w[at+1]=(unsigned char)(sc>>8);
            }
        }else{w[i]=0x00;w[i+1]=0x18;}
    }
    return w;
}
static std::vector<float> dequant(int type,const unsigned char* w,int rows,int cols){
    const auto* tr=ggml_get_type_traits((ggml_type)type);
    if(!tr->to_float)throw std::runtime_error("missing ggml CPU oracle");
    size_t rb=ggml_row_size((ggml_type)type,cols);std::vector<float> f((size_t)rows*cols);
    for(int r=0;r<rows;++r)tr->to_float(w+r*rb,f.data()+(size_t)r*cols,cols);
    for(float v:f)if(!std::isfinite(v))throw std::runtime_error("invalid synthetic fixture");
    return f;
}
static std::vector<float> activations(int n,unsigned seed){
    std::mt19937 rng(seed);std::uniform_real_distribution<float> u(-.2f,.2f);
    std::vector<float> x(n);for(auto& v:x)v=u(rng);return x;
}
static std::vector<float> matref(const std::vector<float>& w,const std::vector<float>& x,int n,int k,int t){
    std::vector<float> y((size_t)n*t);
    for(int c=0;c<t;++c)for(int r=0;r<n;++r){double z=0;for(int j=0;j<k;++j)z+=(double)w[(size_t)r*k+j]*x[(size_t)c*k+j];y[(size_t)c*n+r]=(float)z;}
    return y;
}
static double bench(hipStream_t s,const std::function<void()>& op,int calls=100){
    for(int i=0;i<5;++i)op();HIP(hipStreamSynchronize(s));
    hipGraph_t g;hipGraphExec_t ex;HIP(hipStreamBeginCapture(s,hipStreamCaptureModeThreadLocal));
    for(int i=0;i<calls;++i)op();HIP(hipStreamEndCapture(s,&g));HIP(hipGraphInstantiateWithFlags(&ex,g,0));
    hipEvent_t a,b;HIP(hipEventCreate(&a));HIP(hipEventCreate(&b));std::vector<float> samples;
    for(int i=0;i<9;++i){HIP(hipEventRecord(a,s));HIP(hipGraphLaunch(ex,s));HIP(hipEventRecord(b,s));HIP(hipEventSynchronize(b));float ms;HIP(hipEventElapsedTime(&ms,a,b));if(i>1)samples.push_back(ms);}
    std::sort(samples.begin(),samples.end());double us=samples[samples.size()/2]*1000.0/calls;
    HIP(hipEventDestroy(a));HIP(hipEventDestroy(b));HIP(hipGraphExecDestroy(ex));HIP(hipGraphDestroy(g));return us;
}
static void mmvq(hipStream_t s,int type,int n,int k,int t){
    auto w=weights(type,n,k,123+type);
    auto wf=dequant(type,w.data(),n,k),x=activations(k*t,44);
    auto ref=matref(wf,x,n,k,t);Buffer dw(w.size()),dx(x.size()*4),dq(sk::native_q8_1_bytes(k,t)),dy(ref.size()*4),df(wf.size()*4);
    HIP(hipMemcpy(dw.p,w.data(),w.size(),hipMemcpyHostToDevice));HIP(hipMemcpy(dx.p,x.data(),x.size()*4,hipMemcpyHostToDevice));
    sk::iq_dequant_f32(type,dw.p,(int64_t)n*k,df.as<float>(),s);HIP(hipStreamSynchronize(s));
    std::vector<float> gotf(wf.size());HIP(hipMemcpy(gotf.data(),df.p,gotf.size()*4,hipMemcpyDeviceToHost));double de=rel2(gotf,wf);
    if(de>1e-6)throw std::runtime_error(std::string(ggml_type_name((ggml_type)type))+" dequant oracle mismatch "+std::to_string(de));
    sk::native_quantize_q8_1(dx.as<float>(),dq.p,k,t,s);
    auto op=[&]{sk::native_mmvq(type,dw.p,dq.p,dy.as<float>(),k,n,t,s);};op();HIP(hipStreamSynchronize(s));
    std::vector<float> got(ref.size());HIP(hipMemcpy(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost));double e=rel2(got,ref);
    if(e>2e-2)throw std::runtime_error(std::string(ggml_type_name((ggml_type)type))+" MMVQ oracle mismatch "+std::to_string(e));
    double us=bench(s,op);
    std::printf("RESULT kind=mmvq dtype=%s N=%d K=%d T=%d us=%.6f dequant_rel2=%.8g rel2=%.8g\n",ggml_type_name((ggml_type)type),n,k,t,us,de,e);
}
static void expert(hipStream_t s,int type,int t,int down_type=GGML_TYPE_IQ4_NL){
    constexpr int H=2560,FF=640,G=10;const int entries=G*t;
    auto L=sk::native_expert_layout(type,down_type,H,FF);
    std::vector<unsigned char> blob(L.bytes*G);std::vector<float> ref((size_t)entries*H);
    auto x=activations(H*t,7);
    for(int g=0;g<G;++g){
        auto gate=weights(type,FF,H,100+g),up=weights(type,FF,H,200+g),down=weights(down_type,H,FF,300+g);
        auto gf=dequant(type,gate.data(),FF,H),uf=dequant(type,up.data(),FF,H),df=dequant(down_type,down.data(),H,FF);
        auto gr=matref(gf,x,FF,H,t),ur=matref(uf,x,FF,H,t);std::vector<float> h(gr.size());
        for(size_t i=0;i<h.size();++i)h[i]=(float)((double)gr[i]/(1+std::exp(-(double)gr[i]))*ur[i]);
        auto out=matref(df,h,H,FF,t);std::copy(out.begin(),out.end(),ref.begin()+(size_t)g*t*H);
        auto* dst=blob.data()+g*L.bytes;std::memcpy(dst,gate.data(),gate.size());std::memcpy(dst+L.up_off,up.data(),up.size());std::memcpy(dst+L.down_off,down.data(),down.size());
    }
    Buffer db(blob.size()),dx(x.size()*4),dq(sk::native_q8_1_bytes(H,t)),dy(ref.size()*4),scratch(sk::native_expert_scratch_bytes(entries,FF));
    Buffer dp(G*8),starts((G+1)*4),groups(4),dst(entries*4),tok(entries*4);
    HIP(hipMemcpy(db.p,blob.data(),blob.size(),hipMemcpyHostToDevice));HIP(hipMemcpy(dx.p,x.data(),x.size()*4,hipMemcpyHostToDevice));
    std::vector<unsigned long long> ptr(G);std::vector<int> st(G+1),di(entries),ti(entries);int ng=G;
    for(int g=0;g<G;++g){ptr[g]=(unsigned long long)db.p+g*L.bytes;st[g]=g*t;}st[G]=entries;
    for(int i=0;i<entries;++i){di[i]=i;ti[i]=i%t;}
    HIP(hipMemcpy(dp.p,ptr.data(),G*8,hipMemcpyHostToDevice));HIP(hipMemcpy(starts.p,st.data(),(G+1)*4,hipMemcpyHostToDevice));HIP(hipMemcpy(groups.p,&ng,4,hipMemcpyHostToDevice));
    HIP(hipMemcpy(dst.p,di.data(),entries*4,hipMemcpyHostToDevice));HIP(hipMemcpy(tok.p,ti.data(),entries*4,hipMemcpyHostToDevice));
    sk::quantize_q8_1_rows(dx.as<float>(),t,H,dq.p,s);
    auto op=[&]{sk::native_expert_grouped(L,dp.as<unsigned long long>(),starts.as<int32_t>(),groups.as<int32_t>(),dst.as<int32_t>(),tok.as<int32_t>(),G,entries,dq.p,scratch.p,dy.as<float>(),s);};
    op();HIP(hipStreamSynchronize(s));std::vector<float> got(ref.size());HIP(hipMemcpy(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost));double e=rel2(got,ref);
    if(e>3e-2)throw std::runtime_error(std::string(ggml_type_name((ggml_type)type))+" expert oracle mismatch "+std::to_string(e));
    double us=bench(s,op,20);
    std::printf("RESULT kind=expert dtype=%s/%s G=%d H=%d FF=%d T=%d bytes=%zu us=%.6f rel2=%.8g\n",ggml_type_name((ggml_type)type),ggml_type_name((ggml_type)down_type),G,H,FF,t,blob.size(),us,e);
}
static void gemm(hipStream_t s,bool bf,int t,int n,int k){
    auto x=activations(t*k,21),w=activations(n*k,23);std::vector<uint16_t> xb(x.size()),wb(w.size());
    auto cv=[&](float v){if(bf){uint32_t bits;std::memcpy(&bits,&v,4);return (uint16_t)((bits+0x7fff+((bits>>16)&1))>>16);}__half h=__float2half(v);uint16_t bits;std::memcpy(&bits,&h,2);return bits;};
    auto dc=[&](uint16_t v){if(bf){uint32_t bits=(uint32_t)v<<16;float f;std::memcpy(&f,&bits,4);return f;}__half h;std::memcpy(&h,&v,2);return __half2float(h);};
    for(size_t i=0;i<x.size();++i){xb[i]=cv(x[i]);x[i]=dc(xb[i]);}for(size_t i=0;i<w.size();++i){wb[i]=cv(w[i]);w[i]=dc(wb[i]);}
    Buffer dx(xb.size()*2),dw(wb.size()*2),dy((size_t)t*n*4);HIP(hipMemcpy(dx.p,xb.data(),xb.size()*2,hipMemcpyHostToDevice));HIP(hipMemcpy(dw.p,wb.data(),wb.size()*2,hipMemcpyHostToDevice));
    strata::prefill::Gemm g;std::string err;if(!g.init(s,1,err))throw std::runtime_error("Gemm init: "+err);
    auto op=[&]{if(bf)g.bf16(dx.as<uint16_t>(),dw.as<uint16_t>(),dy.as<float>(),t,n,k);else g.f16(dx.as<uint16_t>(),dw.as<uint16_t>(),dy.as<float>(),t,n,k);};
    op();HIP(hipStreamSynchronize(s));std::vector<float> got((size_t)t*n);HIP(hipMemcpy(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost));
    for(float v:got)if(!std::isfinite(v))throw std::runtime_error("nonfinite Gemm output");
    // Independent double-accumulator oracle on 64 spread output elements at production dimensions.
    double ne=0,de=0;for(size_t q=0;q<64;++q){size_t i=q*(got.size()-1)/63;int c=i/n,r=i%n;double z=0;for(int j=0;j<k;++j)z+=(double)x[(size_t)c*k+j]*w[(size_t)r*k+j];if(!std::isfinite(got[i]))throw std::runtime_error("nonfinite Gemm");ne+=(got[i]-z)*(got[i]-z);de+=z*z;}
    double e=std::sqrt(ne/(de+1e-30));if(e>2e-5)throw std::runtime_error("Gemm double oracle mismatch "+std::to_string(e));
    double us=bench(s,op,10);std::printf("RESULT kind=gemm dtype=%s T=%d N=%d K=%d us=%.6f TFLOPS=%.6f rel2=%.8g\n",bf?"BF16":"FP16",t,n,k,us,2.0*t*n*k/us/1e6,e);
}
static void mmq(hipStream_t s,int type,int n,int k,int t){
    namespace mq = strata::prefill::mmq;
    if(!mq::built()||!mq::supported(type))throw std::runtime_error("MMQ path/type unavailable");
    auto w=weights(type,n,k,991+type);auto wf=dequant(type,w.data(),n,k),x=activations(t*k,23);
    Buffer dw(w.size()+4096),dx(x.size()*4),dq(mq::q8_bytes(t,k)),dy((size_t)t*n*4),bounds(8),ids(t*4);
    HIP(hipMemset(dw.p,0,w.size()+4096));HIP(hipMemcpy(dw.p,w.data(),w.size(),hipMemcpyHostToDevice));HIP(hipMemcpy(dx.p,x.data(),x.size()*4,hipMemcpyHostToDevice));
    int b[2]={0,t};HIP(hipMemcpy(bounds.p,b,8,hipMemcpyHostToDevice));mq::iota(ids.as<int32_t>(),t,s);
    mq::quantize(dx.as<float>(),nullptr,dq.p,type,k,k,t,s);
    mq::Product p;p.w=dw.p;p.type=type;p.w_rows=n;p.w_cols=k;p.expert_bytes=w.size();p.n=1;p.xq=dq.p;p.bounds=bounds.as<int32_t>();p.ids=ids.as<int32_t>();p.total_rows=t;p.max_rows=t;p.dst=dy.as<float>();p.ld_dst=n;
    mq::Context ctx;auto op=[&]{ctx.run(p,s);};op();HIP(hipStreamSynchronize(s));
    std::vector<float> got((size_t)t*n);HIP(hipMemcpy(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost));
    for(float v:got)if(!std::isfinite(v))throw std::runtime_error("nonfinite MMQ");
    double ne=0,de=0;for(size_t q=0;q<64;++q){size_t i=q*(got.size()-1)/63;int c=i/n,r=i%n;double z=0;for(int j=0;j<k;++j)z+=(double)x[(size_t)c*k+j]*wf[(size_t)r*k+j];ne+=(got[i]-z)*(got[i]-z);de+=z*z;}
    double e=std::sqrt(ne/(de+1e-30));if(e>2e-2)throw std::runtime_error("MMQ double oracle mismatch "+std::to_string(e));
    double us=bench(s,op,20);std::printf("RESULT kind=mmq dtype=%s T=%d N=%d K=%d us=%.6f effective_TFLOPS=%.6f sampled_rel2=%.8g\n",ggml_type_name((ggml_type)type),t,n,k,us,2.0*t*n*k/us/1e6,e);
}
int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    try{
        hipDeviceProp_t p{};HIP(hipGetDeviceProperties(&p,0));std::printf("device=%s arch=%s wave=%d\n",p.name,p.gcnArchName,p.warpSize);
        hipStream_t s;HIP(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));std::string mode=argc>1?argv[1]:"mmvq";
        if(mode=="mmvq"){
            for(int type:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ2_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ4_XS,GGML_TYPE_IQ4_NL,GGML_TYPE_Q2_0})for(int t:{1,4,8})mmvq(s,type,640,2560,t);
            for(int type:{GGML_TYPE_IQ4_NL,GGML_TYPE_Q2_0})for(int t:{1,4,8})mmvq(s,type,2560,640,t);
        }else if(mode=="expert"){
            for(int type:{GGML_TYPE_IQ2_XS,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S})for(int t:{1,4})expert(s,type,t);
            // The actual GSQ-RCO IQ2_XS file mixes these gate/up types with Q2_0 down.
            for(int type:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ1_M})for(int t:{1,4})expert(s,type,t,GGML_TYPE_Q2_0);
        }else if(mode=="bf16"||mode=="f16"){
            for(int t:{1,32,128,512})gemm(s,mode=="bf16",t,2560,2560);
        }else if(mode=="mmq"){
            for(int type:{GGML_TYPE_IQ2_XS,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_NL,GGML_TYPE_Q2_0})for(int t:{32,128,512})mmq(s,type,640,2560,t);
            for(int type:{GGML_TYPE_IQ4_NL,GGML_TYPE_Q2_0})for(int t:{32,128,512})mmq(s,type,2560,640,t);
        }else throw std::runtime_error("mode must be mmvq/expert/bf16/f16/mmq");
        HIP(hipStreamDestroy(s));std::puts("PASS: independent synthetic operator oracle (no real model)");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
