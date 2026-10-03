// Opt-in grouped gfx906 MMQ tile experiment. Synthetic operator, not model parity/speed.
// Full output comparison against unchanged GGML dispatch; independent pinned CPU dequantizer
// + double products sample up to 16 elements (4 rows x 4 columns) per expert. Q8 rounding is intentional.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/gfx906_policy.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mq = strata::prefill::mmq;
namespace fs = std::filesystem;
static void ck(hipError_t e, const char* s) {
    if (e != hipSuccess) throw std::runtime_error(std::string(s)+": "+hipGetErrorString(e));
}
#define HIP(x) ck((x), #x)
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { HIP(hipMalloc(&p,n)); }
    ~Buffer() { if(p) hipFree(p); }
    Buffer(const Buffer&)=delete;
    template<class T> T* as() { return static_cast<T*>(p); }
};
template<class T> static void save(const fs::path& path, const std::vector<T>& data) {
    if (fs::exists(path)) throw std::runtime_error("fixture exists: "+path.string());
    std::ofstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("open fixture: "+path.string());
    f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!f) throw std::runtime_error("write fixture: "+path.string());
}
static std::vector<uint8_t> weights(ggml_type t, int n, int k, int groups) {
    const auto* traits = ggml_get_type_traits(t);
    const size_t eb = ggml_row_size(t,k)*n;
    std::vector<uint8_t> w(eb*groups+4096,0);
    std::mt19937 rng(991+(int)t);
    for(size_t i=0;i<eb*groups;++i) w[i]=(uint8_t)rng();
    for(size_t i=0;i<eb*groups;i+=traits->type_size) {
        w[i]=0;w[i+1]=(i/traits->type_size)%5==0 ? 0x98 : 0x18; // finite signed f16 scales
    }
    return w;
}
static double median_us(hipStream_t s, const std::function<void()>& op) {
    for(int i=0;i<3;++i)op();HIP(hipStreamSynchronize(s));
    hipEvent_t a,b;HIP(hipEventCreate(&a));HIP(hipEventCreate(&b));
    std::vector<float> samples;
    for(int i=0;i<9;++i) {
        HIP(hipEventRecord(a,s));for(int j=0;j<10;++j)op();HIP(hipEventRecord(b,s));HIP(hipEventSynchronize(b));
        float ms=0;HIP(hipEventElapsedTime(&ms,a,b));if(i>1)samples.push_back(ms*100);
    }
    std::sort(samples.begin(),samples.end());HIP(hipEventDestroy(a));HIP(hipEventDestroy(b));
    return samples[samples.size()/2];
}
static void product(hipStream_t stream, const fs::path& root, ggml_type type, int n, int k, int mean, bool benchmark) {
    constexpr int groups=32;
    const std::string label=std::string(ggml_type_name(type))+"-n"+std::to_string(n)+"-r"+std::to_string(mean);
    const auto dir=root/label;fs::create_directory(dir);
    std::vector<int32_t> counts(groups),bounds(groups+1),src,dst;
    for(int e=0;e<groups;++e) counts[e]=e==0?0:e==1?1:std::max(1,mean+(e%7-3)*mean/4);
    for(int e=0;e<groups;++e)bounds[e+1]=bounds[e]+counts[e];
    const int rows=bounds.back();src.resize(rows);dst.resize(rows);
    for(int i=0;i<rows;++i){src[i]=(rows-1-i+17)%rows;dst[i]=(i+29)%rows;}
    auto w=weights(type,n,k,groups);
    std::vector<float> x((size_t)rows*k);
    for(int r=0;r<rows;++r) for(int c=0;c<k;++c)
        x[(size_t)r*k+c]=r==rows-1?0.0f:0.19f*std::sin(0.019f*c+0.31f*r)+0.03f*((r+c)%7-3);
    const size_t eb=mq::matrix_bytes(type,n,k);
    // Save inputs BEFORE allocating, launching or comparing, including failing fixtures.
    save(dir/"weights.bin",w);save(dir/"activations.f32",x);save(dir/"bounds.i32",bounds);save(dir/"source.i32",src);save(dir/"destination.i32",dst);
    { std::ofstream f(dir/"shape.json");f<<"{\"type\":"<<(int)type<<",\"weight_rows\":"<<n<<",\"weight_cols\":"<<k
      <<",\"groups\":"<<groups<<",\"rows\":"<<rows<<",\"expert_bytes\":"<<eb
      <<",\"oracle\":\"pinned CPU dequantizer / double products, sampled per expert\",\"activation_rounding\":\"MMQ Q8_1\"}\n"; }
    Buffer dw(w.size()),dx(x.size()*4),dsrc(rows*4),ddst(rows*4),db(bounds.size()*4),dq(mq::q8_bytes(rows,k)),dy((size_t)rows*n*4);
    HIP(hipMemcpy(dw.p,w.data(),w.size(),hipMemcpyHostToDevice));HIP(hipMemcpy(dx.p,x.data(),x.size()*4,hipMemcpyHostToDevice));
    HIP(hipMemcpy(dsrc.p,src.data(),rows*4,hipMemcpyHostToDevice));HIP(hipMemcpy(ddst.p,dst.data(),rows*4,hipMemcpyHostToDevice));
    HIP(hipMemcpy(db.p,bounds.data(),bounds.size()*4,hipMemcpyHostToDevice));
    mq::quantize(dx.as<float>(),dsrc.as<int32_t>(),dq.p,type,k,k,rows,stream);HIP(hipStreamSynchronize(stream));
    const auto* traits=ggml_get_type_traits(type);if(!traits->to_float)throw std::runtime_error("CPU dequantizer missing");
    std::vector<float> expected((size_t)groups*16,0);std::vector<size_t> sampled;
    std::vector<float> wr(k);
    const size_t rb=ggml_row_size(type,k);
    for(int e=0;e<groups;++e) if(counts[e]) for(int a=0;a<4;++a) for(int b=0;b<4;++b) {
        const int r=bounds[e]+a*(counts[e]-1)/3,c=b*(n-1)/3;
        traits->to_float(w.data()+e*eb+c*rb,wr.data(),k);
        double acc=0;for(int q=0;q<k;++q)acc+=(double)wr[q]*x[(size_t)src[r]*k+q];
        expected[sampled.size()]=(float)acc;sampled.push_back((size_t)dst[r]*n+c);
    }
    expected.resize(sampled.size());
    mq::Product p;p.w=dw.p;p.type=type;p.w_rows=n;p.w_cols=k;p.expert_bytes=eb;p.n=groups;
    p.xq=dq.p;p.bounds=db.as<int32_t>();p.ids=ddst.as<int32_t>();p.total_rows=rows;
    p.max_rows=*std::max_element(counts.begin(),counts.end());p.dst=dy.as<float>();p.ld_dst=n;p.layer=0;p.pos0=0;p.group_rows=rows;
    std::vector<float> reference;
    for(const char* tile:{"0","16","32","48","64"}) {
        setenv("STRATA_GFX906_MMQ_J",tile,1);mq::Context ctx;
        auto run=[&]{ctx.run(p,stream);};
        HIP(hipMemset(dy.p,0xff,(size_t)rows*n*4));run();HIP(hipStreamSynchronize(stream));
        std::vector<float> got((size_t)rows*n);HIP(hipMemcpy(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost));
        save(dir/(std::string("output-j")+tile+".f32"),got);
        double error2=0,ref2=0;size_t differences=0;
        for(float v:got)if(!std::isfinite(v))throw std::runtime_error(label+": nonfinite/unwritten output");
        const size_t zero_source=std::find(src.begin(),src.end(),rows-1)-src.begin();
        for(int c=0;c<n;++c)if(got[(size_t)dst[zero_source]*n+c]!=0.0f)throw std::runtime_error(label+": all-zero input row failed");
        for(size_t i=0;i<sampled.size();++i){double d=got[sampled[i]]-expected[i];error2+=d*d;ref2+=(double)expected[i]*expected[i];}
        const double relative=std::sqrt(error2/(ref2+1e-30));
        if(relative>2e-2)throw std::runtime_error(label+": independent sampled oracle failed");
        if(reference.empty())reference=got;
        else for(size_t i=0;i<got.size();++i)if(std::memcmp(&got[i],&reference[i],4))++differences;
        const double us=benchmark?median_us(stream,run):0;
        std::printf("RESULT type=%s N=%d K=%d mean=%d max=%lld groups=%d requested_j=%s median_us=%.6f sampled_rel2=%.9g bit_differences=%zu\n",
                    ggml_type_name(type),n,k,mean,(long long)p.max_rows,groups,tile,us,relative,differences);
        if(differences)throw std::runtime_error(label+": tile output differs from unchanged control; not accepted");
    }
}
int main(int argc,char**argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        int device=-1;fs::path dump;bool benchmark=false;
        for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--device"&&i+1<argc)device=std::stoi(argv[++i]);else if(a=="--dump-dir"&&i+1<argc)dump=argv[++i];else if(a=="--bench")benchmark=true;else throw std::runtime_error("unknown/missing argument");}
        if(device<0||device>1||dump.empty()||fs::exists(dump))throw std::runtime_error("explicit device 0/1 and NEW dump directory required");
        fs::create_directories(dump);int count=0;HIP(hipGetDeviceCount(&count));if(device>=count)throw std::runtime_error("requested card missing, not a skip");
        HIP(hipSetDevice(device));hipDeviceProp_t prop{};HIP(hipGetDeviceProperties(&prop,device));
        if(!strata::prefill::gfx906::architecture(prop.gcnArchName)||prop.warpSize!=64)throw std::runtime_error("actual gfx906 wave64 required");
        std::printf("DEVICE ordinal=%d name=%s arch=%s wave=%d\n",device,prop.name,prop.gcnArchName,prop.warpSize);
        hipStream_t stream;HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        for(auto t:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS})
            for(int mean:{40,80,160})product(stream,dump,t,1280,2560,mean,benchmark);
        for(auto t:{GGML_TYPE_Q2_0,GGML_TYPE_IQ4_NL})for(int mean:{40,80,160})product(stream,dump,t,2560,640,mean,benchmark);
        HIP(hipStreamDestroy(stream));std::puts("PASS: full tile/control outputs + sampled CPU double oracle; not model parity or tok/s");return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
