// Independent f16-dequantization/double-accumulation oracle; this is NOT model parity or model throughput.
#include "strata/kernels/gfx906_prompt_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/prefill/gfx906_policy.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::kernels;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <typename T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc((void**) &p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    std::vector<T> read(size_t n) const {
        std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { ck(cudaStreamCreate(&s)); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
};
// No GPU dequantizer, half conversion, or production softmax in this oracle.
double half(uint16_t x) {
    const int sign = x >> 15, exp = (x >> 10) & 31, mant = x & 1023;
    const double v = exp == 0 ? std::ldexp((double) mant, -24) : std::ldexp(1.0 + mant / 1024.0, exp - 15);
    return sign ? -v : v;
}
struct Fixture {
    int queries, cap, pages, mode;
    QsaShapes s = qsa_real_shapes();
    std::vector<float> q;
    std::vector<int8_t> k, v;
    std::vector<uint16_t> ks, vs;
    std::vector<int32_t> ids, steps, table;
    Fixture(int nq, int capacity, int masking, bool full = false)
        : queries(nq), cap(capacity), pages(32), mode(masking) {
        s.page_size = 256;
        std::mt19937 rng(906u + (unsigned) cap * 31u + (unsigned) mode * 13u);
        q.resize((size_t) nq * 24 * 256);
        k.resize((size_t) pages * 2 * s.page_size * 256); v.resize(k.size());
        ks.resize(k.size() / 64); vs.resize(ks.size());
        ids.resize((size_t) nq * cap, -1); steps.resize((size_t) nq * kStepCount, 0); table.resize(pages);
        for (auto& x : q) x = (float) ((int) (rng() % 65537) - 32768) / 65536.0f;
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = (int8_t) ((int) (rng() % 256) - 128);
            v[i] = (int8_t) ((int) (rng() % 256) - 128);
        }
        for (size_t i = 0; i < ks.size(); ++i) {
            ks[i] = i % 23 == 0 ? 1 : (uint16_t) (((10 + rng() % 4) << 10) | (rng() % 1024));
            vs[i] = (uint16_t) (((10 + rng() % 4) << 10) | (rng() % 1024));
        }
        for (int i = 0; i < pages; ++i) table[i] = mode == 2 || (mode == 1 && i % 3 == 0) ? -1 : pages - 1 - i;
        const int widths[] = {0, 1, 63, 65, 257, cap};
        for (int i = 0; i < nq; ++i) {
            const int width = full ? cap : std::min(cap, widths[i % 6]);
            steps[(size_t) i * kStepCount + kStepWidth] = width;
            for (int j = 0; j < width; ++j) ids[(size_t) i * cap + j] = (i * 43 + j * 97) % (pages * s.page_size);
        }
        if (mode == 3) for (auto& x : q) x *= 64.0f; // Numerically demanding, sharply peaked softmax.
    }
    std::vector<double> oracle(int n) const {
        std::vector<double> out((size_t) n * 24 * 256, 0.0), scores(cap);
        for (int qi = 0; qi < n; ++qi) for (int h = 0; h < 24; ++h) {
            const int width = steps[(size_t) qi * kStepCount + kStepWidth];
            double maximum = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < width; ++j) {
                const int cell = ids[(size_t) qi * cap + j], page = table[cell / s.page_size];
                double dot = 0;
                if (page < 0) { scores[j] = -std::numeric_limits<double>::infinity(); continue; }
                const size_t row = ((size_t) page * 2 + h / 12) * s.page_size + cell % s.page_size;
                for (int d = 0; d < 256; ++d) dot += (double) q[((size_t) qi * 24 + h) * 256 + d] *
                                                                  (double) k[row * 256 + d] * half(ks[row * 4 + d / 64]);
                scores[j] = dot / 16.0; maximum = std::max(maximum, scores[j]);
            }
            if (!std::isfinite(maximum)) continue;
            double sum = 0;
            for (int j = 0; j < width; ++j) {
                const int cell = ids[(size_t) qi * cap + j], page = table[cell / s.page_size];
                if (page < 0) continue;
                const double weight = std::exp(scores[j] - maximum); sum += weight;
                const size_t row = ((size_t) page * 2 + h / 12) * s.page_size + cell % s.page_size;
                for (int d = 0; d < 256; ++d) out[((size_t) qi * 24 + h) * 256 + d] +=
                                                   weight * (double) v[row * 256 + d] * half(vs[row * 4 + d / 64]);
            }
            for (int d = 0; d < 256; ++d) out[((size_t) qi * 24 + h) * 256 + d] /= sum;
        }
        return out;
    }
};
template <typename A, typename B> double compare(const std::vector<A>& out, const std::vector<B>& ref, const char* name) {
    if (out.size() < ref.size()) throw std::runtime_error("short output");
    double max_abs = 0, err2 = 0, ref2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double delta = std::abs((double) out[i] - (double) ref[i]);
        if (!std::isfinite((double) out[i]) || delta > 0.003 + 0.0005 * std::abs((double) ref[i])) {
            std::fprintf(stderr, "%s mismatch at %zu: %.9g vs %.9g\n", name, i, (double) out[i], (double) ref[i]);
            throw std::runtime_error("attention numerical mismatch");
        }
        max_abs = std::max(max_abs, delta); err2 += delta * delta; ref2 += (double) ref[i] * (double) ref[i];
    }
    const double relative = std::sqrt(err2 / std::max(ref2, 1e-30));
    if (relative > 0.0002) throw std::runtime_error(std::string(name) + " relative L2 exceeds 2e-4");
    std::printf("%s max_abs=%.9g relative_l2=%.9g\n", name, max_abs, relative);
    return max_abs;
}
template <typename T> void save(const std::filesystem::path& path, const std::vector<T>& v) {
    if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite a fixture");
    std::ofstream f(path, std::ios::binary); f.write((const char*) v.data(), (std::streamsize) (v.size() * sizeof(T)));
    if (!f) throw std::runtime_error("cannot save fixture");
}
void run_case(Fixture& f, bool graph, bool bench, const std::filesystem::path& dump) {
    const int oracle_queries = bench ? std::min(8, f.queries) : f.queries;
    std::filesystem::path path;
    if (!dump.empty()) {
        path = dump / ("q" + std::to_string(f.queries) + "-cap" + std::to_string(f.cap) + "-mask" + std::to_string(f.mode));
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("fixture directory already exists");
        // Preserve inputs BEFORE allocation/launch/comparison, including on a failing fixture.
        save(path / "q.f32", f.q); save(path / "k.i8", f.k); save(path / "v.i8", f.v);
        save(path / "k-scale.f16", f.ks); save(path / "v-scale.f16", f.vs);
        save(path / "ids.i32", f.ids); save(path / "steps.i32", f.steps); save(path / "pages.i32", f.table);
        std::ofstream meta(path / "shape.json");
        meta << "{\"queries\":" << f.queries << ",\"cap\":" << f.cap << ",\"mode\":" << f.mode
             << ",\"page_size\":256,\"physical_pages\":32,\"head_dim\":256,\"query_heads\":24,\"kv_heads\":2,"
                "\"scale_group\":64,\"step_fields\":" << kStepCount << ",\"width_field\":" << kStepWidth
             << ",\"oracle_queries\":" << oracle_queries << "}\n";
        if (!meta) throw std::runtime_error("cannot save fixture geometry");
    }
    const size_t size = f.q.size();
    Device<float> q(f.q), control(size), result(size), scratch((size_t) std::min(32, f.queries) * qsa_decode_attn_scratch_floats(f.cap, f.s));
    Device<int8_t> k(f.k), v(f.v); Device<uint16_t> ks(f.ks), vs(f.vs);
    Device<int32_t> ids(f.ids), steps(f.steps), table(f.table);
    QsaAttnPools p; p.k_q = k.p; p.v_q = v.p; p.k_scale = ks.p; p.v_scale = vs.p; p.page_table = table.p;
    Stream stream;
    auto old_path = [&] {
        for (int start = 0; start < f.queries; start += 32)
            qsa_decode_attn_batch(q.p + (size_t) start * 24 * 256, p, ids.p + (size_t) start * f.cap,
                                  steps.p + (size_t) start * kStepCount, f.cap, f.s, scratch.p,
                                  control.p + (size_t) start * 24 * 256, std::min(32, f.queries - start), stream.s);
    };
    auto new_path = [&] {
        if (!gfx906_prompt_attn_batch(q.p, p, ids.p, steps.p, f.cap, f.s, result.p, f.queries, stream.s))
            throw std::runtime_error("opt-in path unexpectedly refused the real fixture");
    };
    unsetenv("STRATA_GFX906_PREFILL_ATTN");
    if (gfx906_prompt_attn_batch(q.p, p, ids.p, steps.p, f.cap, f.s, result.p, f.queries, stream.s))
        throw std::runtime_error("attention enabled by default");
    setenv("STRATA_GFX906_PREFILL_ATTN", "1", 1);
    QsaAttnPools incomplete = p; incomplete.v_scale = nullptr;
    if (gfx906_prompt_attn_batch(q.p, incomplete, ids.p, steps.p, f.cap, f.s, result.p, f.queries, stream.s))
        throw std::runtime_error("incomplete pools accepted");
    QsaShapes wrong = f.s; wrong.head_dim = 128;
    if (gfx906_prompt_attn_batch(q.p, p, ids.p, steps.p, f.cap, wrong, result.p, f.queries, stream.s))
        throw std::runtime_error("wrong geometry accepted");
    ck(cudaMemset(result.p, 0x7f, size * sizeof(float)));
    old_path(); new_path(); ck(cudaStreamSynchronize(stream.s));
    const auto ref = f.oracle(oracle_queries);
    const auto old = control.read(size);
    const auto out = result.read(size);
    if (!path.empty()) {
        save(path / "oracle.f64", ref); save(path / "control.f32", old); save(path / "online.f32", out);
    }
    std::printf("fixture queries=%d cap=%d mask=%d oracle_queries=%d\n", f.queries, f.cap, f.mode, oracle_queries);
    compare(old, ref, "control/oracle"); compare(out, ref, "online/oracle"); compare(out, old, "online/control");
    // Also exercise the actual prompt dispatcher, not only its direct experimental entry point.
    if (!qsa_prompt_attn_batch(q.p, p, ids.p, steps.p, f.cap, f.s, result.p, f.queries, stream.s))
        throw std::runtime_error("prompt dispatcher did not select gfx906");
    ck(cudaStreamSynchronize(stream.s)); compare(result.read(size), old, "dispatcher/control");
    if (graph) {
        cudaGraph_t g = nullptr; cudaGraphExec_t exec = nullptr;
        ck(cudaStreamBeginCapture(stream.s, cudaStreamCaptureModeThreadLocal)); new_path();
        ck(cudaStreamEndCapture(stream.s, &g)); ck(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
        ck(cudaGraphLaunch(exec, stream.s)); ck(cudaStreamSynchronize(stream.s));
        compare(result.read(size), ref, "graph/oracle");
        ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(g));
    }
    if (bench) {
        cudaEvent_t begin = nullptr, end = nullptr; ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
        auto measure = [&](bool online) {
            std::vector<float> times;
            for (int i = 0; i < 7; ++i) {
                ck(cudaEventRecord(begin, stream.s)); if (online) new_path(); else old_path();
                ck(cudaEventRecord(end, stream.s)); ck(cudaEventSynchronize(end));
                float ms = 0; ck(cudaEventElapsedTime(&ms, begin, end)); if (i >= 2) times.push_back(ms);
            }
            std::sort(times.begin(), times.end()); return times[times.size() / 2];
        };
        const float old_ms = measure(false), new_ms = measure(true);
        std::printf("operator_only queries=%d cap=%d control32_median_ms=%.6f online_median_ms=%.6f ratio=%.4f (NOT model tok/s)\n",
                    f.queries, f.cap, old_ms, new_ms, old_ms / new_ms);
        ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        int device = 0; bool bench = false; std::filesystem::path dump;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
            else if (a == "--dump-dir" && i + 1 < argc) dump = argv[++i];
            else if (a == "--bench") bench = true;
            else throw std::runtime_error("usage: hip_gfx906_prefill_attn [--device N] [--dump-dir NEW_DIR] [--bench]");
        }
        int count = 0; ck(cudaGetDeviceCount(&count));
        if (device < 0 || device >= count) throw std::runtime_error("requested test card is missing; this is NOT a pass or skip");
        ck(cudaSetDevice(device)); cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, device));
        if (!strata::prefill::gfx906::architecture(prop.gcnArchName) || prop.warpSize != 64)
            throw std::runtime_error("test requires a real gfx906 wave64 card");
        if (!dump.empty() && !std::filesystem::create_directory(dump)) throw std::runtime_error("dump directory already exists");
        std::printf("device=%d arch=%s wave=%d independent QSA operator test\n", device, prop.gcnArchName, prop.warpSize);
        struct Case { int queries, cap, mask; };
        const Case cases[] = {{1,1,0}, {3,63,0}, {17,64,0}, {33,65,1}, {8,257,1}, {8,2051,0}, {8,2051,2}, {8,2051,3}, {128,65,0}};
        for (const auto& c : cases) { Fixture f(c.queries, c.cap, c.mask); run_case(f, c.cap == 2051 && c.mask == 0, false, dump); }
        if (bench) { Fixture f(2048, 2051, 0, true); run_case(f, false, true, dump); }
        std::printf("PASS device=%d; synthetic parity only; bench oracle is sampled if requested\n", device);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
