#include "strata/prefill/gfx906_diagnostic.hpp"
#include "strata/prefill/gfx906_policy.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <stdexcept>

namespace strata::prefill::gfx906 {
namespace {
namespace fs = std::filesystem;
constexpr uint64_t reserve = 4ULL << 30;
[[maybe_unused]] constexpr uint64_t fixture_limit = 64ULL << 20;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
fs::path private_root(const char* name, uint64_t allowance) {
    const char* value = std::getenv(name);
    if (!value || !*value) throw std::runtime_error(std::string(name) + " requires an existing private directory");
    const fs::path p(value);
    if (!fs::is_directory(p) || fs::is_symlink(p) ||
        (fs::status(p).permissions() & (fs::perms::group_all | fs::perms::others_all)) != fs::perms::none)
        throw std::runtime_error("diagnostic directory must be private and not a symlink");
    if (fs::space(p).available < reserve + allowance) throw std::runtime_error("diagnostic would violate disk reserve");
    return p;
}
[[maybe_unused]] int64_t position() {
    const char* v = std::getenv("STRATA_GFX906_QSA_CAPTURE_POS");
    if (!v || !*v) throw std::runtime_error("QSA capture requires an exact nonnegative chunk position");
    for (const char* p = v; *p; ++p) if (*p < '0' || *p > '9') throw std::runtime_error("invalid QSA capture position");
    char* end = nullptr; errno = 0; const long long n = std::strtoll(v, &end, 10);
    if (errno || *end || n < 0) throw std::runtime_error("invalid QSA capture position");
    return n;
}
void save_bytes(const fs::path& p, const void* data, size_t bytes) {
    std::FILE* f = std::fopen(p.string().c_str(), "wbx");
    if (!f) throw std::runtime_error("cannot exclusively create diagnostic " + p.string());
    try {
        fs::permissions(p, fs::perms::owner_read | fs::perms::owner_write);
        if (std::fwrite(data, 1, bytes, f) != bytes) throw std::runtime_error("short diagnostic write");
    } catch (...) { std::fclose(f); throw; }
    if (std::fclose(f)) throw std::runtime_error("cannot close diagnostic");
}
template <typename T> void save(const fs::path& p, const std::vector<T>& v) { save_bytes(p, v.data(), v.size() * sizeof(T)); }
template <typename T> void copy(T* host, const T* device, size_t n, cudaStream_t s) {
    if (n) ck(cudaMemcpyAsync(host, device, n * sizeof(T), cudaMemcpyDeviceToHost, s));
}
} // namespace

bool capture_attention(const float* q, const kernels::QsaAttnPools& p, const int32_t* ids,
                       const int32_t* steps, int64_t cap, const kernels::QsaShapes& shape,
                       int64_t queries, int layer, int64_t pos0, void* stream,
                       AttentionCapture& capture, std::string& error) {
    capture = {};
    if (!opt_in(std::getenv("STRATA_GFX906_QSA_CAPTURE"))) return true;
    try {
#if defined(STRATA_USE_HIP) && defined(STRATA_EXPERIMENTAL_GFX906)
        if (pos0 != position()) return true;
        const fs::path root = private_root("STRATA_GFX906_QSA_CAPTURE_DIR", fixture_limit);
        int device = 0; ck(cudaGetDevice(&device)); hipDeviceProp_t prop{}; ck(hipGetDeviceProperties(&prop, device));
        if (!architecture(prop.gcnArchName) || prop.warpSize != 64 || queries < 1 || queries > 65535 ||
            cap < 1 || cap > 32768 || shape.n_head != 24 || shape.n_head_kv != 2 || shape.head_dim != 256 ||
            shape.page_size < 1 || shape.page_size > 4096 || !q || !ids || !steps || !p.page_table ||
            !p.k_q || !p.v_q || !p.k_scale || !p.v_scale || p.k_q4 || p.v_q4)
            throw std::runtime_error("real-input capture requires admitted gfx906 INT8 QSA geometry/pools");
        capture.rows = {0, queries / 2, queries - 1};
        std::sort(capture.rows.begin(), capture.rows.end());
        capture.rows.erase(std::unique(capture.rows.begin(), capture.rows.end()), capture.rows.end());
        const size_t nq = capture.rows.size(); const cudaStream_t s = (cudaStream_t) stream;
        std::vector<float> hq(nq * 24 * 256);
        std::vector<int32_t> hi(nq * (size_t) cap), hs(nq * kernels::kStepCount);
        for (size_t i = 0; i < nq; ++i) {
            copy(hq.data() + i * 24 * 256, q + capture.rows[i] * 24 * 256, 24 * 256, s);
            copy(hi.data() + i * cap, ids + capture.rows[i] * cap, (size_t) cap, s);
            copy(hs.data() + i * kernels::kStepCount, steps + capture.rows[i] * kernels::kStepCount, kernels::kStepCount, s);
        }
        ck(cudaStreamSynchronize(s));
        int64_t logical_pages = 0;
        for (size_t i = 0; i < nq; ++i) {
            const int width = hs[i * kernels::kStepCount + kernels::kStepWidth];
            if (width < 0 || width > cap) throw std::runtime_error("invalid real selection width");
            for (int j = 0; j < width; ++j) if (hi[i * cap + j] >= 0)
                logical_pages = std::max<int64_t>(logical_pages, hi[i * cap + j] / shape.page_size + 1);
        }
        if (logical_pages > 65536) throw std::runtime_error("selection exceeds diagnostic page-table bound");
        std::vector<int32_t> table((size_t) logical_pages);
        copy(table.data(), p.page_table, table.size(), s); ck(cudaStreamSynchronize(s));
        std::map<int32_t, int32_t> physical;
        std::vector<int32_t> compact(table.size(), -1);
        for (size_t i = 0; i < nq; ++i) {
            const int width = hs[i * kernels::kStepCount + kernels::kStepWidth];
            for (int j = 0; j < width; ++j) {
                const int cell = hi[i * cap + j]; if (cell < 0) continue;
                const int lp = cell / shape.page_size, pp = table[(size_t) lp]; if (pp < 0) continue;
                if (pp > 65535) throw std::runtime_error("physical page exceeds diagnostic bound");
                if (!physical.count(pp)) physical[pp] = (int32_t) physical.size();
                compact[(size_t) lp] = physical[pp];
            }
        }
        if (physical.empty()) throw std::runtime_error("real-input capture has no live pages");
        const size_t page_values = (size_t) 2 * shape.page_size * 256;
        const size_t values = physical.size() * page_values;
        if (values * 2 + values / 64 * 4 + hq.size() * 8 + hi.size() * 4 + hs.size() * 4 +
            compact.size() * 4 + 9 * 8 + capture.rows.size() * 8 > fixture_limit)
            throw std::runtime_error("real-input fixture exceeds 64-MiB allowance");
        std::vector<int8_t> hk(values), hv(values);
        std::vector<uint16_t> hks(values / 64), hvs(values / 64);
        for (const auto& entry : physical) {
            const size_t from = (size_t) entry.first * page_values, to = (size_t) entry.second * page_values;
            copy(hk.data() + to, p.k_q + from, page_values, s); copy(hv.data() + to, p.v_q + from, page_values, s);
            copy(hks.data() + to / 64, p.k_scale + from / 64, page_values / 64, s);
            copy(hvs.data() + to / 64, p.v_scale + from / 64, page_values / 64, s);
        }
        ck(cudaStreamSynchronize(s));
        const fs::path path = root / ("device" + std::to_string(device) + "-layer" + std::to_string(layer) + "-pos" + std::to_string(pos0));
        if (!fs::create_directory(path)) throw std::runtime_error("real-input capture already exists");
        fs::permissions(path, fs::perms::owner_all);
        const std::vector<int64_t> geometry = {1, (int64_t) nq, cap, shape.page_size, (int64_t) physical.size(),
                                               logical_pages, layer, pos0, queries};
        save(path / "geometry.i64", geometry); save(path / "query-rows.i64", capture.rows);
        save(path / "q.f32", hq); save(path / "k.i8", hk); save(path / "v.i8", hv);
        save(path / "k-scale.f16", hks); save(path / "v-scale.f16", hvs);
        save(path / "ids.i32", hi); save(path / "steps.i32", hs); save(path / "pages.i32", compact);
        capture.directory = path.string();
        std::fprintf(stderr, "strata QSA diagnostic: device=%d layer=%d pos=%lld samples=%zu compact_pages=%zu (NOT performance)\n",
                     device, layer, (long long) pos0, nq, physical.size());
#else
        (void) q; (void) p; (void) ids; (void) steps; (void) cap; (void) shape; (void) queries;
        (void) layer; (void) pos0; (void) stream;
        throw std::runtime_error("QSA capture requires an experimental HIP build");
#endif
        return true;
    } catch (const std::exception& e) { error = std::string("QSA capture: ") + e.what(); return false; }
}
bool finish_attention(const AttentionCapture& capture, const float* attn, void* stream, std::string& error) {
    if (capture.directory.empty()) return true;
    try {
        std::vector<float> out(capture.rows.size() * 24 * 256); const cudaStream_t s = (cudaStream_t) stream;
        for (size_t i = 0; i < capture.rows.size(); ++i)
            copy(out.data() + i * 24 * 256, attn + capture.rows[i] * 24 * 256, 24 * 256, s);
        ck(cudaStreamSynchronize(s)); save(fs::path(capture.directory) / "actual.f32", out); return true;
    } catch (const std::exception& e) { error = std::string("QSA capture output: ") + e.what(); return false; }
}
bool capture_logits(int64_t index, int64_t pos, int32_t token, const std::vector<float>& logits, std::string& error) {
    if (!opt_in(std::getenv("STRATA_GFX906_QSA_LOGITS"))) return true;
    try {
        if (index < 0 || pos < 0 || logits.empty() || logits.size() > 1000000) throw std::runtime_error("invalid logits geometry");
        for (float v : logits) if (!std::isfinite(v)) throw std::runtime_error("nonfinite diagnostic logits");
        const fs::path root = private_root("STRATA_GFX906_QSA_LOGITS_DIR", logits.size() * sizeof(float) + 256);
        const std::string name = "prediction" + std::to_string(index);
        const std::vector<int64_t> meta = {1, index, pos, token, (int64_t) logits.size()};
        save(root / (name + ".i64"), meta); save(root / (name + ".f32"), logits); return true;
    } catch (const std::exception& e) { error = std::string("QSA logits: ") + e.what(); return false; }
}
} // namespace strata::prefill::gfx906
