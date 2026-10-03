// Two-card mapped NativeEmbed gather/graph replay. No peer access and no model parity claim.
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/prefill/gfx906_policy.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <typename T> void scalar(std::ofstream& f, T x) { f.write((const char*) &x, sizeof(x)); }
void string(std::ofstream& f, const std::string& x) { scalar(f, (uint64_t) x.size()); f.write(x.data(), x.size()); }
int8_t code(int row, int dim) { return (int8_t) ((row * 37 + dim * 13) % 255 - 127); }
void fixture(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    scalar(f, (uint32_t) 0x46554747); scalar(f, (uint32_t) 3);
    scalar(f, (uint64_t) 1); scalar(f, (uint64_t) 1);
    string(f, "general.architecture"); scalar(f, (uint32_t) 8); string(f, "strata-embd");
    string(f, "token_embd.weight"); scalar(f, (uint32_t) 2);
    scalar(f, (uint64_t) 256); scalar(f, (uint64_t) 3);
    scalar(f, (uint32_t) 8); scalar(f, (uint64_t) 0); // GGML Q8_0; no ggml writer/dequantizer in the oracle.
    while ((uint64_t) f.tellp() % 32) scalar(f, (uint8_t) 0);
    for (int row = 0; row < 3; ++row) for (int block = 0; block < 8; ++block) {
        scalar(f, (uint16_t) 0x3c00); // f16 scale 1; expected output is exactly the signed code.
        for (int d = 0; d < 32; ++d) scalar(f, code(row, block * 32 + d));
    }
    if (!f) throw std::runtime_error("cannot preserve synthetic GGUF fixture");
}
void test_card(const strata::core::NativeEmbed& embed, int device, const std::filesystem::path& dir) {
    const strata::core::OnDevice on_device(device);
    std::string error;
    if (!embed.prepare_current_device(error)) throw std::runtime_error(error);
    cudaStream_t stream = nullptr; ck(cudaStreamCreate(&stream));
    const int32_t tokens[] = {2,0,1}; int32_t* dt = nullptr; float* out = nullptr;
    ck(cudaMalloc((void**) &dt, sizeof(tokens))); ck(cudaMalloc((void**) &out, 3 * 256 * sizeof(float)));
    ck(cudaMemcpy(dt, tokens, sizeof(tokens), cudaMemcpyHostToDevice));
    embed.gather_dev(dt, 3, out, stream); ck(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); embed.gather_dev(dt, 3, out, stream);
    ck(cudaStreamEndCapture(stream, &graph)); ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    std::vector<float> result(3 * 256); ck(cudaMemcpy(result.data(), out, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int t = 0; t < 3; ++t) for (int d = 0; d < 256; ++d)
        if (result[t * 256 + d] != (float) code(tokens[t], d)) throw std::runtime_error("cross-card native embedding gather mismatch");
    std::ofstream raw(dir / ("device" + std::to_string(device) + ".f32"), std::ios::binary);
    raw.write((const char*) result.data(), result.size() * sizeof(float));
    if (!raw) throw std::runtime_error("cannot save gather output");
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaFree(dt)); ck(cudaFree(out)); ck(cudaStreamDestroy(stream));
    std::printf("PASS device=%d native mapped embedding 768 signed values + graph replay (synthetic only)\n", device);
}
} // namespace

int main(int argc, char** argv) {
    try {
        int count = 0; ck(cudaGetDeviceCount(&count));
        if (count < 2) throw std::runtime_error("two visible test cards required; a missing card is NOT a pass or skip");
        for (int d = 0; d < 2; ++d) {
            cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, d));
            if (!strata::prefill::gfx906::architecture(prop.gcnArchName)) throw std::runtime_error("test requires two real gfx906 cards");
        }
        std::filesystem::path dir;
        if (argc == 3 && std::string(argv[1]) == "--dump-dir") dir = argv[2];
        else if (argc == 1) dir = std::filesystem::temp_directory_path() /
            ("strata-native-embed-alias-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        else throw std::runtime_error("usage: hip_native_embed_alias [--dump-dir NEW_DIR]");
        if (!std::filesystem::create_directory(dir)) throw std::runtime_error("fixture directory already exists");
        const auto path = dir / "embedding.gguf"; fixture(path);
        const strata::core::OnDevice owner(0);
        strata::core::NativeEmbed embed; std::string error;
        if (!embed.load({path.string()}, 256, 3, error)) throw std::runtime_error(error);
        test_card(embed, 0, dir); test_card(embed, 1, dir);
        std::printf("raw input/output preserved in %s; no P2P enabled\n", dir.string().c_str());
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
