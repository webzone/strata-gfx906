// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with Strata's own Qwen vision tower (tools/vision/own) and
// the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the rows at the image's pad tokens and gives them their 2-D
// M-RoPE positions (see --serve GENI in generate.cpp). The opt-in gfx906 HIP build also includes the pinned mtmd reader.
//
//   strata-vision --mmproj <mmproj.gguf> [--model <text model .gguf>] [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]
// The gfx906 build also uses --tower mtmd|own|auto, --flash-attn on|off|auto and --verbose for the pinned reader.
//
// Resident: prints "READY <n_embd> <reader>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).
#include "gpu_arch.h"
#include "own_reader.h"
#ifdef STRATA_VIT_GPU
#include "vit_gpu.h"
#include "../../../engine/include/strata/core/kpack_path.hpp"   // #5060 (header only)
#endif
#ifdef STRATA_VISION_GFX906
#include "ggml-cpu.h"
#include "gguf_meta.h"
#include "llama-cpp.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <hip/hip_runtime_api.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#endif

#include <algorithm>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#elif defined(STRATA_VISION_HIP)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef STRATA_VISION_GFX906
namespace {

using Clock = std::chrono::steady_clock;

bool verbose = false;

void vision_log(ggml_log_level level, const char* text, void*) {
    if (verbose || level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

int vision_error(const std::string& message, int status = 1) {
    std::printf("ERR %s\n", message.c_str());
    std::fflush(stdout);
    return status;
}

const mtmd_input_chunk* vision_image_chunk(mtmd_context* ctx, mtmd_input_chunks* chunks,
                                         const mtmd_bitmap* bitmap, std::string& error) {
    const std::string marker = mtmd_default_marker();
    const mtmd_input_text text{marker.c_str(), marker.size(), false, true};
    if (mtmd_tokenize(ctx, chunks, &text, &bitmap, 1) != 0) {
        error = "the image could not be preprocessed";
        return nullptr;
    }
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const auto* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) return chunk;
    }
    error = "no image chunk";
    return nullptr;
}

int vision_warmup(mtmd_context* ctx) {
    // Reserve GPU work buffers before READY so the text engine can size its expert cache against the remaining VRAM.
    constexpr uint32_t side = 2048;
    const std::vector<unsigned char> rgb(size_t(side) * side * 3, 128);
    const mtmd::bitmap_ptr bitmap(mtmd_bitmap_init(side, side, rgb.data()));
    if (!bitmap) return 0;
    const mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    std::string error;
    const auto* chunk = vision_image_chunk(ctx, chunks.get(), bitmap.get(), error);
    return chunk && mtmd_encode_chunk(ctx, chunk) == 0 ? (int) mtmd_input_chunk_get_n_tokens(chunk) : 0;
}

void vision_encode(mtmd_context* ctx, const std::string& image, const std::string& output, int n_embd, int max_tokens) {
    const auto start = Clock::now();
    const auto loaded = mtmd_helper_bitmap_init_from_file(ctx, image.c_str(), false, mtmd_helper_init_opt_default());
    const mtmd::bitmap_ptr bitmap(loaded.bitmap);
    const mtmd_helper::video_ptr video(loaded.video_ctx);
    if (!bitmap) {
        vision_error("cannot read the image " + image);
        return;
    }
    const mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    std::string error;
    const auto* chunk = vision_image_chunk(ctx, chunks.get(), bitmap.get(), error);
    if (!chunk) {
        vision_error(error);
        return;
    }
    const int n = (int) mtmd_input_chunk_get_n_tokens(chunk);
    if (n <= 0) {
        vision_error("the image contains no tokens");
        return;
    }
    const auto* tokens = mtmd_input_chunk_get_tokens_image(chunk);
    const auto last = mtmd_image_tokens_get_decoder_pos(tokens, 0, size_t(n) - 1);
    const int nx = (int) last.x + 1, ny = (int) last.y + 1;
    if (nx * ny != n) {
        vision_error("the image grid is not rectangular (" + std::to_string(n) + " tokens)");
        return;
    }
    // mtmd may ignore the requested cap for some projectors. Allow rounding of up to one eighth.
    if (max_tokens > 0 && n > max_tokens + max_tokens / 8) {
        vision_error("the picture became " + std::to_string(n) + " image tokens, over the limit of " +
                     std::to_string(max_tokens) + " (vision.max_tokens); send a smaller picture or raise the limit");
        return;
    }
    if (mtmd_encode_chunk(ctx, chunk) != 0) {
        vision_error("the vision encoder failed");
        return;
    }
    FILE* file = std::fopen(output.c_str(), "wb");
    if (!file) {
        vision_error("cannot write " + output);
        return;
    }
    const int32_t header[] = {0x31455653, n, nx, ny, n_embd};
    const bool written = std::fwrite(header, sizeof header, 1, file) == 1 &&
                         std::fwrite(mtmd_get_output_embd(ctx), sizeof(float) * size_t(n_embd), size_t(n), file) == size_t(n);
    if (std::fclose(file) != 0 || !written) {
        vision_error("short write to " + output);
        return;
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
    std::fflush(stdout);
}

int run_mtmd(const strata_vit::OwnReaderOptions& o, const std::string& model, const std::string& flash_attn) {
    // The loader expands BF16 weights exactly to FP32 in both modes. Also use FP32 for GEMM inputs and accumulation.
    setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32", 1);
    std::fprintf(stderr, "strata-vision: BF16 vision weights expanded losslessly to FP32; FP32 GEMMs\n");
    llama_log_set(vision_log, nullptr);
    mtmd_helper_log_set(vision_log, nullptr);
    llama_backend_init();
    const auto load_start = Clock::now();
    auto params = mtmd_context_params_default();
    params.use_gpu = o.gpu;
    params.print_timings = false;
    params.warmup = false;
    if (o.gpu) {
        params.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!params.device || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(params.device)), "ROCm") != 0)
            return vision_error("HIP vision requires the ROCm ggml backend");
    }
    if (o.gpu || flash_attn == "off" || (flash_attn.empty() && ggml_cpu_has_avx512()))
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    else if (flash_attn == "on")
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    const int max_tokens = o.gpu && o.max_tokens <= 0 ? 1024 : o.max_tokens;
    int threads = o.threads;
    if (!o.gpu && threads <= 0) threads = std::clamp(std::thread::hardware_concurrency() / 2, 1u, 16u);
    if (threads > 0) params.n_threads = threads;
    if (max_tokens > 0) params.image_max_tokens = max_tokens;
    if (o.min_tokens > 0) params.image_min_tokens = o.min_tokens;

    std::string how;
    const llama_model_ptr text(strata_open_text_model(model, &how));
    if (!text) return vision_error("cannot open the text model " + model + " (" + how + ")");
    const mtmd::context_ptr ctx(mtmd_init_from_file(o.mmproj.c_str(), text.get(), params));
    if (!ctx || !mtmd_support_vision(ctx.get())) return vision_error("cannot load the vision encoder " + o.mmproj);
    std::fprintf(stderr, "strata-vision: model files loaded in %.1f s\n",
                 std::chrono::duration<double>(Clock::now() - load_start).count());

    int n_embd = 0;
    if (auto* gg = gguf_init_from_file(o.mmproj.c_str(), {true, nullptr})) {
        const int64_t key = gguf_find_key(gg, "clip.vision.projection_dim");
        if (key >= 0) n_embd = (int) gguf_get_val_u32(gg, key);
        gguf_free(gg);
    }
    if (n_embd <= 0) return vision_error("the vision encoder has no projection_dim");
    if (o.gpu) {
        const auto warm_start = Clock::now();
        const int n = vision_warmup(ctx.get());
        std::fprintf(stderr, "strata-vision: warmed up at %d image tokens in %.1f s\n", n,
                     std::chrono::duration<double>(Clock::now() - warm_start).count());
        if (n <= 0) return vision_error("GPU vision warm-up failed");
        if (n > max_tokens + max_tokens / 8)
            std::fprintf(stderr, "strata-vision: WARNING the warm-up picture became %d image tokens, over the limit of %d (--max-tokens): "
                                 "the limit is not applied for this model; pictures over it are refused\n", n, max_tokens);
    } else {
        std::fprintf(stderr, "strata-vision: on the CPU, %d threads, no warm-up\n", threads);
    }

    std::printf("READY %d mtmd\n", n_embd);
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "QUIT") break;
        const size_t split = line.rfind(' ');
        if (line.compare(0, 4, "ENC ") != 0 || split <= 4 || split == line.size() - 1) {
            vision_error("expected: ENC <image> <output>");
            continue;
        }
        vision_encode(ctx.get(), line.substr(4, split - 4), line.substr(split + 1), n_embd, max_tokens);
    }
    return 0;
}

} // namespace
#endif

int main(int argc, char** argv) {
#if defined(_WIN32) && defined(STRATA_VISION_HIP)
    // hipblas.dll is delay-loaded (tools/vision/CMakeLists.txt): it and the ROCm DLLs behind it (rocblas.dll finds its kernel
    // library and ../.kpack beside itself) stay in rocm\bin of the engine package, which is not on the DLL search path of a
    // program started with nothing but System32 in PATH.  Put that folder on it before the first call.
    {
        wchar_t exe[MAX_PATH];
        const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe, n);
        const size_t cut = dir.find_last_of(L"\\/");
        if (cut != std::wstring::npos) {
            dir.resize(cut);
            const std::wstring rocm = dir + L"\\rocm\\bin";
            if (GetFileAttributesW(rocm.c_str()) != INVALID_FILE_ATTRIBUTES) SetDllDirectoryW(rocm.c_str());
        }
    }
#endif
#if defined(STRATA_VISION_HIP) && !defined(_WIN32)
    strata::core::kpack_path::apply(strata::core::kpack_path::exe_dir());   // #5060: the package's kpack archives for hipBLAS / rocBLAS
#endif
    strata_vit::OwnReaderOptions o;
#ifdef STRATA_VISION_GFX906
    std::string model, flash_attn;
    const char* tower_env = std::getenv("STRATA_VISION_TOWER");
    std::string tower = tower_env && *tower_env ? tower_env : "auto";
    if (tower != "mtmd" && tower != "own" && tower != "auto") {
        std::fprintf(stderr, "strata-vision: unknown STRATA_VISION_TOWER=%s, using auto\n", tower.c_str());
        tower = "auto";
    }
#else
    // STRATA_VISION_TOWER=mtmd came from the "Picture reader" setting removed in 0.2.16; a run config that still carries it keeps working, the name is no
    // longer read (one log line says so).  The same for --tower mtmd from an older harness.
    if (const char* env = std::getenv("STRATA_VISION_TOWER"); env && *env && std::strcmp(env, "own") != 0 && std::strcmp(env, "auto") != 0)
        std::fprintf(stderr, "strata-vision: STRATA_VISION_TOWER=%s is no longer read; Strata's picture reader runs\n", env);
#endif
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") o.mmproj = next();
        else if (a == "--model") {
#ifdef STRATA_VISION_GFX906
            model = next();
#else
            next();   // Strata's tower does not use the text tokenizer.
#endif
        }
        else if (a == "--gpu") o.gpu = true;
        else if (a == "--tower") {
            const std::string t = next();
            if (t != "mtmd" && t != "own" && t != "auto") { std::fprintf(stderr, "--tower takes mtmd, own or auto\n"); return 2; }
#ifdef STRATA_VISION_GFX906
            tower = t;
#else
            if (t == "mtmd") std::fprintf(stderr, "strata-vision: --tower mtmd is no longer read; Strata's picture reader runs\n");
#endif
        }
        else if (a == "--version") {
#ifdef STRATA_VISION_HIP
            std::printf("strata-vision hip (GPU architectures: %s)\n", STRATA_VISION_HIP_ARCHS);
#elif defined(STRATA_VISION_METAL)
            std::printf("strata-vision metal\n");
#elif defined(STRATA_VISION_SYCL)
            std::printf("strata-vision sycl (Intel GPU)\n");
#else
            std::printf("strata-vision\n");
#endif
            return 0;
        }
        else if (a == "--threads") o.threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") o.max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") o.min_tokens = std::atoi(next().c_str());   // the smallest picture's token count (#767)
        else if (a == "--flash-attn") {
#ifdef STRATA_VISION_GFX906
            flash_attn = next();
#else
            next();   // Strata's attention has one path per backend.
#endif
        }
        else if (a == "--verbose") {
#ifdef STRATA_VISION_GFX906
            verbose = true;
#endif
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.mmproj.empty()) {
        std::fprintf(stderr, "usage: strata-vision --mmproj <mmproj.gguf> [--model <model.gguf>] [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]\n");
        return 2;
    }
    // On the CPU the GPU stays unseen: a CUDA build otherwise opens a context there (measured: 0.4-0.7 GB of VRAM,
    // 150-260 expert slots less for the engine beside it).  Before anything reaches the CUDA runtime.
    if (!o.gpu) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
        _putenv_s("HIP_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
        setenv("HIP_VISIBLE_DEVICES", "-1", 1);
#ifdef STRATA_VISION_GFX906
        setenv("ROCR_VISIBLE_DEVICES", "-1", 1);
#endif
#endif
    }
#ifdef STRATA_VISION_GFX906
    if (o.gpu) {
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, 0) != hipSuccess || strata_gfx_in(props.gcnArchName) != "gfx906")
            return vision_error("HIP vision requires a visible real gfx906 device", 3);
        std::fprintf(stderr, "strata-vision: HIP device %s (%s)\n", props.name, props.gcnArchName);
    }
#elif defined(STRATA_VISION_HIP)
    if (o.gpu) {
        // The AMD GPU reader: a card the build has no code for would fail at its first kernel; say so before any kernel runs, with a line the server reads
        // (it then starts the CPU reader: serve/server.py, Vision._start).  STRATA_VISION_ARCH_CHECK=0 skips the check (a test of a compatible card).
        std::string why;
        if (!strata_vit::VitGpu::available(why)) {
            std::printf("ERR this picture reader found no GPU (it was built for: %s)\n", STRATA_VISION_HIP_ARCHS);
            std::fflush(stdout);
            return 3;
        }
        const std::string gfx = strata_gfx_in(strata_vit::VitGpu::arch_name());
        const char* check = std::getenv("STRATA_VISION_ARCH_CHECK");
        if (!gfx.empty() && (check == nullptr || std::atoi(check) != 0) && !strata_arch_listed(STRATA_VISION_HIP_ARCHS, gfx)) {
            std::printf("ERR this GPU (%s) is not in the picture reader's build (%s)\n", gfx.c_str(), STRATA_VISION_HIP_ARCHS);
            std::fflush(stdout);
            return 3;
        }
    }
#endif
#if !defined(_WIN32) && !defined(__APPLE__)   // macOS has no posix_fadvise
    // ask the OS for the projector file up front (128 KiB steps) so the loader's sequential reads find it cached;
    // STRATA_READ_AHEAD=0 turns it off, as in the engine
    if (const char* ra = std::getenv("STRATA_READ_AHEAD"); ra == nullptr || std::atoi(ra) != 0) {
        const int fd = open(o.mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
#ifdef STRATA_VISION_GFX906
    // Server run configs supply the text model and use the validated Vega20 reader. Model-free invocations use the own tower.
    if (tower == "mtmd" || (tower == "auto" && !model.empty())) {
        if (model.empty()) return vision_error("--tower mtmd requires --model", 2);
        if (o.gpu && flash_attn == "on") return vision_error("gfx906 vision uses --flash-attn off (no compiled flash attention)", 2);
        const int status = run_mtmd(o, model, flash_attn);
        llama_backend_free();
        return status;
    }
#endif
    return strata_vit::own_reader_run(o);
}
