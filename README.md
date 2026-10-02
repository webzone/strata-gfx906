# Strata for AMD Instinct MI50 / MI60

**Source version: Strata v0.1.34**

**`strata-gfx906` is an AMD-focused Strata fork for gfx906 accelerators.** The original Strata was built primarily for NVIDIA GPUs and CUDA and now has an RDNA wave32 HIP backend; this fork extends the native AMD HIP/ROCm path to the wave64 architecture in AMD Instinct MI50 and MI60 cards. It targets the real `gfx906` architecture and does not spoof another GPU generation.

This fork is specifically tuned to run **Qwen3.8-Flash-Next** using model weights in **GGUF** format. The model's native/trained context length is **262,144 tokens**. The `--context` setting configures runtime capacity; hardware limits may justify choosing less. Native context is not an extended-RoPE setting, and it does not imply that full-length inference has completed acceptance testing on every system.

The gfx906 backend is experimental and must be enabled explicitly. The validation documented in this repository has been performed on MI50 hardware. MI60 is an intended same-architecture target, but has not been independently validated here.

## Recent MI50 workload results

These are deployed workload observations from the dual-MI50 T5810 server (ROCm 7.2.4). Every window
includes repeated turns, a warm prompt/KV cache, speculative decoding and a large active context; each is
an operational snapshot, **not a controlled benchmark**. Each subsection keeps its engine-version and
quantization label. The windows differ in workload, context depth and cache state, so figures from the two
quants are an indicative comparison, not a controlled A/B.

### GSQ-RCO IQ2_XS — deployed v0.1.31 observation (2026-10-01)

These v0.1.31 observations replaced the earlier pre-v0.1.31 numbers in this README and are kept under their
original label.

| Metric | v0.1.31 observation | Workload / interpretation |
| --- | ---: | --- |
| Lifetime decode mean | **42.39 tok/s** | 18,757 generated tokens over 442.5 seconds of decoder time. |
| Recent request decode | **40.0–49.8 tok/s** | Per-request observations in the recent history. |
| Live-history bursts | **51–59 tok/s** | Brief peaks during high-speculation phases, not sustained throughput. |
| Speculative decoding | **`spec: 4`, `spec_min_p: 0.50`** | MTP is enabled and contributes to the observed decode rate. |
| Lifetime prompt/KV reuse | **96.9%** | 2,606,953 reused tokens out of 2,690,006 total prompt tokens. |
| Recent prompt/KV reuse | **~97–99.9%** | Recent 83K–100K-token contexts were mostly cache hits; one example reused 100,097 of 100,150 tokens (~99.95%). |
| TTFT, warm/high-hit cache | **~0.8–1.05 s** | Time to first token with a high cache hit rate. |
| TTFT, partial cache miss | **~2.0–9.4 s** | Requests had roughly 800–3,000 new tokens; latency varies with the miss and request. |

The configured context capacity is 262,144 tokens; this does not mean each request processed that many
new tokens. Cache-reuse percentage counts reused prompt tokens, **not** uncached prefill throughput, and
warm-cache TTFT is not a fresh-prefill measurement.

**Resource snapshot while serving:** system RAM use was approximately **40.4 GB of 115.9 GB** (~34.9%).
The dashboard summarized model/arena reservation as **~33.8 GB** (`arena_mib: 33812`); the raw MiB counter
is about **33.0 GiB** (35.4 decimal GB), so use the raw field—not the rounded GB label—for precise
comparisons. Free VRAM was **11,012 MiB** (~10.75 GiB). Aggregate CPU utilization was approximately
**17–18%** on the 12-logical-thread Xeon E5-1650 v3 (roughly two logical CPUs' worth of load).

This live result is substantially above the earlier pre-v0.1.31 decode observation, which is no longer
presented here as the current baseline. The two observations used different software and workload windows;
the gain cannot be attributed to a single upstream kernel or to the three ROCm environment settings
without a matched A/B. Cache state and speculative decoding materially affect the rates, so compare only
matched prompts, context, cache state, MTP settings and runtime environment. This is not a long-soak result.
The bounded v0.1.31 build/model/API validation and its limitations are recorded in the
[deployment evidence](docs/gfx906-results/20261001-deployment-v0.1.31/README.md); separate short-prompt
GPU checks are documented in the [gfx906 development guide](docs/GFX906.md).

### GSQ-RCO IQ3_S — v0.1.34 live observation (2026-10-02)

After the [IQ3_S preparation](docs/gfx906-results/20261002-iq3-s-preparation/README.md) and the in-place
[v0.1.34 deployment](docs/gfx906-results/20261002-in-place-v0.1.34/README.md), the server was switched to
the **GSQ-RCO IQ3_S** pack: the same real-gfx906 binary (SHA256 `d8a59558877074f6…`) with sampling,
MTP/speculative, int8-KV and split-24 settings retained. The figures below come from one live-monitor
snapshot (186 requests over ~3.9 h of sequential agentic turns at 167K–174K-token contexts, warm cache,
`spec: 4`, `spec_min_p: 0.50`) recorded verbatim with derived values in the
[live observation evidence](docs/gfx906-results/20261002-iq3-s-live/README.md). It is an operational
snapshot, **not a controlled benchmark**, and no model-parity or output-quality claim follows from it.

| Metric | IQ2_XS, v0.1.31 (2026-10-01) | IQ3_S, v0.1.34 (2026-10-02) |
| --- | ---: | ---: |
| Lifetime decode mean | **42.39 tok/s** (18,757 tok / 442.5 s) | **39.25 tok/s** (158,099 tok / 4,028.3 s) |
| Recent request decode | 40.0–49.8 tok/s | 37.0–47.0 tok/s (12-request window, mean ≈ 40.1) |
| Speculative decoding | `spec: 4`, `spec_min_p: 0.50`, MTP enabled | `spec: 4`, `spec_min_p: 0.50` (MTP/spec settings retained in the IQ3_S config) |
| Lifetime prompt/KV reuse | 96.9% (2,606,953 / 2,690,006) | 97.0% (23,327,626 / 24,040,403) |
| Recent prompt/KV reuse | ~97–99.9% at 83K–100K-token contexts | ~99.9%; turns added 40–1,029 tokens at 167K–174K |
| TTFT, high-hit small turns | ~0.8–1.05 s | ~0.8–2.2 s (≤ ~200 new tokens) |
| TTFT, larger cache misses | ~2.0–9.4 s (~800–3,000 new tokens) | ~2.2–4.7 s (~490–1,030 new tokens) |
| Effective new-token prefill | not recorded | 446.7 tok/s lifetime (712,777 new of 24.04M prompt tokens); 51–217 tok/s per recent request |
| Expert cache hit (VRAM-resident experts) | not recorded | 98.6–99.4%, `pcie_frac: 0.00` |
| Expert arena (`arena_mib`) | 33,812 | 47,962 |
| Resident experts, primary GPU | not recorded | 12,288 slots / 22,675 MiB (24,072 slots total) |
| System RAM while serving | ~40.4 GB of 115.9 GB | ~55.7 GB of 115.9 GB |
| Free VRAM | 11,012 MiB | 2,876 MiB (GPU0 2,489 / GPU1 403) |
| Aggregate CPU while serving | ~17–18% | not captured (snapshot taken while idle) |
| Model storage | ~68 GB shards + 5.2 GB MTP | +54.8 GB new shard; the 28.8 GB PLE shard is hard-linked (no duplicate) |

`prompt_ms` counts uncached prompt tokens only in both windows, so the TTFT rows are comparable; the decode
rates are not directly attributable. The IQ3_S window ran at roughly twice the context depth, with a ~42%
larger expert arena, on a newer engine and a different workload, so the ~7% lower lifetime decode mean
cannot be assigned to the quantization without a matched A/B on the same binary, prompts and cache state.
IQ3_S is the higher-bitpoint mixed recipe (its per-tensor types are recorded in the preparation receipt);
the observed price is ~+15 GB RAM, ~−8 GiB free VRAM and the slightly lower decode mean. At 167K–174K
tokens the server sat ~316 MiB above the 2,560 MiB conversation-cache VRAM floor, so conversations growing
toward the full 262K context need headroom re-checked on this quant.

## What this fork supports

- AMD Instinct **MI50 / MI60 (`gfx906`)** with Linux, ROCm, HIP, and hipBLAS. MI50 is the hardware validated so far; consult the guide before assuming MI60 compatibility in a particular system.
- **Qwen3.8-Flash-Next GGUF** weights. The engineering/validation (acceptance) configuration uses **GSQ-RCO IQ2_XS**; a **GSQ-RCO IQ3_S** pack is also prepared on the reference server and selected manually at launch (live observations above). Neither is the only supported quantization: the setup supports Qwen GGUF variants including Q2_0, IQ2_XS, IQ3_XXS, and IQ3_S; the Coder family also offers IQ1_M. Choose among available variants according to the desired quality, RAM/VRAM, disk capacity, and speed. The GGUF filename describes the pack variant, not every tensor's internal quant type.
- Single-GPU inference and experimental multi-GPU **contiguous-layer / pipeline splitting**. This is not tensor parallelism.
- CPU/GPU hybrid expert execution and speculative decoding with the supported MTP setup.

This is not a general claim that every feature or model in upstream Strata is available on gfx906. The HTTP server processes requests serially. Upstream now offers an optional Linux CPU image encoder (`--vision cpu`); it is integrated in source but has not been validated with the gfx906 model. There is no HIP GPU image encoder.

## Install and build

### Requirements

- Linux with the AMD GPU driver/KFD and an existing **ROCm 7 HIP + hipBLAS** installation. ROCm 7.2.4 is the version used in the documented MI50 validation. For the setup script, if ROCm is outside `/opt/rocm`, set `ROCM_PATH` to its installation directory. The example CMake preset itself points to `/opt/rocm`.
- An **AVX2-capable x86-64 CPU**; v0.1.31 explicitly rejects CPUs without AVX2. The tested Xeon E5-1650 v3 meets this requirement.
- Python 3.10 or newer with `venv`/`pip`, Git, and a C++ compiler. `setup.sh` creates a project-local `.venv` and installs the pinned Python dependencies there; the HIP engine is compiled locally for gfx906.
- Disk requirements depend on model and quantization. A clean **IQ2_XS** install needs roughly **80 GB or more** of free space across the model/data volume; keep at least 4 GiB free on the system volume. The installer checks required space before downloading. Use `--data-dir` to place model data on a larger disk.

### Recommended install (build engine, prepare model, do not start yet)

```bash
git clone --branch gfx906 https://github.com/webzone/strata-gfx906.git
cd strata-gfx906

export STRATA_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

./setup.sh \
  --backend hip \
  --experimental-gfx906 \
  --family qwen \
  --model IQ2_XS \
  --gpus 0,1 \
  --context 262144 \
  --kv int8 \
  --host 0.0.0.0 \
  --api-key "$STRATA_API_KEY" \
  --port 8095 \
  --yes \
  --no-start
```

This first run creates `.venv`, fetches the pinned llama.cpp source, downloads the two model shards (about 68 GB) and selected MTP tensors (about 5.2 GB), builds the HIP engine for the selected card(s), prepares the model, and writes a config plus a `run-*.sh` launcher. It may take a while and requires a large download. The example selects two GPUs; use `--gpu 0` for one card instead of `--gpus 0,1`. GPU numbers are the AMD indices printed by the installer. The default layer placement is automatic; for two MI50s you may explicitly set `--layer-split 24`.

The install example configures the model's native **262,144-token** context. A lower value can reduce memory use on a constrained system, but it is a runtime cap below the model's native length. Omit `--no-start` if you want setup to launch the server immediately after installation.

For a lower-level developer rebuild after setup has fetched the pinned dependency, use:

```bash
cmake --preset hip-gfx906
cmake --build --preset hip-gfx906
```

This compiles the engine only; it does not download model files or create a serving config. See the [gfx906 development guide](docs/GFX906.md) for the dual-GPU test preset and the full regression coverage. Do not set `HSA_OVERRIDE_GFX_VERSION` or use a wheel built for a different GPU architecture.

Upstream's RDNA paths are retained: gfx1100, gfx1101, gfx1200 and gfx1201, including community-validated
RX 7800 XT / 7700 XT and RX 9060 XT support, plus community-reported gfx1030 (RX 6800 / 6900). See [AMD_HIP](docs/AMD_HIP.md) for their scope and
[Multi-GPU](docs/MULTI_GPU.md) for pipeline details. Those cards may use family-specific TheRock wheels;
**MI50 / MI60 may not** and always require `--experimental-gfx906` plus an existing system ROCm SDK.

## Start and stop

After the install command above, start the generated launcher (for the example, `IQ2_XS`):

```bash
./run-iq2_xs.sh
```

It starts the server in the foreground and opens the local web app when the model is ready. The example listens on all interfaces and requires the API key generated above. On the server itself, open **<http://localhost:8095/>**; from another device, use the MI50 host's IP address and provide the API key. The OpenAI-compatible API is at `http://<MI50-host>:8095/v1`.

To run without the generated launcher or browser-opening behavior, start the server directly:

```bash
.venv/bin/python serve/server.py \
  --engine strata \
  --config strata-iq2_xs.json \
  --host 0.0.0.0 \
  --port 8095
```

The generated config contains the API key, so the server enforces it when started directly. Stop it with **Ctrl+C** in the server terminal. The server shuts down the HTTP listener and engine gracefully; if needed, a second Ctrl+C forces the engine to exit. Restart by running the launcher again. Changes to the setup options below require stopping the server first.

`0.0.0.0` exposes the listener on all network interfaces. Always configure a strong API key and firewall restrictions; never publish the generated config or API key. If you need only local access, use a loopback bind instead.

## Startup and tuning options

There are two groups of options. `setup.sh` options choose and save the model/server configuration. Engine options live in the generated config's `args` list.

### Setup options

| Option | Purpose |
| --- | --- |
| `--backend hip --experimental-gfx906` | Required for the experimental MI50/MI60 gfx906 path. Keep both on initial install or when creating a new gfx906 model config. |
| `--model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S` | Choose a Qwen3.8-Flash-Next GGUF quantization. For the Coder family, use `--family coder --model IQ1_M`. The tested IQ2_XS option is not the only supported one. |
| `--gpu 0` / `--gpus 0,1` | Select one AMD GPU or an ordered multi-GPU layer split. `--gpus` saves the selection; GPU numbering follows the AMD indices reported by setup. |
| `--layer-split 24` | Optional layer boundary for a two-card run; omit it to let the engine place the split automatically. This is a layer/pipeline split, not tensor parallelism. |
| `--context 262144` | Set the runtime capacity up to the model's native 262,144 tokens. Lower it only if hardware memory requires it. |
| `--kv int8`, `--kv q4_0`, `--kv k8v4` | Choose the KV-cache format when supported by the selected context. The installer's default is context-dependent. |
| `--port 8095` | Set the HTTP port. |
| `--host 0.0.0.0` / `--api-key SECRET` | Listen on all interfaces; always use a strong API key and firewall rules. |
| `--data-dir DIR` | Place the model data, packs, and MTP files on another disk. |
| `--models-dir DIR` / `--gguf-dir DIR` | Choose the GGUF download directory or reuse an existing folder containing the model shards. |
| `--no-start` | Finish installation and write the launcher without starting inference. |
| `--setup` | Reconfigure an existing installation. Repeat backend, GPU/model, host, port, and API key choices, e.g. `./setup.sh --setup --backend hip --experimental-gfx906 --family qwen --model IQ2_XS --gpus 0,1 --context 262144 --kv int8 --host 0.0.0.0 --api-key "$STRATA_API_KEY" --port 8095 --yes --no-start`. |

The first setup also compiles the HIP engine automatically; `--build` is not needed for this AMD path. The installer may be run again with `--setup` to change saved options; stop the running server first, repeat the GPU/model/host/port/API-key choices you want to preserve, then relaunch the generated script. The API key is stored in the local generated config; keep that file private. Set `STRATA_API_KEY` to the same saved key before using the reconfiguration example.

### Direct server options

When invoking `serve/server.py` directly, `--engine strata` and `--config <file>` are required. The commonly used wrapper options are:

| Option | Purpose |
| --- | --- |
| `--host 0.0.0.0` | Listen on all interfaces; use with `--api-key` and firewall restrictions. |
| `--port 8095` | Override the configured HTTP port for this start. |
| `--api-key SECRET` | Require a key on API routes; required if listening on `0.0.0.0`. |
| `--gpu 0` | Choose GPU(s) for this start; the AMD indices are those reported by the setup script. |
| `--open` | Open the local web interface after the model is ready. |

### Engine options

To change engine tuning, edit the generated `strata-iq2_xs.json` and preserve its existing required paths/flags. Change the corresponding values in the `args` array rather than adding duplicate flags. Common options include:

| Engine option in `args` | Purpose |
| --- | --- |
| `--max-context N` | Engine context capacity (normally set through installer option `--context`). |
| `--kv fp16|int8|q4_0|k8v4` | KV-cache representation. |
| `--expert-cache auto|N` | GPU-resident expert-cache sizing. |
| `--prefill auto|N` | Prompt-processing chunk size. |
| `--spec N`, `--mtp DIR`, `--spec-min-p P` | Speculative decoding/MTP settings. The generated config supplies the required MTP path and verifier settings. |
| `--pool-workers N` | CPU expert-pool worker count; this does not set request concurrency. |
| `--resident-budget-gib N` | RAM-tier budget for GGUF-in-place experts; remaining experts use the SSD/file tier. Experimental on gfx906 until revalidated. |
| `--prompt-cache N`, `--suffix-draft N` | Conversation/prompt reuse and suffix-based draft options. Prompt caching does not make the server concurrent. |

With a multi-GPU config (`"gpu": [0, 1]`), the server wrapper supplies `--layer-split` from the config automatically; do **not** add a second `--layer-split` to `args`. The HTTP server currently handles requests serially. API request fields such as `max_tokens` control the response length; they are not installer startup flags. This engine does not take llama.cpp-style `-c`, `-ngl`, or `--tensor-split` options.

For the engine's complete option list, run `./engine/strata --help` after installation. The archived **v0.1.30** MI50 regression run had **40 passed, 1 skipped and 0 failed** out of 41 tests, plus **18/18** Python entrypoint/download-safety tests. Skips and excluded tests are described in the [evidence notes](docs/gfx906-results/20261001/README.md); these software checks do not establish production readiness or long-run stability.

## Project background

This repository tracks upstream Strata through **v0.1.34 / `1678de3`** while maintaining its own gfx906
compatibility and optimization work. The upstream project still excludes wave64; this fork's purpose is
to keep MI50 / MI60 usable as upstream evolves. It is not the upstream project's general NVIDIA/CUDA
release. Upstream source: [Niko1221/Strata](https://github.com/Niko1221/Strata). For architecture details,
compatibility limitations, source-sync validation, and reproducible evidence, start with the
[gfx906 development guide](docs/GFX906.md). The deployed v0.1.31 measurements above do not validate
newer source or establish a post-merge speedup. General upstream model choices, installation, MCP tools
and architecture explanations are in [MODELS](docs/MODELS.md), [INSTALL](docs/INSTALL.md),
[MCP_SERVER](docs/MCP_SERVER.md) and [HOW_IT_WORKS](docs/HOW_IT_WORKS.md); use this fork's MI50 instructions
rather than their RDNA wheel/prebuilt defaults.

## Where things are stored

- **Chats stay in the browser.** The Chat tab keeps conversation history, settings and the entered API key
  in local storage (`strata.*` keys), not in server files. Another browser or private session starts empty;
  clearing site data removes those chats. CPU image input has not been validated on gfx906.
- **Launch settings stay in the checkout:** `strata-<model>.json`, `run-<model>.sh`,
  `strata-<model>.log`, and optionally `strata-<model>.shared-settings.json`. Do not publish API keys.
- **Model data stays outside the checkout by default:** `models/`, `packs/` and `mtp/` live in
  the sibling `Strata-data` directory, or the location selected with `--data-dir`. The original engineering
  tests used explicitly selected project-local data paths; updating the source does not require redownloading them.
- **The saved data-folder location on Linux** is `~/.config/strata/settings.json`.
  See [DETAILS](docs/DETAILS.md) for storage and migration behavior.

## Troubleshooting and benchmark reports

If gfx906 is rejected, confirm `--backend hip --experimental-gfx906`, the real compiled architecture,
and the system ROCm path; do not spoof a GPU or install RDNA wheels. If a pinned model/MTP revision
is unavailable, stop and verify its identity rather than replacing it with files from `main`.
Check disk/RAM/GPU availability before retrying; never stop another workload or delete unrelated models
to make room. A host ECC/MCE warning requires hardware investigation, not repeated model stress tests.

For other issues, use the [upstream troubleshooting guide](docs/TROUBLESHOOTING.md) and attach a
sanitized engine log. The [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) provides a report template;
state the fork/upstream commits, GPU architecture, exact weights, prompt and cache conditions, MTP settings,
and skipped tests. NVIDIA/RDNA benchmarks do not establish gfx906 speed or parity.

## Credits

Upstream Strata is maintained by [Niko1221](https://github.com/Niko1221/Strata). Qwen supplies the
[Qwen3.8-Flash-Next model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next);
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) supplies the GSQ-RCO
GGUFs, and [UkisAI](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) supplies
Swift 1.5. The engine uses [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp);
see [upstream credits and licenses](docs/DETAILS.md#credits-and-licenses) for additional acknowledgments.

## License

Strata is open source under the [MIT License](LICENSE). Third-party components and model files may carry separate licenses; see their respective notices and source pages.
