# Strata for AMD Instinct MI50 / MI60

**`strata-gfx906` is an AMD-focused Strata fork for gfx906 accelerators.** The original Strata was built primarily for NVIDIA GPUs and CUDA and now has an RDNA wave32 HIP backend; this fork extends the native AMD HIP/ROCm path to the wave64 architecture in AMD Instinct MI50 and MI60 cards. It targets the real `gfx906` architecture and does not spoof another GPU generation.

This fork is specifically tuned to run **Qwen3.8-Flash-Next** using model weights in **GGUF** format. The model's native/trained context length is **262,144 tokens**. The `--context` setting configures runtime capacity; hardware limits may justify choosing less. Native context is not an extended-RoPE setting, and it does not imply that full-length inference has completed acceptance testing on every system.

The gfx906 backend is experimental and must be enabled explicitly. The validation documented in this repository has been performed on MI50 hardware. MI60 is an intended same-architecture target, but has not been independently validated here.

## Upstream version and fork policy

**Synchronized with upstream Strata v0.1.31**, commit
[`9259cad4cfa3543cd3b8decab5962672b968c649`](https://github.com/Niko1221/Strata/commit/9259cad4cfa3543cd3b8decab5962672b968c649),
by merging into `gfx906` (not rebasing the published fork history). The original port was based on v0.1.30 / `30ec18e`.

This is a dedicated **MI50 / MI60 gfx906 fork because upstream does not support these wave64 GPUs**.
We will continue absorbing upstream fixes and features while retaining and optimizing the real gfx906
HIP path: two logical wave32 groups per physical wave64, native signed dot4, per-card clock calibration,
and per-device mapped-memory layer handoff. Upstream RDNA/NVIDIA support is not a replacement for gfx906 support.

**Source synchronization is not GPU acceptance.** The historical timings and GPU results below predate
this merge. A subsequent [v0.1.31 MI50 deployment](docs/gfx906-results/20261001-deployment-v0.1.31/README.md)
completed the HIP build, 44 selected CTests (one additional hipBLASLt skip), and short IQ2_XS single/dual/
reversed-card generation checks. A loopback API is running with a configured 262K capacity; full-length
and long-soak acceptance are still outstanding. No MI60 hardware has been tested here.

### Changes absorbed from v0.1.31

| Upstream change | Meaning for this fork |
| --- | --- |
| HIP byte permutation and packed-byte arithmetic optimizations; IQ and Q2_0 kernel improvements | Retained alongside our gfx906 dot4/ballot fixes; performance and numerical behavior must be rechecked on gfx906. |
| AMD `--gpus` layer splitting and compilation for every selected architecture | Integrated with the explicit gfx906 gate and system-ROCm requirement; no RDNA wheel substitution. |
| Split-GGUF handling, GGUF-in-place expert reads, RAM/SSD expert tiers and routing prefetch | Avoids a duplicate main-model `experts.bin`; tier budgets and counters are described in [DETAILS](docs/DETAILS.md). |
| Experimental Unsloth UD-Q4_K_XL import; Q4_K/Q5_K/Q5_1/Q8_0 native expert support | Source support is included; this is a manual import, **not a gfx906 model-validation claim**. See [UNSLOTH_Q4](docs/UNSLOTH_Q4.md). GSQ-RCO IQ2_XS remains the acceptance model. |
| Server status/history race, truncated tool-call handling, long-word tokenizer fixes and opt-in reasoning budget | Included without changing the serialized-request model or adding HIP image support. |
| Reproducible package/model pins and improved parity fixtures | Kept with stricter fork source-identity protection: missing pinned model/MTP revisions fail rather than silently switching to mutable `main`. |
| Community benchmark reports and submission guide | See [COMMUNITY_BENCHMARKS](docs/COMMUNITY_BENCHMARKS.md); upstream NVIDIA/RDNA numbers must not be reported as MI50/MI60 results. |

## Recent MI50 workload results

The following figures summarize GSQ-RCO IQ2_XS quantized workload observations on MI50 **before the v0.1.31 merge**. They describe different stages and cache conditions; they are not directly comparable to one another or to measurements from the original NVIDIA-oriented project.

| Stage | Observed result | Workload / interpretation |
| --- | ---: | --- |
| Decode | **~15.1 tokens/s** (13.2–17.5) | Stable observed range for the current quantized model under CPU/GPU hybrid execution. |
| Uncached prefill, compute throughput | **~490 tokens/s** | Long-text input with approximately 32K new tokens and no KV-cache reuse; this is the raw prefill-compute baseline. |
| End-to-end prefill after a KV-cache hit | **11,000–27,800 effective tokens/s** | Prompt handling with very high KV-cache reuse. The effective rate includes reused prompt tokens. |
| First-token latency after a cache hit | **~1.6–4.4 s** | Approximately 48K-token prompt with substantial KV reuse; this is not a fresh, uncached 48K prefill. |

These are operational observations, not a controlled single-card-versus-dual-card benchmark; complete raw timing/configuration records for this summary are not archived here. In particular, cache-assisted effective throughput must not be read as uncached prefill compute speed. The separate short-prompt smoke measurements and their conditions are documented in the [gfx906 development guide](docs/GFX906.md).

## What this fork supports

- AMD Instinct **MI50 / MI60 (`gfx906`)** with Linux, ROCm, HIP, and hipBLAS. MI50 is the hardware validated so far; consult the guide before assuming MI60 compatibility in a particular system.
- **Qwen3.8-Flash-Next GGUF** weights. The current engineering/validation configuration uses **GSQ-RCO IQ2_XS**, but that is not the only supported quantization. The setup supports Qwen GGUF variants including Q2_0, IQ2_XS, IQ3_XXS, and IQ3_S; the Coder family also offers IQ1_M. Choose among available variants according to the desired quality, RAM/VRAM, disk capacity, and speed. The GGUF filename describes the pack variant, not every tensor's internal quant type.
- Single-GPU inference and experimental multi-GPU **contiguous-layer / pipeline splitting**. This is not tensor parallelism.
- CPU/GPU hybrid expert execution and speculative decoding with the supported MTP setup.

This is not a general claim that every feature or model in upstream Strata is available on gfx906. For example, HIP image input is not available in the current validated path, and the HTTP server processes requests serially.

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
RX 7800 XT / 7700 XT and RX 9060 XT support. See [AMD_HIP](docs/AMD_HIP.md) for their scope and
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

This repository tracks upstream Strata through **v0.1.31 / `9259cad`** while maintaining its own gfx906
compatibility and optimization work. The upstream project still excludes wave64; this fork's purpose is
to keep MI50 / MI60 usable as upstream evolves. It is not the upstream project's general NVIDIA/CUDA
release. Upstream source: [Niko1221/Strata](https://github.com/Niko1221/Strata). For architecture details,
compatibility limitations, source-sync validation, and reproducible evidence, start with the
[gfx906 development guide](docs/GFX906.md).

## Where things are stored

- **Chats stay in the browser.** The Chat tab keeps conversation history, settings and the entered API key
  in local storage (`strata.*` keys), not in server files. Another browser or private session starts empty;
  clearing site data removes those chats. HIP image input is not supported.
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

For other issues, use the [upstream troubleshooting guide](docs/DETAILS.md#troubleshooting) and attach a
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
