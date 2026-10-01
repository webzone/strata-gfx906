# Strata for AMD Instinct MI50 / MI60

**`strata-gfx906` is an AMD-focused Strata fork for gfx906 accelerators.** The original Strata was built primarily for NVIDIA GPUs and CUDA; this fork adds a native AMD HIP/ROCm path tuned for the wave64 architecture in AMD Instinct MI50 and MI60 cards. It targets the real `gfx906` architecture and does not spoof another GPU generation.

The gfx906 backend is experimental and must be enabled explicitly. The validation documented in this repository has been performed on MI50 hardware. MI60 is an intended same-architecture target, but has not been independently validated here.

## Recent MI50 workload results

The following figures summarize recent runs of the current GSQ-RCO IQ2_XS quantized workload on MI50. They describe different stages and cache conditions; they are not directly comparable to one another or to measurements from the original NVIDIA-oriented project.

| Stage | Observed result | Workload / interpretation |
| --- | ---: | --- |
| Decode | **~15.1 tokens/s** (13.2–17.5) | Stable observed range for the current quantized model under CPU/GPU hybrid execution. |
| Uncached prefill, compute throughput | **~490 tokens/s** | Long-text input with approximately 32K new tokens and no KV-cache reuse; this is the raw prefill-compute baseline. |
| End-to-end prefill after a KV-cache hit | **11,000–27,800 effective tokens/s** | Prompt handling with very high KV-cache reuse. The effective rate includes reused prompt tokens. |
| First-token latency after a cache hit | **~1.6–4.4 s** | Approximately 48K-token prompt with substantial KV reuse; this is not a fresh, uncached 48K prefill. |

These are operational observations, not a controlled single-card-versus-dual-card benchmark; complete raw timing/configuration records for this summary are not archived here. In particular, cache-assisted effective throughput must not be read as uncached prefill compute speed. The separate short-prompt smoke measurements and their conditions are documented in the [gfx906 development guide](docs/GFX906.md).

## What this fork supports

- AMD Instinct **MI50 / MI60 (`gfx906`)** with Linux, ROCm, HIP, and hipBLAS. MI50 is the hardware validated so far; consult the guide before assuming MI60 compatibility in a particular system.
- The **GSQ-RCO IQ2_XS** Qwen3.8-Flash-Next model path used by the current validation. The model's expert tensors use several quantization types; the filename does not describe every tensor.
- Single-GPU inference and experimental multi-GPU **contiguous-layer / pipeline splitting**. This is not tensor parallelism.
- CPU/GPU hybrid expert execution and speculative decoding with the supported MTP setup.

This is not a general claim that every feature or model in upstream Strata is available on gfx906. For example, HIP image input is not available in the current validated path, and the HTTP server processes requests serially.

## Install and build

### Requirements

- Linux with the AMD GPU driver/KFD and an existing **ROCm 7 HIP + hipBLAS** installation. ROCm 7.2.4 is the version used in the documented MI50 validation. For the setup script, if ROCm is outside `/opt/rocm`, set `ROCM_PATH` to its installation directory. The example CMake preset itself points to `/opt/rocm`.
- Python 3.10 or newer with `venv`/`pip`, Git, and a C++ compiler. `setup.sh` creates a project-local `.venv` and installs Python dependencies there; the HIP engine is compiled locally for gfx906.
- For a clean **IQ2_XS** install, plan for roughly **80 GB or more** of free disk space across the model/data volume and keep at least 4 GiB free on the system volume. The installer checks required space before downloading. You can place model data on a larger disk with `--data-dir`.

### Recommended install (build engine, prepare model, do not start yet)

```bash
git clone --branch gfx906 https://github.com/webzone/strata-gfx906.git
cd strata-gfx906

./setup.sh \
  --backend hip \
  --experimental-gfx906 \
  --family qwen \
  --model IQ2_XS \
  --gpus 0,1 \
  --context 65536 \
  --kv int8 \
  --host 127.0.0.1 \
  --port 8095 \
  --yes \
  --no-start
```

This first run creates `.venv`, fetches the pinned llama.cpp source, downloads the two model shards (about 68 GB) and selected MTP tensors (about 5.2 GB), builds the HIP engine for the selected card(s), prepares the model, and writes a config plus a `run-*.sh` launcher. It may take a while and requires a large download. The example selects two GPUs; use `--gpu 0` for one card instead of `--gpus 0,1`. GPU numbers are the AMD indices printed by the installer. The default layer placement is automatic; for two MI50s you may explicitly set `--layer-split 24`.

`--context 65536` is an example capacity for prompts around 48K tokens plus answer space, not a promise that every machine has enough memory or that every long-context workload is validated. Reduce it if the setup reports insufficient RAM/VRAM. Omit `--no-start` if you want setup to launch the server immediately after installation.

For a lower-level developer rebuild after setup has fetched the pinned dependency, use:

```bash
cmake --preset hip-gfx906
cmake --build --preset hip-gfx906
```

This compiles the engine only; it does not download model files or create a serving config. See the [gfx906 development guide](docs/GFX906.md) for the dual-GPU test preset and the full regression coverage. Do not set `HSA_OVERRIDE_GFX_VERSION` or use a wheel built for a different GPU architecture.

## Start and stop

After the install command above, start the generated launcher (for the example, `IQ2_XS`):

```bash
./run-iq2_xs.sh
```

It starts the server in the foreground and opens the local web app when the model is ready. The default binding is local-only; with the example settings, open **<http://127.0.0.1:8095/>**. The OpenAI-compatible API is at `http://127.0.0.1:8095/v1`.

To run without the generated launcher or browser-opening behavior, start the server directly:

```bash
.venv/bin/python serve/server.py \
  --engine strata \
  --config strata-iq2_xs.json \
  --host 127.0.0.1 \
  --port 8095
```

Stop it with **Ctrl+C** in the server terminal. The server shuts down the HTTP listener and engine gracefully; if needed, a second Ctrl+C forces the engine to exit. Restart by running the launcher again. Changes to the setup options below require stopping the server first.

For LAN access, set both `--host 0.0.0.0` and a strong `--api-key <secret>`; do not expose an unauthenticated server. For remote access, a loopback-only server plus an SSH tunnel is preferable.

## Startup and tuning options

There are two groups of options. `setup.sh` options choose and save the model/server configuration. Engine options live in the generated config's `args` list.

### Setup options

| Option | Purpose |
| --- | --- |
| `--backend hip --experimental-gfx906` | Required for the experimental MI50/gfx906 path. Keep both on initial install or when creating a new gfx906 model config. |
| `--gpu 0` / `--gpus 0,1` | Select one AMD GPU or an ordered multi-GPU layer split. `--gpus` saves the selection; GPU numbering follows the AMD indices reported by setup. |
| `--layer-split 24` | Optional layer boundary for a two-card run; omit it to let the engine place the split automatically. This is a layer/pipeline split, not tensor parallelism. |
| `--context 65536` | Set the model's context capacity. Larger contexts need more memory and may leave less VRAM for expert caching. |
| `--kv int8`, `--kv q4_0`, `--kv k8v4` | Choose the KV-cache format when supported by the selected context. The installer's default is context-dependent. |
| `--port 8095` | Set the HTTP port. |
| `--host 127.0.0.1` / `--api-key SECRET` | Bind locally by default; use `0.0.0.0` only with a strong API key. |
| `--data-dir DIR` | Place the model data, packs, and MTP files on another disk. |
| `--models-dir DIR` / `--gguf-dir DIR` | Choose the GGUF download directory or reuse an existing folder containing the model shards. |
| `--no-start` | Finish installation and write the launcher without starting inference. |
| `--setup` | Reconfigure an existing installation. Repeat the backend, GPU/model selection, host, and port you want to keep, e.g. `./setup.sh --setup --backend hip --experimental-gfx906 --family qwen --model IQ2_XS --gpus 0,1 --context 65536 --kv int8 --host 127.0.0.1 --port 8095 --yes --no-start`. |

The first setup also compiles the HIP engine automatically; `--build` is not needed for this AMD path. The installer may be run again with `--setup` to change saved options; stop the running server first, repeat the GPU/model/host/port choices you want to preserve, then relaunch the generated script.

### Direct server options

When invoking `serve/server.py` directly, `--engine strata` and `--config <file>` are required. The commonly used wrapper options are:

| Option | Purpose |
| --- | --- |
| `--host 127.0.0.1` | Bind to this host only (default and recommended). |
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
| `--prompt-cache N`, `--suffix-draft N` | Conversation/prompt reuse and suffix-based draft options. Prompt caching does not make the server concurrent. |

With a multi-GPU config (`"gpu": [0, 1]`), the server wrapper supplies `--layer-split` from the config automatically; do **not** add a second `--layer-split` to `args`. The HTTP server currently handles requests serially. API request fields such as `max_tokens` control the response length; they are not installer startup flags. This engine does not take llama.cpp-style `-c`, `-ngl`, or `--tensor-split` options.

For the engine's complete option list, run `./engine/strata --help` after installation. The archived MI50 regression run completed **40 tests, with 1 skipped and 0 failed**, plus **18/18** Python entrypoint/download-safety tests. Skips and excluded tests are described in the [evidence notes](docs/gfx906-results/20261001/README.md); these software checks do not establish production readiness or long-run stability.

## Project background

This repository is based on upstream Strata 0.1.30. It carries a gfx906-specific HIP implementation and experimental MI50/MI60 support work; it is not the upstream project's general NVIDIA/CUDA release. Upstream source: [Niko1221/Strata](https://github.com/Niko1221/Strata). For architecture details, compatibility limitations, and reproducible evidence, start with the [gfx906 development guide](docs/GFX906.md).

## License

Strata is open source under the [MIT License](LICENSE). Third-party components and model files may carry separate licenses; see their respective notices and source pages.
