# Strata for AMD Instinct MI50 / MI60

**Source version: Strata v0.1.40.1 / `82f46a8`** (the engine reports v0.1.40).

**Key features in this release:**
- **Concurrent execution:** Multi-slot concurrent request batching and decoding across independent context sequences.
- **GPU vision:** Experimental gfx906 HIP GPU-accelerated image encoding (`--vision gpu`) alongside upstream CPU fallback.
- **Measured on dual MI50 (v0.1.40, ROCm 10, IQ3_S):** decode **48.4–59.7 tok/s** per stream; cold prefill
  **487–670 tok/s**; prompt handling with a warm K/V cache reaches **~247,000 effective tok/s** (reused
  tokens counted, not compute throughput). See [MI50 workload results](#mi50-workload-results).

**`strata-gfx906` is an AMD-focused Strata fork for gfx906 accelerators.** Current T5810 builds use the upstream `STRATA_HIP_GFX906` text engine and the fork HIP GPU vision encoder, both with ROCm 10. The fork wave64 text backend is deprecated for deployment. Both current components target the real `gfx906` architecture. See [current build selection](docs/GFX906.md#deployment-decision-which-gfx906-path-is-current-2026-10-06).

This fork is specifically tuned to run **Qwen3.8-Flash-Next** using model weights in **GGUF** format. The model's native/trained context length is **262,144 tokens**. The `--context` setting configures runtime capacity; hardware limits may justify choosing less. Native context is not an extended-RoPE setting, and it does not imply that full-length inference has completed acceptance testing on every system.

The gfx906 backend is experimental and must be enabled explicitly. The validation documented in this repository has been performed on MI50 hardware. MI60 is an intended same-architecture target, but has not been independently validated here.

## MI50 workload results

This section keeps **only the newest** workload observation. Older observations are archived under their
own dated headings in [`docs/GFX906.md`](docs/GFX906.md). Every number below is an operational observation
from the dual-MI50 T5810 server, **not a controlled benchmark**.

### Current result — v0.1.40 on ROCm 10 (2026-10-07, GSQ-RCO IQ3_S)

Engine `build-text-rocm10/strata` = `bc1102ba…` (upstream gfx906 path, source v0.1.40.1 / `82f46a8` plus
engine fix `74583c6`), vision encoder `build-vision-rocm10/bin/strata-vision` = `709fc4d2…`, both on
`10.0.0-gfx906+20260917140126`. Model GSQ-RCO IQ3_S, 262,144 configured context, `--kv int8`,
`--kv-resident 32768`, `--spec 4`, `--batch 4 --batch-groups 2`, `--layer-split 24`, `--pcie-frac 0`.

#### Prefill and decode rates

| Rate | Value | How it was measured |
| --- | ---: | --- |
| **Prefill — real (cold, no reuse)** | **486.7 tok/s** at 6,642 tokens; **646.2 tok/s** at 26,765; **670.2 tok/s** at 53,846 | Three one-shot probes with `cached_tokens: 0`; the engine's own `prompt_per_second` |
| **Prefill — effective (reused tokens counted)** | **14,904 tok/s** across the window; **29,129–246,878 tok/s** per request | `prompt_tokens / prompt_ms` on production traffic with a warm K/V cache |
| **Decode — production traffic** | **48.4–57.9 tok/s**, weighted **52.9 tok/s** | 12 request records in the window (`decode_tok_s`) |
| **Decode — probe runs** | **55.5–59.7 tok/s** | Same three probes, the engine's `predicted_per_second`, MTP `--spec 4` active |

The two prefill numbers answer different questions. **Real prefill** is compute: new prompt tokens divided
by prefill time, with no cache hit. **Effective prefill** divides the whole prompt, including tokens served
from the K/V cache, by the prompt-phase time. It shows how fast a long conversation turn is accepted; it is
not compute throughput. Do not compare them, and do not quote the effective number as a prefill speed.

#### Cold prefill and decode probe (bounded, 2026-10-07 03:26:40–03:29:24 UTC)

Three one-shot requests with unique text, so nothing was reused (`cached_tokens: 0`), `temperature 0`,
`max_tokens 64`, sent to the idle production service. The engine reported these timings:

| Prompt tokens | Reused | Prefill s | **Prefill tok/s** | **Decode tok/s** | MTP drafts accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6,642 | 0 | 13.65 | **486.7** | **59.7** | 30 / 36 |
| 26,765 | 0 | 41.42 | **646.2** | **57.7** | 30 / 34 |
| 53,846 | 0 | 80.34 | **670.2** | **55.5** | 30 / 36 |

- Prefill gets faster per token as the prompt grows: 486.7 tok/s at 6.6K, 670.2 tok/s at 53.8K.
- Decode stayed 55.5–59.7 tok/s while the prompt grew to 53,846 tokens.
- Cost: about 137 s of GPU prefill on the idle service. No restart, no config change, no model change.
- Raw payload: [`probe-results.json`](docs/gfx906-results/20261007-prefill-decode-probe/probe-results.json).
- Limits: synthetic filler text, one run per size, `max_tokens 64`; not a matched A/B against another build
or ROCm version, and not a long-soak measurement.

#### Production window (rolling `/metrics`, 2026-10-07 02:37:57 → 03:03:45 UTC)

A rolling `/metrics` window from the production `:8082` service, 2026-10-07 02:37:57 → 03:03:45 UTC
(25.8 min). The engine is the upstream gfx906 text engine `bc1102ba…` (source v0.1.40.1 / `82f46a8` plus
engine fix `74583c6`) on ROCm 10, with this fork's HIP GPU vision encoder loaded. The traffic was the
owner's agent sessions; no client-side timing was recorded.

| Item | Value |
| --- | --- |
| Requests | 38 |
| Prompt tokens | 3,699,741 |
| Reused K/V | 3,596,208 (**97.20%**) |
| New prompt tokens | 103,533 |
| Output tokens | 40,072 |
| Output over the whole window | 25.9 tok/s (includes idle gaps; not a decode rate) |
| Decode occupancy | 733.5 s of decode work inside a 1,548 s window (47.4% of one slot) |
| MTP (`--spec 4`) | 34,513 drafts offered, 26,529 accepted (**76.87%**) |
| Largest prompt | 134,007 tokens: 133,594 reused, 413 new |
| Conversation cache (8 GiB, 4 slots) | 4 sessions parked, 4.60 GiB; 6 parks, 1 restore, 1 eviction; 35 of 38 requests reused (92.1%) |
| VRAM resident | 61.35 / 63.97 GiB (the engine reports 2,186 MiB free) |
| Host RAM | 72.46 / 107.96 GiB |
| State at capture | idle: 0% GPU utilization, 42 W of a 450 W limit, 35 °C |

The 12 request records the payload keeps (the most recent of the 38) ran on one growing session,
126,422 → 134,007 tokens:

| Time (UTC) | Prompt tokens | Reused | New | Output | Decode tok/s | Duration s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 02:59:51 | 126,422 | 126,384 | 38 | 495 | 53.3 | 9.9 |
| 03:00:04 | 127,034 | 126,916 | 118 | 521 | 55.4 | 11.3 |
| 03:00:19 | 127,805 | 127,554 | 251 | 472 | 52.3 | 11.5 |
| 03:00:33 | 129,131 | 128,276 | 855 | 1,019 | 49.5 | 25.1 |
| 03:00:59 | 130,201 | 130,149 | 52 | 269 | 51.0 | 6.1 |
| 03:01:05 | 130,521 | 130,469 | 52 | 510 | 51.5 | 10.7 |
| 03:01:23 | 131,125 | 131,033 | 92 | 314 | 48.4 | 8.3 |
| 03:01:32 | 131,596 | 131,438 | 158 | 641 | 56.5 | 13.4 |
| 03:01:46 | 132,276 | 132,236 | 40 | 209 | 53.1 | 4.5 |
| 03:01:51 | 132,524 | 132,487 | 37 | 656 | 57.9 | 11.9 |
| 03:02:05 | 133,257 | 133,180 | 77 | 335 | 52.4 | 8.1 |
| 03:02:14 | 134,007 | 133,594 | 413 | 1,130 | 53.0 | 24.4 |
| **Sample total** | **1,565,899** | **1,563,716** | **2,183** | **6,571** | **52.9** (weighted) | **145.2** |

What this shows:

- K/V reuse holds at ~130K context in production. 99.86% of the sample's prompt tokens were reused, and
each turn read only 37–855 new tokens. Before fix `74583c6` the same server logged a 65,846-token prompt
with zero reuse.
- Single-stream decode stayed at 48.4–57.9 tok/s (weighted 52.9 tok/s) with four batch slots and the vision
encoder resident.
- Parking works under pressure: four sessions parked inside the 8 GiB budget, one eviction, and one restore
served a later turn.

What this does not show:

- This is not a controlled benchmark and not a matched A/B: one rolling window, mixed owner traffic, warm
cache, no client-side timing.
- A new-token prefill rate is **not** derivable from this payload: `prompt_ms` also covers cache restore
and admission work, so 103,533 new tokens over 248.2 s of `prompt_ms` is not a prefill throughput. The
measured cold prefill rates are in the probe subsection above.
- No accuracy, output-quality, full-262K-context, MI60, or long-soak measurement.
- The payload's `hardware_static.gpu_name` string reads "Radeon Instinct MI50 16GB"; `mem_total` reports
31.98 GiB per card, which matches the hardware. The name string is a monitor label, not a capacity claim.
- Raw payload: [`metrics-snapshot.json`](docs/gfx906-results/20261007-mi50-workload/metrics-snapshot.json).
The all-zero `history` sample arrays are omitted; every other key is verbatim.

## What this fork supports

- AMD Instinct **MI50 / MI60 (`gfx906`)** with Linux, ROCm, HIP, and hipBLAS. MI50 is the hardware validated so far; consult the guide before assuming MI60 compatibility in a particular system.
- **Qwen3.8-Flash-Next GGUF** weights. The engineering/validation (acceptance) configuration uses **GSQ-RCO IQ2_XS**; a **GSQ-RCO IQ3_S** pack is also prepared on the reference server and selected manually at launch (live observations above). Neither is the only supported quantization: the setup supports Qwen GGUF variants including Q2_0, IQ2_XS, IQ3_XXS, and IQ3_S; the Coder family also offers IQ1_M. Choose among available variants according to the desired quality, RAM/VRAM, disk capacity, and speed. The GGUF filename describes the pack variant, not every tensor's internal quant type.
- Single-GPU inference and experimental multi-GPU **contiguous-layer / pipeline splitting**. This is not tensor parallelism.
- CPU/GPU hybrid expert execution and speculative decoding with the supported MTP setup.

This is not a general claim that every feature or model in upstream Strata is available on gfx906. The HTTP server serializes requests by default; configured batch slots can decode concurrently. This fork adds an experimental Linux gfx906 HIP image encoder (`--experimental-gfx906 --vision gpu`) alongside upstream's CPU encoder. It expands BF16 vision weights losslessly to FP32 in memory; see [the measured scope and activation-arithmetic limits](docs/GFX906.md#experimental-hip-vision-on-v0139) before enabling it. The [current T5810 deployment](docs/GFX906.md#context-cache-fix-validated-on-real-models-2026-10-07) is the upstream gfx906 text engine built from the **v0.1.40.1 / `82f46a8`** merge for **ROCm 10** (`build-text-rocm10/strata`, the root-promoted validated `bc1102ba…` artifact with engine fix `74583c6`) plus this fork's HIP GPU vision encoder. Accepted scopes ([independent audit receipt](docs/gfx906-results/20261007-rocm10-context-cache/README.md)): four-slot pipeline/2-group ~32K-token parity with slot-cache reuse, 65K-token solo parking/reuse, 36K-token HTTP bursts, and Web CORS support. The 8082 service runs these two artifacts as of 2026-10-07 02:36 UTC (started by the owner, not by the agent); a read-only check at 02:43 UTC confirmed both processes load ROCm 10 only and that the vision encoder no longer links ROCm 7.2.4. See [the deployment follow-up](docs/GFX906.md#deployment-follow-up-2026-10-07-0243-utc-read-only).

## Install and build

### Requirements

- Linux with the AMD GPU driver/KFD and an existing **ROCm 10 HIP + hipBLAS** installation. The current deployment/build target is **ROCm 10**, on T5810 explicitly `/opt/rocm-10.0/core-10.0`; see [current build selection](docs/GFX906.md#deployment-decision-which-gfx906-path-is-current-2026-10-06). For the setup script, if ROCm is outside `/opt/rocm`, set `ROCM_PATH` to its installation directory. The example CMake preset itself points to `/opt/rocm`.
- An **AVX2-capable x86-64 CPU**; v0.1.31 explicitly rejects CPUs without AVX2. The tested Xeon E5-1650 v3 meets this requirement.
- Python 3.10 or newer with `venv`/`pip`, Git, and a C++ compiler. `setup.sh` creates a project-local `.venv` and installs the pinned Python dependencies there; the HIP engine is compiled locally for gfx906.
- Disk requirements depend on model and quantization. A clean **IQ2_XS** install needs roughly **80 GB or more** of free space across the model/data volume; keep at least 4 GiB free on the system volume. The installer checks required space before downloading. Use `--data-dir` to place model data on a larger disk.

### Archived experimental installer (build engine, prepare model, do not start yet)

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

**Next time**, run `./setup.sh` again with the same options: it starts right away and nothing large is
downloaded twice. `./update.sh` updates the source and rebuilds the engine and Python packages without
starting inference; on this fork it runs `git pull --ff-only` on the `gfx906` branch, so upstream merges
arrive as reviewed fork commits rather than through that pull. Model files are not touched. For gfx906,
always pass `--experimental-gfx906` again; see [docs/GFX906.md](docs/GFX906.md) rather than upstream's
RDNA wheel defaults.

## Start and stop

On T5810, use the existing `run-iq3-s.sh` / `run-iq2-xs.sh` launchers and the
[ROCm 10 build selection](docs/GFX906.md), rather than rebuilding the deprecated text backend
with the experimental installer/preset above. The install examples describe that archived path.

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

Web API CORS defaults to accepting any origin on `/v1/*`. Set `"cors_origins": []` in the saved
config to disable it, or supply an explicit origin list to restrict it.

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
| `--parallel N` | Number of concurrent requests to decode simultaneously (batch slots, 1..8; default 1). Enables multi-slot batching in the generated config (`"parallel": N`). |
| `--vision gpu\|cpu\|none` | Image processing mode. On gfx906, `--vision gpu` builds and configures the experimental HIP vision encoder (`strata-vision`) with FP32 weight expansion; `cpu` runs the encoder on host CPU threads. |
| `--vision-tokens N` | Maximum image tokens per picture (e.g. `--vision-tokens 768`). |
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
| `--slot-save-path DIR` | Directory for saving/restoring slot session state via `/slots/{id}?action=save|restore`. |

### Engine options

To change engine tuning, edit the generated `strata-iq2_xs.json` and preserve its existing required paths/flags. Change the corresponding values in the `args` array rather than adding duplicate flags. Common options include:

| Engine option in `args` | Purpose |
| --- | --- |
| `--max-context N` | Engine context capacity (normally set through installer option `--context`). |
| `--kv fp16|int8|q4_0|k8v4` | KV-cache representation. `k8v4` (int8 K + rotated Q4_0 V, 816 B/cell) now supports streaming with `--kv-resident`. |
| `--kv-grow` / `--no-kv-grow` | Elastic KV-cache sizing; allocates VRAM only for cells reached, freeing remainder for expert cache (also `STRATA_KV_GROW=1/0`). |
| `--expert-cache auto|N` | GPU-resident expert-cache sizing. |
| `--prefill auto|N` | Prompt-processing chunk size. |
| `--spec N`, `--mtp DIR`, `--spec-min-p P` | Speculative decoding/MTP settings. The generated config supplies the required MTP path and verifier settings. |
| `--batch N` / `--slots N` | Engine batch slot count for concurrent decoding (2..8). Each slot maintains independent KV cache, GDN recurrence, and PLE states across GPU stages. |
| `--batch-mtp` | Enables MTP speculative drafting inside concurrent batch slots (needs `--batch`, `--mtp`, and `--spec`; also `STRATA_BATCH_MTP=1`). |
| `--batch-groups G` | Number of pipeline groups across layer-split GPUs (e.g. `--batch-groups 2`). Allows GPUs to decode distinct slot groups concurrently. |
| `--pipeline-windows N` | Dual-GPU layer-split pipelined verify windows (`1` or `2`); overlaps verification and prompt reading across cards. |
| `--trim-stage-weights` | With explicit `--layer-split`, keeps each GPU from allocating non-resident dense layers, freeing VRAM for batch slots and expert cache. |
| `--vram-reserve-later-mib N` | Separate VRAM headroom reserved on subsequent GPUs in a layer split. |
| `--lookup-chain K` | Chained prompt-lookup speculative drafting after MTP proposals (up to K tokens; `--lookup-chain-min M` sets prefix threshold, default 3). |
| `--mtp-q4 all|proj|head` | 4-bit quantized copies of MTP draft projections and/or draft head to reduce draft memory bandwidth. |
| `--mtp-hnorm pooled|stream` | Draft layer hidden-input RMS normalization mode (pooled over all 4 streams or per-stream). |
| `--adapt-async 1` | Asynchronous adaptive expert tier swapping on a helper thread with resident RAM mode. |
| `--prompt-cache N`, `--suffix-draft N` | Conversation/prompt reuse and suffix-based draft options. Prompt caching does not make the server concurrent. |
| `--prompt-cache-tail` | Single GPU: extra prompt checkpoint near prompt end at an existing chunk boundary. |
| `--pool-workers N` | CPU expert-pool worker count; this does not set request concurrency. |
| `--host-core first|last` | Host thread CPU core pinning (`first` physical core by default, or `last`; also `STRATA_HOST_CORE`). |
| `--vision` | Enables image token embedding ingestion support in the engine. |
| `--vram-reserve-mib N` | Headroom reserved in GPU VRAM for the vision encoder process (e.g. `1024`). |
| `--resident-budget-gib N` | RAM-tier budget for GGUF-in-place experts; remaining experts use the SSD/file tier. Experimental on gfx906 until revalidated. |

With a multi-GPU config (`"gpu": [0, 1]`), the server wrapper supplies `--layer-split` from the config automatically; do **not** add a second `--layer-split` to `args`. The HTTP server serializes requests by default, but decodes concurrently up to the configured batch slots (`"parallel": N` in config / `--batch N` in engine args). API request fields such as `max_tokens` control the response length; they are not installer startup flags. This engine does not take llama.cpp-style `-c`, `-ngl`, or `--tensor-split` options.

For the engine's complete option list, run `./engine/strata --help` after installation. The archived **v0.1.30** MI50 regression run had **40 passed, 1 skipped and 0 failed** out of 41 tests, plus **18/18** Python entrypoint/download-safety tests. Skips and excluded tests are described in the [evidence notes](docs/gfx906-results/20261001/README.md); these software checks do not establish production readiness or long-run stability.

## Project background

This repository tracks upstream Strata through **v0.1.40.1 / `82f46a8`** while maintaining its own gfx906
compatibility and optimization work. The upstream project still excludes wave64; this fork's purpose is
to keep MI50 / MI60 usable as upstream evolves. It is not the upstream project's general NVIDIA/CUDA
release. Upstream source: [Niko1221/Strata](https://github.com/Niko1221/Strata). For architecture details,
compatibility limitations, source-sync validation, and reproducible evidence, start with the
[gfx906 development guide](docs/GFX906.md). The deployed workload observations above do not validate
newer source or establish a post-merge speedup. General upstream model choices, installation, MCP tools
and architecture explanations are in [MODELS](docs/MODELS.md), [INSTALL](docs/INSTALL.md),
[MCP_SERVER](docs/MCP_SERVER.md) and [HOW_IT_WORKS](docs/HOW_IT_WORKS.md); use this fork's MI50 instructions
rather than their RDNA wheel/prebuilt defaults.

## Where things are stored

- **Chats stay in the browser.** The Chat tab keeps conversation history, settings and the entered API key
  in local storage (`strata.*` keys), not in server files. Another browser or private session starts empty;
  clearing site data removes those chats. Image-input validation and encoder precision limits are
  recorded in [the gfx906 guide](docs/GFX906.md#experimental-hip-vision-on-v0139).
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
