# 尽量使用 ASD-STE100（Simplified Technical English）

# strata-gfx906 — Development Rules

This is a dedicated experimental MI50/MI60 gfx906 fork of `Niko1221/Strata`, synchronized through
v0.1.40.1 / `82f46a8` (original port: v0.1.30 / `30ec18e`).
The working branch is `gfx906`; `origin` is `webzone/strata-gfx906`. Absorb upstream updates with
reviewed merge commits, never rebasing published fork history or pushing to `upstream`.
Preserving and optimizing real gfx906 support is the fork's purpose: upstream's wave32-only GPU
support must not replace it. MI50 has archived hardware validation; MI60 still needs its own testing.
All project documentation and maintenance notes must be written in English. Preserve raw test inputs and captured output verbatim as evidence.

1. **Read the device record before operating the machine.** The T5810 authority record is
   `/Users/chris/dev/local_devices/t5810/README.md`. Use its documented `chris@t5810` SSH entry;
   do not guess accounts, scan ports, or search unrelated projects for credentials.
2. **Protect connectivity and existing workloads.** Before builds/tests, inspect GPU, RAM, disk,
   Docker, and listener state. Do not stop workloads to free resources. Do not change drivers, ROCm,
   networking, thermal/tuning services, or restore historical containers. Keep project dependencies
   in this repository's `.venv`; install no system packages.
3. **Make gfx906 compatibility opt-in.** `STRATA_EXPERIMENTAL_GFX906` defaults OFF; the installer
   requires `--experimental-gfx906`. Compile for the real gfx906 architecture and retain runtime
   architecture checks. Never impersonate a GPU or use `HSA_OVERRIDE_GFX_VERSION`. Do not substitute
   RDNA TheRock wheels; there is no gfx906 wheel index.
4. **Preserve logical wave32 on physical wave64.** Do not mechanically change every `32` to `64`.
   Select ballot halves using linear block-thread indexing. Numerically test 2D/3D blocks, signed
   INT8 dot4 behavior/overflow, and shuffles. Recalibrate the gfx906 clock conversion on every test card.
5. **Use layer/pipeline splitting, not tensor parallelism.** Each GPU owns its layers and caches.
   Obtain a separate mapped-host alias on each device and synchronize the producer before the consumer
   reads. Do not require P2P or run dual-GPU tests while either GPU is busy. Tests enabled by
   `STRATA_HIP_MULTIGPU_TESTS=ON` require two visible cards; a missing-card skip is not a pass.
6. **Pin dependencies and model identity.** Pin llama.cpp/ggml to
   `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The acceptance model is GSQ-RCO IQ2_XS; use the
   Hugging Face revision and LFS SHA256 values in the tools. A filename is not the type of every
   tensor: this model mixes IQ2_S, IQ2_XXS, and IQ1_M gate/up tensors with Q2_0 down tensors. Do not
   force every expert through MMQ or write a duplicate ~35.5 GB main-model `experts.bin`; use the
   existing resident-from-GGUF path.
7. **Protect disk space and source data.** Keep at least 4 GiB free during downloads/preparation.
   Calculate peak space for GGUF, float pack, MTP source tensors, and quantized/runtime outputs first.
   Resume only with validated source identity and ranges; never overwrite a conflicting/corrupt file
   or delete unrelated models/images to make room.
8. **Report evidence accurately.** An independent CPU dequantizer/double-accumulation oracle is a
   synthetic operator check; a dual-GPU handoff probe is not model parity. Disclose skipped tests,
   missing fixtures, and sampled-only validation. Report model tok/s only from real generation; do
   not convert dot4 microbenchmarks or warm-cache operator timings into model speed.
9. **Test, document, commit, and push changes.** Commit related changes to `origin/gfx906`; never
   force-push or discard another author's work. Update `docs/GFX906.md`. After every upstream merge,
   update the current source-version line at the top of `README.md` and its current version/commit
   references to match the merged source, in the same merge commit. Preserve historical benchmark and
   deployment version labels; never relabel old evidence as the new version. Keep unrelated machine
   diagnostics out of this project. Do not commit weights, binaries, virtual environments, or full
   compiler-warning logs; small, auditable validation evidence may be committed.
10. **Keep only the newest workload result in `README.md`, and always name both rates.** The *MI50 workload
   results* section holds one observation: the latest. Move the previous one to `docs/GFX906.md` under its
   own dated heading, with its original version and ROCm labels intact, before adding the new one. State the
   decode rate and the prefill rate explicitly. Give prefill in two labelled forms: the **real** rate (new
   prompt tokens per prefill second, measured with `cached_tokens: 0`) and the **effective** rate (the whole
   prompt per prompt-phase second, reused K/V tokens counted). The effective number shows how fast a warm
   turn is accepted; it is not compute throughput, so never quote it as a prefill speed. Record how each
   number was measured, and keep the raw payload under `docs/gfx906-results/`.

## Upstream navigation

When invoking `omp`, use the `herdr` skill from a Herdr-managed pane, as requested by the owner.
The primary agent/thread owns only architecture, organization, review, and integration closeout;
concrete tasks are delegated through `herdr` to fresh omp instances, at most four concurrent, each
with a disjoint file scope (worktrees allowed as needed). When a task ends, the primary closes that
round's omp instances and removes only that round's worktrees/branches, never another round's leftovers.

Upstream's general user workflow is in `docs/AI_SETUP.md` and `docs/MCP_SERVER.md`; for this fork's
MI50/MI60 installation, use `docs/GFX906.md` and the rules above instead of upstream's RDNA/prebuilt defaults.
The owner-authorized T5810 private-LAN deployment uses `0.0.0.0:8082` without an API key; do not add
one unless the owner requests it. Other non-loopback deployments require an API key. Engine/API details are in `docs/DETAILS.md`,
AMD upstream scope in `docs/AMD_HIP.md`, and pipeline splitting in `docs/MULTI_GPU.md`.
GPU-independent setup tests run with `.venv/bin/python -m unittest discover -s tools -p 'test_setup_*.py'`.
Document measured results with their hardware, workload and validation limits, not unsupported performance claims.

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
