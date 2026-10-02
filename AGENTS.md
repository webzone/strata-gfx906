# strata-gfx906 — Development Rules

This is a dedicated experimental MI50/MI60 gfx906 fork of `Niko1221/Strata`, synchronized through
v0.1.34 / `1678de333d0e0711bc414ad992b640e1a37dd814` (original port: v0.1.30 / `30ec18e`).
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
   force-push or discard another author's work. Update `docs/GFX906.md`. Keep unrelated machine
   diagnostics out of this project. Do not commit weights, binaries, virtual environments, or full
   compiler-warning logs; small, auditable validation evidence may be committed.

## Upstream navigation

Upstream's general user workflow is in `docs/AI_SETUP.md` and `docs/MCP_SERVER.md`; for this fork's
MI50/MI60 installation, use `docs/GFX906.md` and the rules above instead of upstream's RDNA/prebuilt defaults.
Never expose a server beyond loopback without an API key. Engine/API details are in `docs/DETAILS.md`,
AMD upstream scope in `docs/AMD_HIP.md`, and pipeline splitting in `docs/MULTI_GPU.md`.
GPU-independent setup tests run with `.venv/bin/python -m unittest discover -s tools -p 'test_setup_*.py'`.
Document measured results with their hardware, workload and validation limits, not unsupported performance claims.
