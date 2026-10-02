# T5810 IQ3_S preparation — no inference, startup or tests

The owner requested the original GSQ-RCO **IQ3_S**, not IQ3_XXS, alongside the existing IQ2_XS, with manual
selection of one model at a time. After clearing disk space, the owner chose the original project directory
instead of `/data`. Preparation finished at 2026-10-02 04:22 UTC. No new model/server was started.

## Prepared layout

All paths are under `/home/chris/dev/strata-gfx906`:

| Slot | GGUF | Pack | Config | Foreground launcher |
| --- | --- | --- | --- | --- |
| IQ2_XS | Existing `models/IQ2_XS` | Existing `packs/iq2_xs` | `run-iq2-xs.json` | `run-iq2-xs.sh` |
| IQ3_S | New `models/IQ3_S` | New `packs/iq3_s` | `run-iq3-s.json` | `run-iq3-s.sh` |

Both use the same real gfx906 `build-hip/strata` binary and project `.venv`, with 0.0.0.0:8082 and no API key
at the owner's explicit private-LAN preference. New config mode is 0600. Sampling, GPU order `[0, 1]`,
262,144-token context, int8 KV, MTP/spec settings, split 24 and the three ROCm environment entries were retained.
The IQ2 config stayed byte-identical; the engine SHA256 is unchanged:
`d8a59558877074f605116e396f305d0d1d1cf8a7c2f2512918b1c4fbb9f440cb`.
Only the IQ2 launcher was updated to use the common pre-start model lock/port guard.

## Identity, storage and packing

Repo `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, pinned revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- IQ3_S shard 1: 54,817,524,224 bytes;
  SHA256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`.
- IQ3_S shard 2: 28,800,138,432 bytes;
  SHA256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.

Both full GGUF hashes were checked. Shard 2 is byte-identical to IQ2_XS's PLE shard and was hard-linked after
validation: same device/inode, two names, no extra 28.8 GB allocation. These weight files must not be modified
in place. Removing one name does not remove the other's hard link.
The first shard alone was downloaded, with a 40 MiB/s aggregate cap, validated source/ranges and a 4 GiB floor.
There was no copying/migration of IQ2_XS, duplicate program version, `/data` use or unrelated deletion.

Initial root free was about 111 GiB. Conservative peak budget: 51.05 GiB new GGUF + 20 GiB pack allowance
+ 4 GiB safety floor; MTP additional source/runtime bytes were zero because the existing pinned common runtime
was reused (`de4b8e4d43b917e7706784d8bb445c9af86a3540`). The existing MTP source verifier exited 0; it may use
size/mtime/SHA256 stamps, so this is not a claim of freshly hashing every MTP tensor or numerical model parity.
Actual headers predicted 1,538,035,200 dense bytes, and `tools/iq_pack.py --compat-bf16` produced exactly that
size. Total pack/metadata is 1,547,978,412 bytes (~1.442 GiB); root free remained ~57.96 GiB.
No main-model `experts.bin` was generated and the PLE table was not expanded. Small projection conversions
may round to BF16; native expert/PLE/source GGUF bytes are unchanged.

The published IQ3_S recipe is mixed: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS gate/up tensors, and IQ4_NL/Q2_0 down tensors.
That is not an IQ3_XXS model substitution. The receipt records all 144 expert tensor types; filenames do not
justify forcing them through one MMQ type.

## Evidence boundary

- `preparation-receipt.json`: identity, exact commands, disk/RAM planning, unchanged IQ2 config and no-start boundary.
- `download-integrity.log`: small raw SHA256/completion output, extracted verbatim from the complete private log.
- `pack.log`: complete small raw packing/tokenizer output; no weights included.
- `mtp-integrity.log`: empty successful verifier stdout; its exit code is in the receipt.
- `file-checks.json`: static config/binary/shell-syntax checks, without executing a launcher or model.

Full private inputs/status/logs remain in `logs/prepare-IQ3_S-20261002T035542Z/`; worker input is
`logs/iq3-s-prepare-k2vskkh4.py`. Preparation used lower CPU/I/O priority, no system/project package installs,
no driver/ROCm/network/Docker/tuning/thermal changes, and no service stop/start/restart commands.
At preflight an IQ2 server occupied 8082; at completion it was unbound. The later exit cause was not established
and the preparer did not automatically restore it.

**No CTest, GPU probe, model/API/image request, generation benchmark or new-model startup was performed.**
File hashing/packing/syntax checks are preparation integrity, not operator/model acceptance. IQ3_S's configured
262K/int8/MTP behavior, memory fit, token parity, speed and stability on MI50 remain unvalidated.

## Manual selection

On T5810, use one of these after manually exiting any current server:

```bash
cd /home/chris/dev/strata-gfx906
./run-iq2-xs.sh       # existing IQ2_XS
# OR, instead of IQ2_XS:
./run-iq3-s.sh        # prepared IQ3_S
```

`tools/gfx906_launch.sh` holds one project model lock and rejects an occupied 8082 before starting any engine.
It never kills a running service. There is no autoboot or automatic model switch, and no API key is required.
