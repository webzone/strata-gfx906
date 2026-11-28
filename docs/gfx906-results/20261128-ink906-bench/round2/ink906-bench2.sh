#!/usr/bin/env bash
# ink906 benchmark round 2 (item 1: two cards + tiled MoE) on the T5810 (owner-approved 2026-11-28). Stops the 8080 strata server, runs strata-inkling --bench, restarts the server.
set -uo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ); OUT=$HOME/dev/strata-gfx906/logs/ink906-bench-$TS; mkdir -p "$OUT"
EXE=$HOME/.local/share/strata/components/shapes/inkling2/engine/0.1.70-linux-x86_64-hip-local/strata-inkling
M=$HOME/.local/share/strata/Strata-data/models/inkling-2-mini/Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf
SMI=/opt/rocm-10.0/core-10.0/bin/rocm-smi
S=$OUT/summary.txt
say() { printf '%s\n' "$*" | tee -a "$S"; }
restart() {
  say "=== restart the server"
  (cd ~ && timeout 900 ~/.local/bin/strata launch --preset gsq-rco-iq3_s-256k --no-tui --yes) > "$OUT/restart.log" 2>&1; say "restart exit $?"
  tail -5 "$OUT/restart.log" | tee -a "$S"
  ss -ltn | grep -q ':8080 ' && say "8080 listening" || say "WARNING: 8080 not listening"
  say "=== DONE"
}
say "bench $TS · sha256 $(sha256sum "$EXE" | cut -c1-16) · model $(stat -c%s "$M") B"
$SMI --showmeminfo vram --showuse > "$OUT/smi-before.txt" 2>&1
pgrep -af 'strata-inkling|serve.server' | cut -c1-160 > "$OUT/procs-before.txt"
say "=== stop the server"
(cd ~ && timeout 120 ~/.local/bin/strata launch --stop) > "$OUT/stop.log" 2>&1; say "stop exit $?: $(tail -2 "$OUT/stop.log" | tr '\n' ' ')"
trap restart EXIT
for i in $(seq 60); do pgrep -f 'strata-inkling|serve.server' >/dev/null || break; sleep 2; done
if pgrep -af 'strata-inkling|serve.server' >/dev/null; then say "FAIL: a Strata process still runs"; exit 1; fi
sleep 3; $SMI --showmeminfo vram --showuse --showpids > "$OUT/smi-idle.txt" 2>&1
say "idle: $(grep -E 'VRAM Total Used' "$OUT/smi-idle.txt" | tr -s ' ' | tr '\n' ' ')"
run() {
  local name=$1; shift
  local t0=$(date +%s)
  env "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  say "$name exit $rc ($(( $(date +%s) - t0 )) s): $(grep -h '^\[bench\] backend' "$OUT/$name.log" | tail -1)"
  grep -h -E 'EP2 active|KV cache|max_prefill_chunk|chunk' "$OUT/$name.log" | head -4 | cut -c1-200 | sed 's/^/    /' | tee -a "$S"
}
B="$EXE --native $M --bench --prompt-len 16384 --gen 256 --max-context 262144"
D2="--devices 1,0 --vram-experts-gb 11.99"
run G_dual_tiles_fp8      STRATA_HIP_RDNA2_MOE=1 timeout 1500 $B $D2 --kv fp8
run H_dual_tiles_fp8_c2k  STRATA_HIP_RDNA2_MOE=1 timeout 1500 $B $D2 --kv fp8 --chunk 2048
run I_dual_tiles_fp16_c2k STRATA_HIP_RDNA2_MOE=1 timeout 1500 $B $D2 --kv fp16 --chunk 2048
