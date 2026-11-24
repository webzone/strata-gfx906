#!/usr/bin/env bash
# t5810 · v0.2.20 (upstream tag v0.2.20 / a34979fc9 plus 16 commits, upstream main 6a6db7ff5; fork
# merge given with --merge): sync the machine checkout, compile the gfx906 text engine and the picture
# reader, install the text engine only. No stop, no restart, no model test, no ctest. The script
# verifies its own result: source identity, the vendored pin, every hash before and after, the engine
# version string, the run config, the expert-profile copy, and the running PIDs.
# Usage: deploy-v0220.sh --merge <fork-commit> [--dry-run] [--no-build] [--with-vision]
set -euo pipefail
ROOT=/home/chris/dev/strata-gfx906
UPSTREAM=6a6db7ff5       # upstream main this merge carries (tag v0.2.20 = a34979fc9, plus 16 commits)
ENGINE=0.1.66            # project(strata VERSION) of this source
ROC=/opt/rocm-10.0/core-10.0
CONFIG=$ROOT/run-iq3-s.json
CONFIG_MD5=e996f0f9e1e60b2dade05f188ee8c4e3   # run-iq3-s.json as the owner last edited it (2026-11-17 19:17 UTC)
CUR_TEXT=$ROOT/build-text-rocm10/strata
CUR_VIS=$ROOT/build-vision-rocm10/bin/strata-vision
TEXT_DIR=$ROOT/build-text-rocm10-v0220
VIS_DIR=$ROOT/build-vision-rocm10-v0220
SCRIPT=/home/chris/mi50-t5810/scripts/strata-current.sh
EXPECT_CUR_TEXT=eb789fe3e9a76f355c01220ae1a3cf69   # md5 of the deployed 0.1.62 engine (v0.2.16 round)
EXPECT_CUR_VIS=eafb5d76f03635a2f8424a6ef5cda41f    # md5 of the deployed picture reader
BAK=$ROOT/build-text-rocm10/strata.v0216-a9c9ac4a.bak
BAK_SHA=a9c9ac4a7347060393dc5a08fac33a976a268f73cc06e4797d398b4d389dea3b
PROFILE=$ROOT/data/expert-profile.bin              # untracked copy the run config names (restart fix)
PROFILE_SRC=$ROOT/terminal/data/expert-profile.bin
RAM_FLOOR_MIB=4096       # abort the build if MemAvailable drops under this
MIN_FREE_GIB=4           # rule 7: never build with less free disk than this
JOBS=4                   # 12 cores; the live trio keeps running

MERGE=""; DRY=0; NOBUILD=0; WITH_VISION=0
while [ $# -gt 0 ]; do
  case "$1" in
    --merge) MERGE=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    --no-build) NOBUILD=1; shift;;
    --with-vision) WITH_VISION=1; shift;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
[ -n "$MERGE" ] || { echo "usage: deploy-v0220.sh --merge <fork-commit>"; exit 2; }

h() { sha256sum "$1" | cut -d" " -f1; }
m() { md5sum "$1" | cut -d" " -f1; }
say() { printf '%s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
free_mib() { awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
free_gib() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOGD=$ROOT/logs/v0220-build-$TS

say "=== 1. machine state ==="
mkdir -p "$LOGD"
ss -ltnp 2>/dev/null | grep ':8082' | head -2 | tee "$LOGD/listeners-before.txt" || say "8082: not listening"
ps -eo pid,etime,cmd | grep -E 'serve/server.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' \
  | grep -v grep | tee "$LOGD/procs-before.txt" || say "no Strata processes"
say "MemAvailable: $(free_mib) MiB · free disk: $(free_gib) GiB"
rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'used' | tee "$LOGD/vram-before.txt" | head -4
docker ps --format '{{.Names}}' > "$LOGD/docker.txt" 2>/dev/null || true
[ "$(free_gib)" -ge "$MIN_FREE_GIB" ] || fail "less than ${MIN_FREE_GIB} GiB free on $ROOT (rule 7)"
[ "$(free_mib)" -ge 3000 ] || fail "MemAvailable under 3000 MiB: the build would push the live engine out (rule 2)"
[ "$(m "$CUR_TEXT")" = "$EXPECT_CUR_TEXT" ] || fail "deployed text hash differs from $EXPECT_CUR_TEXT"
[ "$(m "$CUR_VIS")" = "$EXPECT_CUR_VIS" ] || fail "deployed picture reader hash differs from $EXPECT_CUR_VIS"
[ "$(m "$CONFIG")" = "$CONFIG_MD5" ] || fail "run config changed from $CONFIG_MD5"
python3 -c "import json;json.load(open('$CONFIG'))" || fail "run config not parseable"
[ -f "$PROFILE" ] || fail "$PROFILE missing: the run config names it and the engine dies without it"
[ "$(h "$PROFILE")" = "$(h "$PROFILE_SRC")" ] || say "note: $PROFILE differs from $PROFILE_SRC (the copy is the one the config reads)"
say "deployed text engine: $("$CUR_TEXT" --version 2>&1 | head -1)"

if [ "$DRY" = 1 ]; then say "DRY RUN: nothing changed."; exit 0; fi

say "=== 2. checkout ==="
git -C "$ROOT" fetch -q origin gfx906 || fail "git fetch failed"
HEAD_NOW=$(git -C "$ROOT" rev-parse --short HEAD)
if [ "$HEAD_NOW" != "$MERGE" ]; then
  git -C "$ROOT" merge --ff-only origin/gfx906 || fail "checkout is not fast-forwardable from $HEAD_NOW"
fi
HEAD_NOW=$(git -C "$ROOT" rev-parse --short HEAD)
[ "$HEAD_NOW" = "$MERGE" ] || fail "checkout at $HEAD_NOW, expected $MERGE"
git -C "$ROOT" merge-base --is-ancestor "$UPSTREAM" HEAD || fail "upstream $UPSTREAM is not in $HEAD_NOW"
say "checkout: $HEAD_NOW (upstream $UPSTREAM inside)"
git -C "$ROOT" log -1 --format='%H %s' > "$LOGD/checkout.txt"
git -C "$ROOT" ls-files .github/workflows > "$LOGD/workflows.txt"
if grep -q . "$LOGD/workflows.txt"; then fail "the merged tree carries workflow files: $(cat "$LOGD/workflows.txt")"; fi
say "workflow files in the tree: $(wc -l < "$LOGD/workflows.txt")"
# What the merge touched, so the rebuild claim is auditable.
git -C "$ROOT" diff --stat ba64bc244..$HEAD_NOW > "$LOGD/scope-stat.txt"
for a in engine terminal app docs tools; do
  say "scope $a/: $(git -C "$ROOT" diff --name-only ba64bc244..$HEAD_NOW -- $a | wc -l) paths"
done
CHECK=$( [ -f "$ROOT/tools/check_vendored.py" ] && echo "$ROOT/tools/check_vendored.py" || echo "$ROOT/terminal/tools/check_vendored.py" )
"$ROOT/.venv/bin/python" "$CHECK" > "$LOGD/vendored.txt" 2>&1 \
  || fail "$CHECK failed (see $LOGD/vendored.txt)"
grep -qiE 'ok|3cf03257f' "$LOGD/vendored.txt" || fail "$CHECK reported nothing usable"
say "vendored ggml pin: ok ($CHECK: $(head -1 "$LOGD/vendored.txt"))"

# One venv per checkout: terminal/ links the existing .venv, never a second environment (rule 2).
if [ ! -e "$ROOT/terminal/.venv" ]; then
  ln -s ../.venv "$ROOT/terminal/.venv"
  say "terminal/.venv: linked to the existing $ROOT/.venv"
fi
[ -x "$ROOT/terminal/.venv/bin/python" ] || fail "terminal/.venv/bin/python is not usable"
"$ROOT/terminal/.venv/bin/python" -c "import sys;print('terminal venv python', sys.version.split()[0])" \
  || fail "terminal/.venv/bin/python does not run"
"$ROOT/terminal/.venv/bin/python" -m py_compile "$ROOT/terminal/serve/server.py" \
  || fail "terminal/serve/server.py does not compile with the checkout venv"
say "terminal/serve/server.py compiles with the checkout venv"

TEXT_SRC=$ROOT
[ -f "$ROOT/CMakeLists.txt" ] || TEXT_SRC=$ROOT/engine
[ -f "$TEXT_SRC/CMakeLists.txt" ] || fail "no CMakeLists.txt in $ROOT or $ROOT/engine"
say "text engine source: $TEXT_SRC (version $(grep -m1 -oE 'VERSION [0-9.]+' "$TEXT_SRC/CMakeLists.txt"))"

VIS_SRC=tools/vision
[ -f "$ROOT/tools/vision/CMakeLists.txt" ] || VIS_SRC=terminal/tools/vision
[ -f "$ROOT/$VIS_SRC/CMakeLists.txt" ] || fail "no fork picture-reader CMakeLists in tools/vision or terminal/tools/vision"
say "picture reader source: $VIS_SRC"

build_one() {  # source dir, build dir, cmake args
  local src=$1 dir=$2; shift 2
  say "--- configure+build $dir (source $src)"
  if ! ( cd "$ROOT" && cmake -S "$src" -B "$(basename "$dir")" -G Ninja "$@" ) > "$LOGD/configure-$(basename "$dir").log" 2>&1; then
    fail "configure of $dir failed (see $LOGD/configure-$(basename "$dir").log)"
  fi
  grep -iE 'gfx906|HIP|ggml commit' "$LOGD/configure-$(basename "$dir").log" | head -5
  ( cd "$ROOT" && cmake --build "$(basename "$dir")" -j "$JOBS" ) > "$LOGD/build-$(basename "$dir").log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    [ "$(free_mib)" -ge "$RAM_FLOOR_MIB" ] || { pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; fail "MemAvailable under $RAM_FLOOR_MIB MiB during $dir: build stopped, nothing installed"; }
    sleep 10
  done
  wait "$pid" || fail "build of $dir failed (see $LOGD/build-$(basename "$dir").log)"
  say "build log: $(grep -c '' "$LOGD/build-$(basename "$dir").log") lines · errors: $(grep -c 'error:' "$LOGD/build-$(basename "$dir").log") · warnings: $(grep -c 'warning:' "$LOGD/build-$(basename "$dir").log") · targets: $(grep -cE '^\[' "$LOGD/build-$(basename "$dir").log")"
}

if [ "$NOBUILD" = 0 ]; then
  say "=== 3. build (the live trio keeps running) ==="
  build_one "$TEXT_SRC" "$TEXT_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DCMAKE_HIP_COMPILER=$ROC/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROC \
    -DCMAKE_PREFIX_PATH=$ROC -DSTRATA_PORTABLE=ON -DSTRATA_PREFILL_MMQ=OFF \
    -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  [ -x "$TEXT_DIR/strata" ] || [ -x "$TEXT_DIR/bin/strata" ] || fail "no strata binary under $TEXT_DIR"
  [ -x "$TEXT_DIR/strata" ] || mv "$TEXT_DIR/bin/strata" "$TEXT_DIR/strata"
  VER=$("$TEXT_DIR/strata" --version 2>&1 | head -1)
  case "$VER" in *"$ENGINE"*) say "engine version string: $VER";; *) fail "$TEXT_DIR/strata reports '$VER', expected $ENGINE";; esac
  SRC_TEXT_HASH=$(h "$TEXT_DIR/strata")
  say "text sha256: $SRC_TEXT_HASH ($(stat -c %s "$TEXT_DIR/strata") B)"
  build_one "$VIS_SRC" "$VIS_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_VISION_HIP=ON -DSTRATA_VISION_CUDA=OFF -DSTRATA_VISION_HIP_ARCHS=gfx906 \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_HIP_COMPILER=$ROC/lib/llvm/bin/clang++ \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROC -DCMAKE_PREFIX_PATH=$ROC -DSTRATA_PORTABLE=ON
  [ -x "$VIS_DIR/bin/strata-vision-hip" ] || fail "no $VIS_DIR/bin/strata-vision-hip"
  SRC_VIS_HASH=$(h "$VIS_DIR/bin/strata-vision-hip")
  say "picture reader sha256 (compile check): $SRC_VIS_HASH ($(stat -c %s "$VIS_DIR/bin/strata-vision-hip") B)"
else
  SRC_TEXT_HASH=$(h "$TEXT_DIR/strata")
  SRC_VIS_HASH=$(h "$VIS_DIR/bin/strata-vision-hip" 2>/dev/null || echo "")
  say "=== 3. --no-build: reusing $TEXT_DIR/strata ($SRC_TEXT_HASH)"
fi

say "=== 4. install the text engine (no restart) ==="
cp -a "$CUR_TEXT" "$BAK"
[ "$(h "$BAK")" = "$BAK_SHA" ] || fail "backup $BAK hash != $BAK_SHA"
install -m 0755 "$TEXT_DIR/strata" "$CUR_TEXT"
[ "$(h "$CUR_TEXT")" = "$SRC_TEXT_HASH" ] || fail "installed text hash != the built hash"
say "installed text engine: $CUR_TEXT = $SRC_TEXT_HASH"
say "engine reports: $("$CUR_TEXT" --version 2>&1 | head -1)"

if [ "$WITH_VISION" = 1 ]; then
  cp -a "$CUR_VIS" "$ROOT/build-vision-rocm10/bin/strata-vision.v0216-943a54bd.bak"
  install -m 0755 "$VIS_DIR/bin/strata-vision-hip" "$CUR_VIS"
  [ "$(h "$CUR_VIS")" = "$SRC_VIS_HASH" ] || fail "installed picture reader hash != the built hash"
  say "installed picture reader: $CUR_VIS = $SRC_VIS_HASH"
else
  say "picture reader left untouched: $CUR_VIS stays $(m "$CUR_VIS") (the fork GPU reader on this card is validated at that build; the new one is compile-verified only)"
fi

say "=== 5. the running trio is unchanged ==="
ps -eo pid,etime,cmd | grep -E 'serve/server.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' \
  | grep -v grep > "$LOGD/procs-after.txt"
if diff -q <(awk '{print $1}' "$LOGD/procs-before.txt") <(awk '{print $1}' "$LOGD/procs-after.txt") >/dev/null; then
  say "same PIDs before and after: $(awk '{print $1}' "$LOGD/procs-before.txt" | tr '\n' ' ')"
else fail "process list changed: nothing should have been started or stopped"; fi
say "health: $(curl -s -m 10 http://127.0.0.1:8082/health | head -c 160)"
say "metrics engine version still: $(curl -s -m 10 http://127.0.0.1:8082/metrics | python3 -c 'import json,sys;print(json.load(sys.stdin)["engine"]["version"])')"
say "the running engine keeps its old inode: $(ls -l /proc/$(pgrep -f 'build-text-rocm10/strata --serve' | head -1)/exe | sed 's/.*exe -> //')"
say "run config md5 still: $(m "$CONFIG") · expert-profile copy still: $(h "$PROFILE" | cut -c1-16)…"
say "free disk now: $(free_gib) GiB · MemAvailable now: $(free_mib) MiB"

say "=== 6. strata-current.sh expectation ==="
cp "$SCRIPT" "$SCRIPT.bak-$TS"
python3 - "$SCRIPT" "$SRC_TEXT_HASH" <<'PY'
import sys, re
path, want = sys.argv[1], sys.argv[2]
txt = open(path).read()
old = re.search(r"^EXPECT_TEXT=([0-9a-f]{64})", txt, re.M).group(1)
txt = txt.replace(f"EXPECT_TEXT={old}", f"EXPECT_TEXT={want}")
txt = txt.replace('EXPECT_TEXT_SUPERSEDED="', 'EXPECT_TEXT_SUPERSEDED="' + old + " ")
open(path, "w").write(txt)
print("EXPECT_TEXT %s -> %s; %s added to the superseded list" % (old[:8], want[:8], old[:8]))
PY
bash "$SCRIPT" | tail -12 || true   # exits 1 until the owner restarts onto the new file

say "=== 7. build identity ==="
python3 - "$LOGD" "$TS" "$MERGE" "$SRC_TEXT_HASH" "$SRC_VIS_HASH" "$VIS_SRC" "$HEAD_NOW" <<'PY'
import json, os, sys
logd, ts, merge, text_hash, vis_hash, vis_src, head = sys.argv[1:8]
root = os.path.dirname(os.path.dirname(logd))   # logd is $ROOT/logs/v0220-build-<ts>
rec = {
 "upstream": "6a6db7ff5", "upstream_tag": "v0.2.20", "merge": merge, "checkout": head,
 "engine_version": "0.1.66", "min_engine": "(0, 1, 46)",
 "llama_cpp": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
 "built_utc": ts, "deployed_utc": ts, "restart": False, "model_test": False, "ctest": False,
 "text": {"path": "build-text-rocm10/strata", "sha256": text_hash,
          "bytes": os.path.getsize(os.path.join(root, "build-text-rocm10", "strata")),
          "build_dir": "build-text-rocm10-v0220",
          "flags": ["STRATA_HIP_GFX906=ON", "STRATA_ENABLE_HIP=OFF", "CMAKE_HIP_ARCHITECTURES=gfx906",
                    "STRATA_PORTABLE=ON", "STRATA_PREFILL_MMQ=OFF", "STRATA_BUILD_TESTS=ON",
                    "STRATA_BUILD_CONVERSATION_TESTS=ON", "ROCm 10 (/opt/rocm-10.0/core-10.0)", "-j 4"]},
 "vision": {"status": "compile check only, not deployed",
           "reason": "the deployed fork reader is the validated one; the new build has no gfx906 run yet",
           "source": vis_src, "build_dir": "build-vision-rocm10-v0220",
           "compile_check_sha256": vis_hash,
           "path": "build-vision-rocm10/bin/strata-vision",
           "sha256_deployed": "943a54bd7a683446a63904ae37f39f2f40add0f79a10b8a33ef9e60582f02d7f"},
 "backup": {"text": "build-text-rocm10/strata.v0216-a9c9ac4a.bak", "sha256":
            "a9c9ac4a7347060393dc5a08fac33a976a268f73cc06e4797d398b4d389dea3b"},
 "config": "run-iq3-s.json (unchanged by this round, md5 e996f0f9e1e60b2dade05f188ee8c4e3, last owner edit 2026-11-17 19:17 UTC)",
}
with open(os.path.join(logd, "build-identity.json"), "w") as f: json.dump(rec, f, indent=1)
print("wrote " + os.path.join(logd, "build-identity.json"))
PY

say "=== PASS: text engine of v0.2.20 installed, no restart, no test ==="
