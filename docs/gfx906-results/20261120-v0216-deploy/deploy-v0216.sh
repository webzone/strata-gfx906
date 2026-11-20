#!/usr/bin/env bash
# t5810 · v0.2.16 (upstream tag v0.2.16 / 64524fb9f, fork merge passed with --merge): sync the machine
# checkout, compile the gfx906 text engine and the picture reader, install the text engine only.
# No stop, no restart, no model test, no ctest. The script verifies its own result: source identity,
# the vendored pin, every hash before and after, the engine version string, and the running PIDs.
# Usage: deploy-v0216.sh --merge <fork-commit> [--dry-run] [--no-build] [--with-vision]
set -euo pipefail
ROOT=/home/chris/dev/strata-gfx906
UPSTREAM=64524fb9f       # upstream tag v0.2.16
ENGINE=0.1.62            # project(strata VERSION) of this source
ROC=/opt/rocm-10.0/core-10.0
CONFIG=$ROOT/run-iq3-s.json
CUR_TEXT=$ROOT/build-text-rocm10/strata
CUR_VIS=$ROOT/build-vision-rocm10/bin/strata-vision
TEXT_DIR=$ROOT/build-text-rocm10-v0216
VIS_DIR=$ROOT/build-vision-rocm10-v0216
SCRIPT=/home/chris/mi50-t5810/scripts/strata-current.sh
EXPECT_CUR_TEXT=bc8cee524a538e87aa79397bd992d1cf   # md5 of the deployed 0.1.61 engine (v0.2.15)
EXPECT_CUR_VIS=eafb5d76f03635a2f8424a6ef5cda41f    # md5 of the deployed picture reader
BAK=$ROOT/build-text-rocm10/strata.v0215-b424d55b.bak
BAK_SHA=b424d55bea4a8b791c4d2594e4d31907bf1893fe29773542fc4c4877e9187568
CONFIG_MD5=e996f0f9e1e60b2dade05f188ee8c4e3   # run-iq3-s.json as the owner last edited it (2026-11-17 19:17 UTC)
RAM_FLOOR_MIB=1024       # abort the build if MemAvailable drops under this
MIN_FREE_GIB=4           # rule 7: never build with less free disk than this

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
[ -n "$MERGE" ] || { echo "usage: deploy-v0216.sh --merge <fork-commit>"; exit 2; }

h() { sha256sum "$1" | cut -d" " -f1; }
m() { md5sum "$1" | cut -d" " -f1; }
say() { printf '%s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
free_mib() { awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
free_gib() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOGD=$ROOT/logs/v0216-build-$TS

say "=== 1. machine state ==="
ss -ltnp 2>/dev/null | grep ':8082' | head -2 || say "8082: not listening"
ps -eo pid,etime,cmd | grep -E 'serve/server.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' \
  | grep -v grep | tee /tmp/v0216-procs-$TS.txt || say "no Strata processes"
say "MemAvailable: $(free_mib) MiB · free disk: $(free_gib) GiB"
rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'used' | head -4
[ "$(free_gib)" -ge "$MIN_FREE_GIB" ] || fail "less than ${MIN_FREE_GIB} GiB free on $ROOT (rule 7)"
[ "$(free_mib)" -ge 3000 ] || fail "MemAvailable under 3000 MiB: the build would push the live engine out (rule 2)"
[ "$(m "$CUR_TEXT")" = "$EXPECT_CUR_TEXT" ] || fail "deployed text hash differs from $EXPECT_CUR_TEXT"
[ "$(m "$CUR_VIS")" = "$EXPECT_CUR_VIS" ] || fail "deployed picture reader hash differs from $EXPECT_CUR_VIS"
[ "$(m "$CONFIG")" = "$CONFIG_MD5" ] || fail "run config changed from $CONFIG_MD5"
python3 -c "import json;json.load(open('$CONFIG'))" || fail "run config not parseable"
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
mkdir -p "$LOGD"
git -C "$ROOT" log -1 --format='%H %s' > "$LOGD/checkout.txt"
git -C "$ROOT" ls-files .github/workflows > "$LOGD/workflows.txt"
if grep -q . "$LOGD/workflows.txt"; then fail "the merged tree carries workflow files: $(cat "$LOGD/workflows.txt")"; fi
say "workflow files in the tree: $(wc -l < "$LOGD/workflows.txt")"
CHECK=$( [ -f "$ROOT/tools/check_vendored.py" ] && echo "$ROOT/tools/check_vendored.py" || echo "$ROOT/terminal/tools/check_vendored.py" )
"$ROOT/.venv/bin/python" "$CHECK" > "$LOGD/vendored.txt" 2>&1 \
  || fail "$CHECK failed (see $LOGD/vendored.txt)"
grep -qiE 'ok|3cf03257f' "$LOGD/vendored.txt" || fail "$CHECK reported nothing usable"
say "vendored ggml pin: ok ($CHECK: $(head -1 "$LOGD/vendored.txt"))"

# Upstream moved the serving layer to terminal/. The launcher now runs terminal/serve/server.py from
# terminal/.venv. Keep one venv per checkout: link it, never install a second one (rule 2).
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

# Upstream moved the engine sources to engine/ in v0.2.16. Pick whichever root the checkout has.
TEXT_SRC=$ROOT
[ -f "$ROOT/CMakeLists.txt" ] || TEXT_SRC=$ROOT/engine
[ -f "$TEXT_SRC/CMakeLists.txt" ] || fail "no CMakeLists.txt in $ROOT or $ROOT/engine"
say "text engine source: $TEXT_SRC"

# The picture reader lives in tools/vision or, after upstream's move, in terminal/tools/vision.
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
  grep -iE 'gfx906|HIP' "$LOGD/configure-$(basename "$dir").log" | head -4
  ( cd "$ROOT" && cmake --build "$(basename "$dir")" -j 2 ) > "$LOGD/build-$(basename "$dir").log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    [ "$(free_mib)" -ge "$RAM_FLOOR_MIB" ] || { pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; fail "MemAvailable under $RAM_FLOOR_MIB MiB during $dir: build stopped, nothing installed"; }
    sleep 10
  done
  wait "$pid" || fail "build of $dir failed (see $LOGD/build-$(basename "$dir").log)"
  say "build log: $(grep -c '' "$LOGD/build-$(basename "$dir").log") lines · errors: $(grep -c 'error:' "$LOGD/build-$(basename "$dir").log") · warnings: $(grep -c 'warning:' "$LOGD/build-$(basename "$dir").log")"
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
  cp -a "$CUR_VIS" "$ROOT/build-vision-rocm10/bin/strata-vision.v0215-943a54bd.bak"
  install -m 0755 "$VIS_DIR/bin/strata-vision-hip" "$CUR_VIS"
  [ "$(h "$CUR_VIS")" = "$SRC_VIS_HASH" ] || fail "installed picture reader hash != the built hash"
  say "installed picture reader: $CUR_VIS = $SRC_VIS_HASH"
else
  say "picture reader left untouched: $CUR_VIS stays $(m "$CUR_VIS") (upstream's picture tower changed in this range and is not validated on gfx906)"
fi

say "=== 5. the running trio is unchanged ==="
ps -eo pid,etime,cmd | grep -E 'serve/server.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' \
  | grep -v grep > /tmp/v0216-procs-after-$TS.txt
if diff -q <(awk '{print $1}' /tmp/v0216-procs-$TS.txt) <(awk '{print $1}' /tmp/v0216-procs-after-$TS.txt) >/dev/null; then
  say "same PIDs before and after: $(awk '{print $1}' /tmp/v0216-procs-$TS.txt | tr '\n' ' ')"
else fail "process list changed: nothing should have been started or stopped"; fi
say "health: $(curl -s -m 10 http://127.0.0.1:8082/health | head -c 160)"
say "metrics engine version still: $(curl -s -m 10 http://127.0.0.1:8082/metrics | python3 -c 'import json,sys;print(json.load(sys.stdin)["engine"]["version"])')"
say "the running engine keeps its old inode: $(ls -l /proc/$(pgrep -f 'build-text-rocm10/strata --serve' | head -1)/exe | sed 's/.*exe -> //')"
say "run config md5 still: $(m "$CONFIG")"

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
python3 - "$LOGD" "$TS" "$MERGE" "$SRC_TEXT_HASH" "$SRC_VIS_HASH" "$VIS_SRC" <<'PY'
import json, os, sys
logd, ts, merge, text_hash, vis_hash, vis_src = sys.argv[1:7]
root = os.path.dirname(logd)
rec = {
 "upstream": "64524fb9f", "upstream_tag": "v0.2.16", "merge": merge,
 "engine_version": "0.1.62", "min_engine": "(0, 1, 46)",
 "llama_pin": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
 "built_utc": ts, "deployed_utc": ts, "restart": False, "model_test": False, "ctest": False,
 "text": {"build_dir": "build-text-rocm10-v0216", "sha256": text_hash,
          "bytes": os.path.getsize(os.path.join(root, "build-text-rocm10-v0216", "strata")),
          "flags": ["STRATA_HIP_GFX906=ON", "STRATA_ENABLE_HIP=OFF", "CMAKE_HIP_ARCHITECTURES=gfx906",
                    "STRATA_PORTABLE=ON", "STRATA_PREFILL_MMQ=OFF", "STRATA_BUILD_TESTS=ON",
                    "STRATA_BUILD_CONVERSATION_TESTS=ON", "ROCm 10 (/opt/rocm-10.0/core-10.0)", "-j 2"],
          "deployed_as": "build-text-rocm10/strata"},
 "vision": {"status": "compile check only, not deployed",
           "reason": "upstream's picture tower changed in c77572806..64524fb9f and is not validated on gfx906",
           "source": vis_src, "build_dir": "build-vision-rocm10-v0216",
           "compile_check_sha256": vis_hash,
           "deployed_sha256": "943a54bd7a683446a63904ae37f39f2f40add0f79a10b8a33ef9e60582f02d7f",
           "deployed_as": "build-vision-rocm10/bin/strata-vision"},
 "backup": {"text": "build-text-rocm10/strata.v0215-b424d55b.bak", "sha256": "b424d55bea4a8b791c4d2594e4d31907bf1893fe29773542fc4c4877e9187568"},
 "config": "run-iq3-s.json (unchanged by this round, md5 e996f0f9e1e60b2dade05f188ee8c4e3, last owner edit 2026-11-17 19:17 UTC)",
}
with open(os.path.join(logd, "build-identity.json"), "w") as f: json.dump(rec, f, indent=1)
print("wrote " + os.path.join(logd, "build-identity.json"))
PY

say "=== PASS: text engine of v0.2.16 installed, no restart, no test ==="
