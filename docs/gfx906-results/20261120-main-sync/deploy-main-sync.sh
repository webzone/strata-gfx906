#!/usr/bin/env bash
# t5810 · upstream main 58ee32277 (fork merge b3aaee7e4, no tag after v0.2.16): sync the machine
# checkout only. No build, no stop, no restart, no model request, no ctest. The script verifies its own
# result: source identity, the changed-path scope, the vendored pin, both deployed binary hashes, the
# run config, the running PIDs, and the listener.
# Usage: deploy-main-sync.sh [--dry-run] [--tip]   (--tip syncs to the origin/gfx906 tip instead of MERGE)
set -euo pipefail
ROOT=/home/chris/dev/strata-gfx906
MERGE=b3aaee7e4          # fork merge commit of upstream main 58ee32277
PREV=3c28629c9           # checkout the machine holds before this sync
UPSTREAM=58ee32277       # upstream commit the merge carries
ENGINE=0.1.62            # project(strata VERSION) of this source
CUR_TEXT=$ROOT/build-text-rocm10/strata
CUR_VIS=$ROOT/build-vision-rocm10/bin/strata-vision
EXPECT_TEXT_SHA=a9c9ac4a7347060393dc5a08fac33a976a268f73cc06e4797d398b4d389dea3b
EXPECT_VIS_SHA=943a54bd7a683446a63904ae37f39f2f40add0f79a10b8a33ef9e60582f02d7f
CONFIG=$ROOT/run-iq3-s.json
CONFIG_MD5=e996f0f9e1e60b2dade05f188ee8c4e3   # owner's last edit: 2026-11-17 19:17 UTC
MIN_FREE_GIB=4           # rule 7
RAM_FLOOR_MIB=3000       # rule 2: the live engine must not be pushed out
VENV=$ROOT/.venv/bin/python

DRY=0
TIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift;;
    --tip) TIP=1; shift;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done

h() { sha256sum "$1" | cut -d" " -f1; }
m() { md5sum "$1" | cut -d" " -f1; }
say() { printf '%s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
free_mib() { awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
free_gib() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOGD=$ROOT/logs/main-sync-$TS

say "=== 1. machine state ==="
ss -ltnp 2>/dev/null | grep ':8082' | head -2 || say "8082: not listening"
ps -eo pid,cmd | grep -E 'serve/server\.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' \
  | grep -v grep | tee /tmp/main-sync-procs-$TS.txt
[ "$(grep -c . /tmp/main-sync-procs-$TS.txt)" = 3 ] || fail "expected three Strata processes, found $(grep -c . /tmp/main-sync-procs-$TS.txt)"
ps -o pid,etime -p "$(awk '{print $1}' /tmp/main-sync-procs-$TS.txt | tr '\n' ',' | sed 's/,$//')" | tail -3
say "MemAvailable: $(free_mib) MiB · free disk: $(free_gib) GiB"
rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'used' | head -4
[ "$(free_gib)" -ge "$MIN_FREE_GIB" ] || fail "less than ${MIN_FREE_GIB} GiB free on $ROOT (rule 7)"
[ "$(free_mib)" -ge "$RAM_FLOOR_MIB" ] || fail "MemAvailable under ${RAM_FLOOR_MIB} MiB (rule 2)"
[ "$(h "$CUR_TEXT")" = "$EXPECT_TEXT_SHA" ] || fail "deployed text hash differs from $EXPECT_TEXT_SHA"
[ "$(h "$CUR_VIS")" = "$EXPECT_VIS_SHA" ] || fail "deployed picture reader hash differs from $EXPECT_VIS_SHA"
[ "$(m "$CONFIG")" = "$CONFIG_MD5" ] || fail "run config changed from $CONFIG_MD5"
"$VENV" -c "import json;json.load(open('$CONFIG'))" || fail "run config not parseable"
say "deployed text engine: $("$CUR_TEXT" --version 2>&1 | head -1)"

if [ "$DRY" = 1 ]; then say "DRY RUN: nothing changed."; exit 0; fi

say "=== 2. checkout ==="
git -C "$ROOT" fetch -q origin gfx906 || fail "git fetch failed"
TARGET=$( [ "$TIP" = 1 ] && git -C "$ROOT" rev-parse --short=9 origin/gfx906 || echo "$MERGE" )
HEAD_BEFORE=$(git -C "$ROOT" rev-parse --short=9 HEAD)
if [ "$HEAD_BEFORE" != "$TARGET" ]; then
  git -C "$ROOT" merge --ff-only origin/gfx906 || fail "checkout is not fast-forwardable from $HEAD_BEFORE"
fi
HEAD_NOW=$(git -C "$ROOT" rev-parse --short=9 HEAD)
[ "$HEAD_NOW" = "$TARGET" ] || fail "checkout at $HEAD_NOW, expected $TARGET"
git -C "$ROOT" merge-base --is-ancestor "$UPSTREAM" HEAD || fail "upstream $UPSTREAM is not in $HEAD_NOW"
git -C "$ROOT" merge-base --is-ancestor "$MERGE" HEAD || fail "merge $MERGE is not in $HEAD_NOW"
[ "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)" = "gfx906" ] || fail "branch is not gfx906"
say "checkout: $HEAD_BEFORE -> $HEAD_NOW (upstream $UPSTREAM and merge $MERGE inside)"
mkdir -p "$LOGD"
git -C "$ROOT" log -1 --format='%H %s' > "$LOGD/checkout.txt"
git -C "$ROOT" ls-files .github/workflows > "$LOGD/workflows.txt"
if grep -q . "$LOGD/workflows.txt"; then fail "the tree carries workflow files: $(cat "$LOGD/workflows.txt")"; fi
say "workflow files in the tree: $(wc -l < "$LOGD/workflows.txt")"

say "=== 3. changed-path scope ==="
git -C "$ROOT" diff --stat "$PREV" "$HEAD_NOW" > "$LOGD/sync-scope.txt"
git -C "$ROOT" diff --name-only "$PREV" "$HEAD_NOW" > "$LOGD/sync-paths.txt"
say "$(cat "$LOGD/sync-scope.txt")"
N=$(wc -l < "$LOGD/sync-paths.txt")
say "paths changed since $PREV: $N"
ENG=$(git -C "$ROOT" diff --name-only "$PREV" "$HEAD_NOW" -- engine terminal serve tools/vision tools/hip | wc -l)
[ "$ENG" = 0 ] || fail "$ENG engine or Terminal paths changed since $PREV: a rebuild would be required"
say "engine/Terminal/vision paths changed since $PREV: 0 → no rebuild needed"

CHECK=$( [ -f "$ROOT/tools/check_vendored.py" ] && echo "$ROOT/tools/check_vendored.py" || echo "$ROOT/terminal/tools/check_vendored.py" )
"$VENV" "$CHECK" > "$LOGD/vendored.txt" 2>&1 || fail "vendored pin check failed"
say "vendored pin: $(head -1 "$LOGD/vendored.txt")"

say "=== 4. binaries and startup files unchanged ==="
[ "$(h "$CUR_TEXT")" = "$EXPECT_TEXT_SHA" ] || fail "text hash changed by the sync: $(h "$CUR_TEXT")"
[ "$(h "$CUR_VIS")" = "$EXPECT_VIS_SHA" ] || fail "picture reader hash changed by the sync: $(h "$CUR_VIS")"
say "text engine:  $EXPECT_TEXT_SHA ($(stat -c%s "$CUR_TEXT") B) · reader: $EXPECT_VIS_SHA ($(stat -c%s "$CUR_VIS") B)"
"$VENV" -m py_compile "$ROOT/terminal/serve/server.py" || fail "terminal/serve/server.py does not compile"
say "terminal/serve/server.py compiles with $("$VENV" -V 2>&1)"
[ -L "$ROOT/terminal/.venv" ] || fail "terminal/.venv is not a link to the checkout venv"
say "terminal/.venv -> $(readlink "$ROOT/terminal/.venv")"

say "=== 5. service untouched (read-only) ==="
ps -eo pid,cmd | grep -E 'serve/server\.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' \
  | grep -v grep > /tmp/main-sync-procs-after-$TS.txt
diff /tmp/main-sync-procs-$TS.txt /tmp/main-sync-procs-after-$TS.txt > "$LOGD/pids.txt" \
  || fail "a process changed during the sync: $(cat "$LOGD/pids.txt")"
say "PIDs unchanged: $(awk '{print $1}' /tmp/main-sync-procs-after-$TS.txt | tr '\n' ' ')"
ss -ltnp 2>/dev/null | grep ':8082' | head -1
curl -s --max-time 5 localhost:8082/health | head -c 200 > "$LOGD/health.json"
grep -q '"status": "ok"' "$LOGD/health.json" || fail "health endpoint did not report ok"
say "health: $(cat "$LOGD/health.json")"
curl -s --max-time 5 localhost:8082/metrics | grep -o '"engine": *"[^"]*"' | head -1 > "$LOGD/engine-version.txt" || true
say "running engine version: $(cat "$LOGD/engine-version.txt") (still the pre-restart binary)"
SC=0
bash /home/chris/mi50-t5810/scripts/strata-current.sh > "$LOGD/strata-current.txt" 2>&1 || SC=$?
say "strata-current.sh exit: $SC (1 until the owner restarts 8082)"

say "=== PASS ==="
say "checkout $HEAD_NOW · binaries $EXPECT_TEXT_SHA / $EXPECT_VIS_SHA · no build, no restart, no test"
say "log dir: $LOGD"
