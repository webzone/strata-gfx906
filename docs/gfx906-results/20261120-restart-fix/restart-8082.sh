#!/usr/bin/env bash
# t5810 · restart 8082 after upstream's 0.2.16 layout move (4058aa782) deleted the tracked
# data/expert-profile.bin that run-iq3-s.json names. The engine died on
# "read_expert_profile: cannot open .../data/expert-profile.bin", so the server died with it.
# The fork keeps its own untracked copy at that path, byte-identical to upstream's new
# terminal/data/expert-profile.bin, so the owner's config stays byte-unchanged.
# Then start ./run-iq3-s.sh detached, wait until the model is ready, and verify.
# Usage: restart-8082.sh [--dry-run] [--no-start] [--adopt]
#   --adopt  verify a service that is already starting or running; do not start anything
set -euo pipefail
ROOT=/home/chris/dev/strata-gfx906
CONFIG=$ROOT/run-iq3-s.json
CONFIG_MD5=e996f0f9e1e60b2dade05f188ee8c4e3      # owner's last edit: 2026-11-17 19:17 UTC
PROFILE=$ROOT/data/expert-profile.bin
PROFILE_SRC=$ROOT/terminal/data/expert-profile.bin
PROFILE_SHA=8f59b4aa8873209d   # prefix of both copies, 196,632 B
TEXT_BIN=$ROOT/build-text-rocm10/strata
VIS_BIN=$ROOT/build-vision-rocm10/bin/strata-vision
EXPECT_TEXT=a9c9ac4a7347060393dc5a08fac33a976a268f73cc06e4797d398b4d389dea3b
EXPECT_VISION=943a54bd7a683446a63904ae37f39f2f40add0f79a10b8a33ef9e60582f02d7f
EXPECT_ENGINE=0.1.62
# where the two deployed artifacts were compiled; strata-current.sh reads each build dir's CMakeCache
TEXT_BUILT_IN=$ROOT/build-text-rocm10-v0216/strata
VIS_BUILT_IN=$ROOT/build-vision-rocm10-v0215/bin/strata-vision-hip
UPSTREAM=58ee32277
MERGE=b3aaee7e4
LLAMA_PIN=3cf03257f219afbe7334045ff7c6a06ac68c627d
VERIFY=/home/chris/mi50-t5810/scripts/verify-v0216.sh
READY_TIMEOUT_S=240
MIN_FREE_GIB=4
RAM_FLOOR_MIB=3000

DRY=0; NOSTART=0; ADOPT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift;;
    --no-start) NOSTART=1; shift;;
    --adopt) ADOPT=1; shift;;
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
LOGD=$ROOT/logs/restart-$TS
LAUNCH_LOG=$LOGD/launch.log
ENGINE_LOG=$ROOT/logs/cachefix-rocm10-20261007/iq3-engine.log

say "=== 1. state ==="
ss -ltnp 2>/dev/null | grep ':8082' | head -2 || say "8082: not listening (nothing to stop)"
if [ "$ADOPT" = 0 ]; then
  [ -z "$(ss -H -lnt 'sport = :8082')" ] || fail "8082 is occupied; stop the service first (rule 2)"
  N=$(ps -eo cmd | grep -E 'terminal/serve/server\.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' | grep -v grep | wc -l) || true
  [ "$N" = 0 ] || fail "$N Strata processes are still running"
else
  say "adopt: the service is already running or starting, nothing will be started"
fi
say "MemAvailable: $(free_mib) MiB · free disk: $(free_gib) GiB · checkout: $(git -C "$ROOT" rev-parse --short=9 HEAD)"
[ "$(free_gib)" -ge "$MIN_FREE_GIB" ] || fail "less than ${MIN_FREE_GIB} GiB free (rule 7)"
[ "$(free_mib)" -ge "$RAM_FLOOR_MIB" ] || fail "MemAvailable under ${RAM_FLOOR_MIB} MiB (rule 2)"
[ "$(m "$CONFIG")" = "$CONFIG_MD5" ] || fail "run config changed from $CONFIG_MD5"
[ "$(h "$TEXT_BIN")" = "$EXPECT_TEXT" ] || fail "text binary differs from $EXPECT_TEXT"
[ "$(h "$VIS_BIN")" = "$EXPECT_VISION" ] || fail "picture reader differs from $EXPECT_VISION"

say "=== 2. profile path ==="
CFG_PROFILE=$("$ROOT/.venv/bin/python" -c "import json;a=json.load(open('$CONFIG'))['args'];print(a[a.index('--expert-profile')+1])")
say "config names: $CFG_PROFILE"
if [ "$DRY" = 1 ]; then
  say "DRY RUN: profile present: $([ -e "$CFG_PROFILE" ] && echo yes || echo no) · would copy $PROFILE_SRC if absent · nothing changed"
  exit 0
fi
if [ ! -e "$CFG_PROFILE" ]; then
  say "missing (upstream moved the tracked file in 4058aa782). restoring the fork copy from $PROFILE_SRC"
  [ -f "$PROFILE_SRC" ] || fail "$PROFILE_SRC is not in the tree"
  mkdir -p "$(dirname "$CFG_PROFILE")"
  cp "$PROFILE_SRC" "$CFG_PROFILE"
  [ "$(h "$CFG_PROFILE" | head -c 16)" = "$PROFILE_SHA" ] || fail "restored profile hash differs from $PROFILE_SHA"
  [ "$(stat -c%s "$CFG_PROFILE")" = "$(stat -c%s "$PROFILE_SRC")" ] || fail "restored profile size differs"
  say "restored: $CFG_PROFILE ($(stat -c%s "$CFG_PROFILE") B, sha256 $(h "$CFG_PROFILE" | head -c 16)…)"
fi
"$ROOT/.venv/bin/python" - "$CONFIG" <<'PY' || fail "a config path is missing"
import json, os, sys
cfg = json.load(open(sys.argv[1]))
paths = [cfg["exe"], cfg["tokenizer"], cfg["cwd"], cfg["vision"]["exe"], cfg["vision"]["mmproj"], cfg["vision"]["model"]]
paths += [v for v in cfg["args"] if v.startswith("/")]
bad = [p for p in paths if not os.path.exists(p)]
print("config paths:", "all present" if not bad else "MISSING " + " ".join(bad))
sys.exit(1 if bad else 0)
PY
say "config md5 after the fix: $(m "$CONFIG") (unchanged: $CONFIG_MD5)"

if [ "$DRY" = 1 ]; then say "DRY RUN: nothing changed."; exit 0; fi
mkdir -p "$LOGD"
cp "$ENGINE_LOG" "$LOGD/engine-log-before.log" 2>/dev/null || true
if [ "$NOSTART" = 1 ]; then say "NOSTART: the profile is restored; the service was not started."; exit 0; fi

if [ "$NOSTART" = 1 ] || [ "$ADOPT" = 1 ]; then
  say "nothing started; waiting for the service that is already running"
else
  say "=== 3. start 8082 ==="
  LAUNCH_LOG=$LOGD/launch.log
  setsid sh -c "cd $ROOT && ./run-iq3-s.sh > $LAUNCH_LOG 2>&1" < /dev/null &
  sleep 3
  pgrep -f 'terminal/serve/server\.py' >/dev/null || { tail -30 "$LAUNCH_LOG"; fail "the server process exited at once"; }
  say "server pid: $(pgrep -f 'terminal/serve/server\.py' | head -1) · the expert load takes 1-3 minutes"
fi

say "=== 4. wait for the model ==="
READY=0
for i in $(seq 1 "$READY_TIMEOUT_S"); do
  if curl -s --max-time 3 localhost:8082/health 2>/dev/null | grep -q '"loaded": true'; then READY=1; break; fi
  pgrep -f 'terminal/serve/server\.py' >/dev/null || { [ -f "$LAUNCH_LOG" ] && tail -40 "$LAUNCH_LOG"; fail "the server died after $i s"; }
  sleep 1
done
[ "$READY" = 1 ] || { [ -f "$LAUNCH_LOG" ] && tail -40 "$LAUNCH_LOG"; fail "model not ready within ${READY_TIMEOUT_S} s"; }
say "ready after ${i} s"
say "listening: $(ss -ltnp 2>/dev/null | grep ':8082' | head -1)"
say "health: $(curl -s --max-time 5 localhost:8082/health | head -c 200)"

say "=== 5. build identity ==="
# strata-current.sh looks the running hashes up in logs/*/build-identity.json and reads the flags from
# each artifact's own build dir. The v0.2.15 and v0.2.16 records used other key names, so the running
# pair had no record in the shape the checker reads. This round writes one.
if [ ! -f "$LOGD/build-identity.json" ]; then
  cat > "$LOGD/build-identity.json" <<JSON
{
 "upstream": "$UPSTREAM",
 "upstream_note": "upstream main after tag v0.2.16 / 64524fb9f; no new tag was published",
 "merge": "$MERGE",
 "engine_change": "none in 64524fb9f..$UPSTREAM: docs, supporters and the docker build context only",
 "llama_cpp": "$LLAMA_PIN",
 "engine_version": "$EXPECT_ENGINE",
 "min_engine": "(0, 1, 46)",
 "text": {
  "path": "$TEXT_BUILT_IN",
  "sha256": "$EXPECT_TEXT",
  "bytes": $(stat -c%s "$TEXT_BIN"),
  "built_utc": "20261120T025858Z",
  "deployed_as": "build-text-rocm10/strata"
 },
 "vision": {
  "path": "$VIS_BUILT_IN",
  "sha256": "$EXPECT_VISION",
  "bytes": $(stat -c%s "$VIS_BIN"),
  "built_utc": "20261117T111553Z",
  "deployed_as": "build-vision-rocm10/bin/strata-vision",
  "note": "the v0.2.16 round compiled a picture-reader check (51f830a4...) and left the deployed v0.2.15 artifact in place"
 },
 "round": "restart of 8082 after upstream's 0.2.16 layout move: data/expert-profile.bin restored from terminal/data/expert-profile.bin (both sha256 8f59b4aa8873209d..., 196,632 B)",
 "restart": true,
 "model_test": true,
 "ctest": false
}
JSON
fi
say "identity record: $LOGD/build-identity.json"

say "=== 6. verify ==="
ps -eo pid,etime,cmd | grep -E 'terminal/serve/server\.py|build-text-rocm10/strata|build-vision-rocm10/bin/strata-vision' \
  | grep -v grep | tee "$LOGD/pids.txt"
say "running engine: $(curl -s --max-time 5 localhost:8082/metrics | grep -o '"engine": *"[^"]*"' | head -1)"
curl -s --max-time 5 localhost:8082/metrics | grep -o '"engine": *"[^"]*"' | head -1 | grep -q "$EXPECT_ENGINE" \
  || fail "running engine is not $EXPECT_ENGINE"
"$ROOT/.venv/bin/python" - "$ROOT" "$CONFIG" <<'PY' || fail "the engine binary the config names is not the one the process runs"
import json, os, subprocess, sys
root, cfg_path = sys.argv[1], sys.argv[2]
exe = json.load(open(cfg_path))["exe"]
out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
lines = [l for l in out.splitlines() if root in l and "strata --serve" in l]
print("engine process:", lines[0].split()[0] if lines else "none")
sys.exit(0 if lines else 1)
PY
V=0
"$VERIFY" > "$LOGD/verify.txt" 2>&1 || V=$?
say "verify-v0216.sh exit: $V"; tail -20 "$LOGD/verify.txt"
[ "$V" = 0 ] || fail "verify-v0216.sh reported a problem"
cp "$ENGINE_LOG" "$LOGD/engine-log-after.log" 2>/dev/null || true
SC=0
bash /home/chris/mi50-t5810/scripts/strata-current.sh > "$LOGD/strata-current.txt" 2>&1 || SC=$?
say "strata-current.sh exit: $SC"

say "=== PASS ==="
say "8082 runs engine $EXPECT_ENGINE · profile $CFG_PROFILE · config md5 $CONFIG_MD5 · log dir $LOGD"
