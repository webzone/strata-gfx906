#!/usr/bin/env bash
# tools/deploy-ink906.sh — T5810: build strata-inkling with its HIP plug-in for gfx906 (ROCm 7.10 at /opt/rocm-10.0/core-10.0)
# and install it into the owner's `strata` install as a local, unsigned engine package of the inkling2 shape, pinned, so
# `strata launch` (exe "component:strata-inkling") starts it. Starts nothing, stops nothing, loads no model.
# The package layout is the one `strata components add-build` writes for the Qwen engine (terminal/strata_control/local_engine.py),
# placed in the shape's own folder (terminal/strata_control/shape_engine.py): components/shapes/inkling2/engine/<pkg>/.
# Usage: deploy-ink906.sh [--dry-run] [--commit <sha>]   (default: origin/gfx906)
set -euo pipefail
ROOT=/home/chris/dev/strata-gfx906
WT=/home/chris/dev/strata-gfx906-ink906
BUILD=$WT/build-ink906
ROC=/opt/rocm-10.0/core-10.0          # ROCm 7.10 (its version file says 10.0.0)
TERM_DIR=$HOME/.local/share/strata/terminal/current
TPY=$TERM_DIR/python/bin/python3.12
COMP=${XDG_DATA_HOME:-$HOME/.local/share}/strata/components
SHAPE=$COMP/shapes/inkling2
PKG_NAME=0.1.70-linux-x86_64-hip-local
PKG=$SHAPE/engine/$PKG_NAME
ENGINE=0.1.70
SE_REL=strata_control/shape_engine.py
SE_ORIG=30f80455ead391c349242d25ced50733d935e3b3befcb6b5dd8eae6e089659fd   # upstream Terminal 0.1.70 shape_engine.py
MIN_FREE_GIB=4
DRY=0; COMMIT=""
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1; shift;; --commit) COMMIT=$2; shift 2;; *) echo "unknown argument: $1"; exit 2;; esac; done
say() { printf '%s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
TS=$(date -u +%Y%m%dT%H%M%SZ); LOGD=$ROOT/logs/ink906-deploy-$TS; mkdir -p "$LOGD"

say "=== 1. machine state"
if pgrep -af 'strata|server\.py' | grep -v -E 'pgrep|deploy-ink906' > "$LOGD/procs.txt"; then fail "a Strata process runs: $(cat "$LOGD/procs.txt")"; fi
say "no Strata process"
FREE=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'); [ "$FREE" -ge "$MIN_FREE_GIB" ] || fail "under $MIN_FREE_GIB GiB free"
say "free disk $FREE GiB · MemAvailable $(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo) MiB · log $LOGD"
[ -x "$ROC/lib/llvm/bin/clang++" ] || fail "no ROCm 7.10 compiler at $ROC"
[ -x "$TPY" ] || fail "no installed strata Terminal at $TERM_DIR"
say "strata Terminal: $(cat "$TERM_DIR/VERSION")"

say "=== 2. source"
git -C "$ROOT" fetch -q origin gfx906
COMMIT=$(git -C "$ROOT" rev-parse "${COMMIT:-origin/gfx906}")
[ -d "$WT" ] || git -C "$ROOT" worktree add -q --detach "$WT" "$COMMIT"
git -C "$WT" checkout -q --detach "$COMMIT"
[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || fail "build worktree has tracked changes"
say "source $COMMIT ($(git -C "$WT" log -1 --format=%s | cut -c1-80))"
[ "$DRY" = 1 ] && { say "DRY RUN: nothing built or installed."; exit 0; }

say "=== 3. build strata-inkling (HIP plug-in, gfx906)"
cmake -S "$WT/engine" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_INKLING=ON -DSTRATA_BUILD_HY4=OFF \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_EXPERIMENTAL_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_HIP_COMPILER=$ROC/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROC \
  -DCMAKE_PREFIX_PATH=$ROC -DSTRATA_PORTABLE=ON > "$LOGD/configure.log" 2>&1 || fail "configure failed ($LOGD/configure.log)"
grep -q 'HIP enabled, arch gfx906' "$LOGD/configure.log" || fail "configure did not select HIP gfx906"
nice -n 10 cmake --build "$BUILD" --target strata-inkling -j 6 > "$LOGD/build.log" 2>&1 || fail "build failed ($LOGD/build.log)"
EXE=$BUILD/src/inkling2/strata-inkling
[ -x "$EXE" ] || fail "no $EXE"
VER=$("$EXE" --version 2>&1 | head -1); case "$VER" in *"$ENGINE"*) ;; *) fail "engine reports '$VER', expected $ENGINE";; esac
ldd "$EXE" > "$LOGD/ldd.txt"
grep -q "libamdhip64.so.7 => $ROC/lib/" "$LOGD/ldd.txt" || fail "strata-inkling does not load libamdhip64 from $ROC"
! grep -q 'not found' "$LOGD/ldd.txt" || fail "missing libraries: $(grep 'not found' "$LOGD/ldd.txt")"
SHA=$(sha256sum "$EXE" | cut -d' ' -f1)
say "built: $VER · sha256 $SHA · $(stat -c%s "$EXE") B · errors $(grep -c 'error:' "$LOGD/build.log")"

say "=== 4. package (local, unsigned) and pin"
"$TPY" - "$EXE" "$PKG" "$SHAPE" "$PKG_NAME" "$COMMIT" "$ENGINE" <<'PY'
import hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
exe, pkg, shape, name, commit, engine = sys.argv[1:7]
exe, pkg, shape = Path(exe), Path(pkg), Path(shape)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
want = sha(exe)
if (pkg / "strata-inkling").is_file() and sha(pkg / "strata-inkling") == want:
    print(f"package {name} already holds this build; kept")
else:
    if pkg.exists():
        bak = pkg.with_name(pkg.name + ".bak-" + __import__("time").strftime("%Y%m%dT%H%M%SZ", __import__("time").gmtime()))
        os.replace(pkg, bak); print(f"previous package moved to {bak.name}")
    pkg.parent.mkdir(parents=True, exist_ok=True)
    st = Path(tempfile.mkdtemp(prefix=".add-", dir=pkg.parent))
    shutil.copy2(exe, st / "strata-inkling")
    build = {"version": engine, "backend": "hip", "archs": ["gfx906"], "source": f"strata-gfx906 {commit}", "shape": "inkling2",
             "entrypoint": "strata-inkling", "toolchain": "ROCm 7.10 (/opt/rocm-10.0/core-10.0)",
             "flags": ["STRATA_ENABLE_HIP=ON", "STRATA_EXPERIMENTAL_GFX906=ON", "CMAKE_HIP_ARCHITECTURES=gfx906", "STRATA_BUILD_INKLING=ON", "STRATA_PORTABLE=ON"]}
    (st / "BUILD.json").write_text(json.dumps(build, indent=1) + "\n")
    files = [{"path": n, "sha256": sha(st / n), "bytes": (st / n).stat().st_size} for n in ("strata-inkling", "BUILD.json")]
    man = {"schema_version": 2, "id": "strata-inkling", "name": "strata-inkling", "shape": "inkling2", "version": engine, "platform": "linux-x86_64",
           "backend": "hip", "archs": ["gfx906"], "min_driver": {}, "build_id": f"local:strata-gfx906 {commit[:12]} rocm-7.10 gfx906",
           "min_runtime": None, "entrypoint": "strata-inkling", "files": files, "signature": None, "local": True,
           "notes": "strata-gfx906 fork build for MI50/MI60 (gfx906, wave64), built on this computer; not signed by Strata"}
    (st / "engine-component.json").write_text(json.dumps(man, indent=1) + "\n")
    os.replace(st, pkg); print(f"installed package {name}")
state = shape / ".updates" / "state.json"
state.parent.mkdir(parents=True, exist_ok=True)
try: data = json.loads(state.read_text())
except (OSError, ValueError): data = {"schema_version": 1}
data.setdefault("engine", {})["pinned"] = name
tmp = state.with_suffix(".tmp"); tmp.write_text(json.dumps(data, indent=1) + "\n"); os.replace(tmp, state)
print(f"pinned {name} in {state}")
PY

say "=== 4b. Terminal fix: a local shape engine runs when the feed has none for this PC"
SE_DST=$TERM_DIR/$SE_REL; SE_SRC=$WT/terminal/$SE_REL
SE_NEW=$(sha256sum "$SE_SRC" | cut -d' ' -f1); SE_HAVE=$(sha256sum "$SE_DST" | cut -d' ' -f1)
if [ "$SE_HAVE" = "$SE_NEW" ]; then say "installed $SE_REL already has the fix"
elif [ "$SE_HAVE" = "$SE_ORIG" ]; then
  cp -p "$SE_DST" "$SE_DST.orig-$TS"; cp "$SE_SRC" "$SE_DST.tmp-$TS"; mv "$SE_DST.tmp-$TS" "$SE_DST"
  [ "$(sha256sum "$SE_DST" | cut -d' ' -f1)" = "$SE_NEW" ] || fail "patched $SE_REL hash differs"
  say "patched $SE_REL (backup $SE_REL.orig-$TS)"
else fail "installed $SE_REL is neither upstream 0.1.70 nor the fork's file (sha256 $SE_HAVE): not patched"; fi

say "=== 5. verify with the installed strata Terminal (no start)"
cd "$TERM_DIR"
"$TPY" - "$PKG" "$SHA" <<'PY' | tee "$LOGD/resolve.txt"
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from strata_control import engine_component as ec
pkg, sha = Path(sys.argv[1]), sys.argv[2]
cfg = json.loads(Path.home().joinpath(".local/state/strata/launch/gsq-rco-iq3_s-256k.json").read_text())
print("detected backend:", ec.detect_backend(), "· hip archs here:", ec.hip_archs_present())
exe = ec.expand("component:strata-inkling", cfg, full=True)            # the start path: every file hashed
print("component:strata-inkling ->", exe)
assert Path(exe).resolve() == (pkg / "strata-inkling").resolve(), "the start path does not resolve to the gfx906 package"
import hashlib; assert hashlib.sha256(Path(exe).read_bytes()).hexdigest() == sha, "resolved engine hash differs"
print("resolve OK: the saved launch starts the gfx906 build")
import argparse
from strata_cli import update_cmd
from strata_control import shape_engine
said = []
got = shape_engine.ensure("inkling2", update_cmd._feed_source(argparse.Namespace(feed=None), ec.components_dir()), emit=said.append, title="Inkling-2-Mini", automatic=True)
print("launch engine step (reads the feed) ->", got, "· said:", said)
assert got and Path(got).resolve() == (pkg / "strata-inkling").resolve(), "strata launch's engine step does not return the gfx906 build"
print("launch engine step OK")
PY
say "doctor: $(~/.local/bin/strata doctor 2>&1 | grep -iE 'inkling|shape|local' | head -4 | tr '\n' ' ')"
if pgrep -af 'strata|server\.py' | grep -v -E 'pgrep|deploy-ink906' >/dev/null; then fail "a Strata process appeared"; fi
say "=== PASS: strata-inkling $ENGINE (gfx906) installed and pinned; nothing started. Start: strata launch"
