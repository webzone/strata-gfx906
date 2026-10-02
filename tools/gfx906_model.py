#!/usr/bin/env python3
"""Pinned, resumable IQ2_XS / IQ3_S downloads for the MI50 workspace.

Default: print the exact model/storage plan. --fetch downloads the selected shards
(no pack, MTP, packages or services). --share-ple-from can hard-link an identical,
SHA256-verified PLE shard on the same filesystem. Full SHA256 is checked before a
.part is renamed. Existing weights are never overwritten or deleted. Linux/macOS only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
FILES = (
    ("Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf", 39225954592,
     "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7"),
    ("Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf", 28800138432,
     "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
)
MODEL_FILES = {
    "IQ2_XS": FILES,  # Keep FILES and the default behavior compatible with existing callers.
    "IQ3_S": (
        ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", 54817524224,
         "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
        ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf", 28800138432,
         "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    ),
}
GIB = 1 << 30


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(folder: Path, entry, mib_per_sec: float, floor: int, stop: threading.Event,
             model: str = "IQ2_XS"):
    name, size, sha = entry
    final = folder / name
    partial = folder / (name + ".part")
    identity = {"repo": REPO, "revision": REVISION, "bytes": size, "sha256": sha}
    marker = folder / (name + ".download.json")
    if final.exists():
        if final.stat().st_size != size or digest(final) != sha:
            raise RuntimeError(f"existing file differs; refusing to overwrite: {final}")
        print(f"VERIFIED {name}", flush=True)
        return
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise RuntimeError(f"download identity changed; refusing to resume: {marker}")
    if partial.exists() and not marker.exists():
        raise RuntimeError(f"partial without source identity; refusing to resume: {partial}")
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{model}/{name}"
    # Retries re-open at the bytes already on disk; HTTP Range must be honored.
    for attempt in range(5):
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            raise RuntimeError(f"oversized partial: {partial}")
        if have == size:
            break
        if stop.is_set():
            raise RuntimeError("another download failed")
        request = urllib.request.Request(url, headers={"User-Agent": f"strata-gfx906/{model.lower()}"})
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if have and (response.status != 206 or
                             not response.headers.get("Content-Range", "").startswith(f"bytes {have}-")):
                    raise RuntimeError("server ignored resume range; refusing to append")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != size - have:
                    raise RuntimeError("unexpected content length; refusing this response")
                started, received, reported = time.monotonic(), 0, have
                with partial.open("ab") as out:
                    while have < size:
                        if stop.is_set():
                            raise RuntimeError("another download failed")
                        if shutil.disk_usage(folder).free < floor + (1 << 20):
                            raise RuntimeError("disk safety floor reached; partial kept for resume")
                        block = response.read(min(1 << 20, size - have))
                        if not block:
                            raise OSError("short download")
                        out.write(block)
                        have += len(block)
                        received += len(block)
                        delay = received / (mib_per_sec * (1 << 20)) - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
                        if have - reported >= 512 << 20 or have == size:
                            print(f"DOWNLOAD {name}: {have}/{size} ({100 * have / size:.1f}%)", flush=True)
                            reported = have
            break
        except (OSError, TimeoutError) as error:
            if attempt == 4:
                raise
            print(f"RETRY {name}: {type(error).__name__}; partial retained", flush=True)
            time.sleep(2 ** attempt)
    if partial.stat().st_size != size or digest(partial) != sha:
        raise RuntimeError(f"SHA256/size mismatch; partial retained: {partial}")
    if final.exists():
        raise RuntimeError(f"destination appeared; refusing to overwrite: {final}")
    os.rename(partial, final)
    print(f"VERIFIED {name}: {sha}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODEL_FILES, default="IQ2_XS")
    ap.add_argument("--directory", type=Path, help="default: models/<model>")
    ap.add_argument("--share-ple-from", type=Path,
                    help="hard-link a SHA256-verified, identical PLE shard on the same filesystem")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--rate-mib", type=float, default=40, help="aggregate average rate cap (up to two workers)")
    ap.add_argument("--reserve-gib", type=float, default=12, help="free after model for pack/MTP/build/safety")
    args = ap.parse_args()
    if not math.isfinite(args.rate_mib) or not math.isfinite(args.reserve_gib) or args.rate_mib <= 0 or args.reserve_gib < 4:
        ap.error("rate must be positive; reserve must be at least 4 GiB")
    files = MODEL_FILES[args.model]
    folder = (args.directory or Path("models") / args.model).expanduser().resolve()
    parent = folder
    while not parent.exists():
        parent = parent.parent
    remaining = sum(max(0, size - sum(p.stat().st_size for p in (folder / name, folder / (name + ".part"))
                                      if p.exists())) for name, size, _ in files)
    share_source = None
    shared_bytes = 0
    ple_name, ple_size, ple_sha = files[1]
    if args.share_ple_from and not (folder / ple_name).exists():
        share_source = args.share_ple_from.expanduser().resolve(strict=True)
        if not share_source.is_file() or share_source.stat().st_size != ple_size:
            ap.error("shared PLE source has the wrong type/size; nothing overwritten")
        if share_source.stat().st_dev != parent.stat().st_dev:
            ap.error("hard-link sharing requires the same filesystem; no cross-drive copy is made")
        if (folder / (ple_name + ".part")).exists():
            ap.error("PLE partial already exists; retain it and resume without --share-ple-from")
        shared_bytes = ple_size
        remaining -= shared_bytes
    free = shutil.disk_usage(parent).free
    # Re-checking already present shards writes no more model bytes. Do not
    # demand the original pack/MTP planning reserve again after those exist.
    reserve = int(args.reserve_gib * GIB) if remaining else 4 * GIB
    plan = {"repo": REPO, "revision": REVISION, "model": args.model, "directory": str(folder),
            "model_bytes": sum(e[1] for e in files), "remaining_bytes": remaining,
            "free_bytes": free, "reserve_bytes": reserve,
            "fits": free >= remaining + reserve, "shared_ple_bytes": shared_bytes,
            "share_ple_from": str(share_source) if share_source else None,
            "experts_bin": False, "files": [dict(name=n, bytes=s, sha256=h) for n, s, h in files]}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.fetch:
        return
    if not plan["fits"]:
        raise SystemExit("model plus reserve does not fit; nothing downloaded/deleted")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = folder / "source.json"
        if manifest.exists():
            prior = json.loads(manifest.read_text())
            if any(prior.get(k) != plan[k] for k in ("repo", "revision", "model", "files")):
                raise RuntimeError(f"source identity changed; refusing to overwrite: {manifest}")
        if share_source:
            before = share_source.stat()
            if digest(share_source) != ple_sha:
                raise RuntimeError(f"shared PLE SHA256 mismatch; source retained: {share_source}")
            after = share_source.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError("shared PLE source changed while hashing; nothing linked")
            os.link(share_source, folder / ple_name)  # Atomic; fails if a destination appeared.
            print(f"SHARED_PLE {ple_name}: verified SHA256 {ple_sha}", flush=True)
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
        stop = threading.Event()
        workers = max(1, min(2, sum(not (folder / e[0]).exists() for e in files)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download, folder, entry, args.rate_mib / workers,
                                   4 * GIB, stop, args.model) for entry in files]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                stop.set()
                raise
    print(f"MODEL_READY: both {args.model} shards verified", flush=True)


if __name__ == "__main__":
    main()
