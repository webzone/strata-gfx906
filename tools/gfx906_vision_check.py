#!/usr/bin/env python3
"""Compare full real-mmproj image embeddings from CPU and one gfx906 HIP device.

Run only after checking that the selected GPU is free. This starts encoder
subprocesses, never a model server. Input images and raw stdout/stderr are retained.
Requires the repository's pinned NumPy; no weights or output embeddings are committed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import statistics
import struct
import subprocess
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fixture(path, width, height):
    """Deterministic RGB gradient/checker BMP, including non-square patch grids."""
    stride = (width * 3 + 3) & ~3
    data = bytearray()
    for y in range(height):
        for x in range(width):
            data.extend((x * 255 // max(1, width - 1), y * 255 // max(1, height - 1),
                         255 if ((x // 28) ^ (y // 28)) & 1 else 0))
        data.extend(b"\0" * (stride - width * 3))
    hdr = struct.pack("<2sIHHI", b"BM", 54 + len(data), 0, 0, 54)
    hdr += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(data), 2835, 2835, 0, 0)
    path.write_bytes(hdr + data)


def read_sve(path):
    with path.open("rb") as stream:
        hdr = struct.unpack("<5i", stream.read(20))
        values = np.frombuffer(stream.read(), dtype="<f4").astype(np.float64)
    magic, n, nx, ny, width = hdr
    if magic != 0x31455653 or n <= 0 or nx * ny != n or values.size != n * width:
        raise RuntimeError(f"invalid SVE1 dimensions: {path}: {hdr}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite embeddings: {path}")
    return hdr, values.reshape(n, width)


def encode(args, folder, images, mode):
    tag = f"{mode}-{args.max_tokens}-gpu{args.device}"
    env = dict(os.environ)
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"):
        env.pop(key, None)
    env["HIP_VISIBLE_DEVICES"] = str(args.device)
    cmd = [str(args.exe.resolve()), "--mmproj", str(args.mmproj.resolve()), "--model", str(args.model.resolve()),
           "--threads", str(args.threads), "--max-tokens", str(args.max_tokens), "--flash-attn", "off", "--verbose"]
    if mode == "hip":
        cmd.append("--gpu")
    results = {}
    with (folder / f"{tag}.stderr.log").open("w") as err, (folder / f"{tag}.stdout.log").open("w") as out:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                text=True, bufsize=1, env=env)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)

        def line():
            if not selector.select(args.timeout):
                raise TimeoutError(f"encoder timeout: {tag}")
            value = proc.stdout.readline()
            out.write(value)
            out.flush()
            return value.strip()

        try:
            ready = line()
            if not ready.startswith("READY "):
                raise RuntimeError(f"encoder startup: {tag}: {ready}")
            # Device memory after largest-image GPU warm-up, before the actual fixtures.
            smi = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showuse", "--showmeminfo", "vram"],
                                 capture_output=True, text=True, timeout=20)
            (folder / f"{tag}.memory.log").write_text(smi.stdout + smi.stderr)
            for img in images:
                times = []
                for iteration in range(args.repeat):
                    dest = folder / f"{tag}-{img.stem}-{iteration}.sve"
                    proc.stdin.write(f"ENC {img.resolve()} {dest.resolve()}\n")
                    proc.stdin.flush()
                    response = line().split()
                    if len(response) != 5 or response[0] != "OK":
                        raise RuntimeError(f"encode failed: {tag}: {response}")
                    times.append(float(response[4]))
                    read_sve(dest)
                results[img.name] = {"milliseconds": times, "median_ms": statistics.median(times),
                                     "embedding": str(dest)}
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            if proc.wait(timeout=20) != 0:
                raise RuntimeError(f"encoder exit: {tag}: {proc.returncode}")
        finally:
            selector.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdin.close()
            proc.stdout.close()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--mmproj", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--photo", type=Path, required=True, help="real photograph, retained verbatim")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()
    if args.repeat < 1 or args.max_tokens < 1 or args.threads < 1:
        ap.error("repeat, max-tokens and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    images = []
    for name, w, h in (("square", 224, 224), ("wide", 896, 448), ("tall", 448, 896), ("large", 896, 896)):
        path = args.output / f"{name}.bmp"
        fixture(path, w, h)
        images.append(path)
    if args.max_tokens >= 1024:
        path = args.output / "limit.bmp"
        fixture(path, 2048, 2048)
        images.append(path)
    photo = args.output / ("photo" + args.photo.suffix)
    shutil.copy2(args.photo, photo)
    images.append(photo)
    info = {"exe_sha256": sha256(args.exe), "mmproj_sha256": sha256(args.mmproj),
            "model": str(args.model.resolve()), "device": args.device, "max_tokens": args.max_tokens,
            "threads": args.threads, "repeat": args.repeat, "flash_attention": "off",
            "inputs": {p.name: sha256(p) for p in images}, "results": [],
            "limits": "Full embedding comparison to this HIP build's CPU mode (BF16 weights expanded losslessly to FP32), not upstream BF16 activation arithmetic, model logits or generation parity."}
    cpu = encode(args, args.output, images, "cpu")
    hip = encode(args, args.output, images, "hip")
    passed = True
    for img in images:
        h1, a = read_sve(Path(cpu[img.name]["embedding"]))
        h2, b = read_sve(Path(hip[img.name]["embedding"]))
        if h1 != h2:
            raise RuntimeError(f"CPU/HIP grid mismatch: {img.name}: {h1} != {h2}")
        rel_l2 = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
        row_cos = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
        ok = rel_l2 <= 0.005 and float(row_cos.min()) >= 0.999
        passed &= ok
        info["results"].append({"image": img.name, "shape": list(a.shape), "grid": list(h1[2:4]),
                                "relative_l2": rel_l2, "minimum_token_cosine": float(row_cos.min()),
                                "max_absolute_error": float(np.max(np.abs(b - a))), "pass": ok,
                                "cpu": cpu[img.name], "hip": hip[img.name],
                                "encoder_speedup": cpu[img.name]["median_ms"] / max(hip[img.name]["median_ms"], 1)})
    info["pass"] = passed
    info["thresholds"] = {"relative_l2_max": 0.005, "minimum_token_cosine": 0.999}
    text = json.dumps(info, indent=2) + "\n"
    (args.output / "summary.json").write_text(text)
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
