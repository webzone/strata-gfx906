#!/usr/bin/env python3
"""Real IQ2_XS pipe-only smoke: no HTTP listener, downloads or service changes.

Requires the prepared GGUF / native pack / MTP runtime. Run under a process-group
`timeout` and only after checking that the selected GPUs are idle. JSON records
real generated IDs, text and the engine's timings/acceptance, not synthetic tok/s.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, engine_args
from serve.frontend import ChatTemplate
from tools.strata_tokenizer import Tokenizer

CASES = (
    ("arithmetic", "只回答一个数字：7+5等于几？", 16),
    ("literal", "Reply with exactly: MI50 ready.", 16),
    ("code", "Write a Python function named square(x) that returns x*x. Output only Python code.", 64),
    ("sequence", "从1写到50，每个数字之间用逗号分隔，不要解释。", 128),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--split", default="24", help="first layer on GPU1, or auto; ignored on one GPU")
    ap.add_argument("--min-p", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context", type=int, default=4096)
    a = ap.parse_args()
    if not 0 <= a.min_p <= 1 or a.context < 512:
        ap.error("min-p must be 0..1; context must be >=512")
    ids = [int(x) for x in a.gpus.split(",")]
    if not ids or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
        ap.error("choose distinct nonnegative GPU ordinals")
    shards = [ROOT / "models/IQ2_XS" / f"Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{i}-of-00002.gguf" for i in (1, 2)]
    pack, mtp = ROOT / "packs/iq2_xs", ROOT / "mtp/rt"
    for p in [*shards, pack / "native_experts.txt", mtp / "experts.bin", mtp / "dense.txt"]:
        if not p.is_file():
            raise SystemExit(f"prepare the real model first: missing {p}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite acceptance evidence: {a.out}")
    cfg = {"exe": str(ROOT / "build-hip/strata"), "backend": "hip", "gpu": ids,
           "layer_split": a.split, "args": [
               "--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(shards[1]),
               "--expert-profile", str(ROOT / "data/expert-profile.bin"), "--expert-cache", "auto",
               "--max-context", str(a.context), "--kv", "int8", "--pool-workers", "6",
               "--prefill", "128", "--spec", "4", "--spec-min-p", str(a.min_p),
               "--mtp", str(mtp), "--suffix-draft", "0", "--prompt-cache", "0",
               "--adapt-every", "0", "--pcie-frac", "0", "--vram-reserve-mib", "1024"]}
    tk = Tokenizer.from_gguf(shards[0])
    tpl = ChatTemplate(pack / "tokenizer/chat_template.jinja")
    report = {"status": "starting", "model": "GSQ-RCO IQ2_XS", "config": cfg,
              "limitations": ["short prompts, 4K default context; not a CPU-logit oracle or long-soak test",
                              "min-p=1 thresholds drafts; MTP remains loaded and may still compute"], "cases": []}
    def save():
        tmp = a.out.with_suffix(a.out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(a.out)
    save()
    engine = None
    try:
        started = time.monotonic()
        engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=str(ROOT),
                              log=str(a.out.with_suffix(".engine.log")), env=child_env(cfg))
        report.update(status="ready", load_seconds=time.monotonic()-started, info=engine.info)
        save()
        for name, prompt, max_new in CASES:
            text = tpl.render([{"role": "user", "content": prompt}], enable_thinking=False)
            prompt_ids = tk.encode(text, parse_special=True)
            started = time.monotonic()
            generated = [t for t in engine.generate(prompt_ids, max_new, {"temperature": 0}, threading.Event())
                         if t is not None]
            if not generated or any(t < 0 or t >= len(tk.tokens) for t in generated):
                raise RuntimeError("empty or invalid generated token stream")
            result = {"name": name, "prompt": prompt, "prompt_ids": prompt_ids, "max_new": max_new,
                      "ids": generated, "text": tk.decode(generated), "seconds": time.monotonic()-started,
                      "engine": dict(engine.last)}
            ms = result["engine"].get("decode_ms", 0)
            if ms > 0:
                # DONE.decode_ms covers the engine's generated-token windows,
                # including the first output; match its own generated/ms metric.
                result["decode_tok_s"] = len(generated)*1000/ms
            report["cases"].append(result)
            save()
            print(json.dumps(result, ensure_ascii=False), flush=True)
        report["status"] = "passed_transport_and_generation"
    except BaseException as e:
        report.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        if engine is not None:
            engine.unload()
            report["exit_code"] = engine.exit_code()
            if hasattr(engine.log, "close"):
                engine.log.close()
        save()
    if report.get("exit_code") != 0:
        raise SystemExit("engine did not exit cleanly")


if __name__ == "__main__":
    main()
