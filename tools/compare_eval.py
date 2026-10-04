#!/usr/bin/env python3
"""A/B Benchmark comparison: Fork gfx906 vs Upstream gfx906 on 2x MI50 (T5810)."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path("/home/chris/dev/strata-gfx906")
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, engine_args
from serve.frontend import ChatTemplate
from tools.strata_tokenizer import Tokenizer

BINARIES = {
    "fork": Path("/data/strata-gfx906/eval-v0139/build-fork/strata"),
    "upstream": Path("/data/strata-gfx906/eval-v0139/build-upstream/strata")
}

PROMPTS = [
    {
        "name": "arithmetic",
        "prompt": "只回答一个数字：7+5等于几？",
        "max_new": 16
    },
    {
        "name": "code_max",
        "prompt": "写一个 Python 函数返回列表中的最大值，不要写解释，只输出代码。",
        "max_new": 64
    },
    {
        "name": "logic",
        "prompt": "所有的猫都是哺乳动物。所有的哺乳动物都有脊椎。波波是一只猫。波波有脊椎吗？只回答是或否。",
        "max_new": 16
    },
    {
        "name": "quantum_physics",
        "prompt": "请详细解释量子力学中的双缝干涉实验，包括其实验现象、波粒二象性解释以及观察者效应的本质。",
        "max_new": 128
    }
]

def get_vram():
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"], text=True)
        used = re.findall(r"VRAM Total Used Memory \(B\):\s*(\d+)", out)
        if used:
            return [round(int(x) / (1024**3), 3) for x in used]
    except Exception:
        pass
    return []

def run_suite(label, bin_path, out_dir):
    shard1 = ROOT / "models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
    shard2 = ROOT / "models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"
    pack = ROOT / "packs/iq2_xs"
    mtp = ROOT / "mtp/rt"
    tk = Tokenizer.from_gguf(shard1)
    tpl = ChatTemplate(pack / "tokenizer/chat_template.jinja")

    cfg = {
        "exe": str(bin_path),
        "backend": "hip",
        "gpu": [0, 1],
        "layer_split": "24",
        "args": [
            "--pack", str(pack),
            "--native", str(shard1),
            "--ple-gguf", str(shard2),
            "--expert-profile", str(ROOT / "data/expert-profile.bin"),
            "--expert-cache", "auto",
            "--max-context", "4096",
            "--kv", "int8",
            "--pool-workers", "6",
            "--prefill", "2048",
            "--spec", "4",
            "--spec-min-p", "0.5",
            "--mtp", str(mtp),
            "--suffix-draft", "0",
            "--prompt-cache", "0",
            "--adapt-every", "0",
            "--pcie-frac", "0",
            "--vram-reserve-mib", "1024"
        ]
    }

    env = child_env(cfg)
    env["GPU_MAX_HW_QUEUES"] = "8"
    env["HSA_ENABLE_SDMA"] = "0"
    env["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)

    log_path = out_dir / f"{label}_engine.log"
    print(f"[{label.upper()}] Launching StrataEngine on 2x MI50...", flush=True)
    t_start = time.perf_counter()
    vram_init = get_vram()

    engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=str(ROOT),
                          log=str(log_path), env=env)
    load_time = round(time.perf_counter() - t_start, 2)
    vram_loaded = get_vram()
    print(f"[{label.upper()}] Engine ready in {load_time}s. VRAM: {vram_loaded} GiB", flush=True)

    suite_results = []

    try:
        for case in PROMPTS:
            case_name = case["name"]
            prompt = case["prompt"]
            max_new = case["max_new"]
            print(f"[{label.upper()}] Running case: {case_name} (max_new={max_new})...", flush=True)

            text = tpl.render([{"role": "user", "content": prompt}], enable_thinking=False)
            prompt_ids = tk.encode(text, parse_special=True)

            t0 = time.perf_counter()
            cancel_evt = threading.Event()
            generated = [t for t in engine.generate(prompt_ids, max_new, {"temperature": 0}, cancel_evt) if t is not None]
            elapsed_s = time.perf_counter() - t0

            gen_text = tk.decode(generated) if generated else ""
            last = dict(engine.last)

            prompt_tokens = len(prompt_ids)
            gen_tokens = len(generated)
            prompt_ms = last.get("prompt_ms", 0.0)
            decode_ms = last.get("decode_ms", 0.0)

            prefill_tok_s = round(prompt_tokens * 1000.0 / prompt_ms, 2) if prompt_ms > 0 else 0.0
            decode_tok_s = round(gen_tokens * 1000.0 / decode_ms, 2) if decode_ms > 0 else 0.0

            drafts_acc = last.get("drafts_accepted", 0)
            drafts_off = last.get("drafts_offered", 0)
            draft_rate = round(drafts_acc / drafts_off, 4) if drafts_off > 0 else 0.0

            hits = last.get("hits", 0)
            lookups = last.get("lookups", 0)
            hit_rate = round(hits / lookups, 4) if lookups > 0 else 0.0

            case_data = {
                "label": label,
                "case": case_name,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "output_text": gen_text,
                "elapsed_wall_s": round(elapsed_s, 3),
                "prompt_ms": round(prompt_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "prefill_tok_s": prefill_tok_s,
                "decode_tok_s": decode_tok_s,
                "speculation": {
                    "drafts_accepted": drafts_acc,
                    "drafts_offered": drafts_off,
                    "rate": draft_rate
                },
                "cache": {
                    "hits": hits,
                    "lookups": lookups,
                    "hit_rate": hit_rate
                },
                "engine_last": last
            }
            suite_results.append(case_data)
            print(f"[{label.upper()}] Case {case_name} done: {gen_tokens} tokens in {round(elapsed_s,2)}s "
                  f"(prefill: {prefill_tok_s} tok/s, decode: {decode_tok_s} tok/s, spec: {draft_rate*100:.1f}%)", flush=True)

    finally:
        print(f"[{label.upper()}] Unloading engine...", flush=True)
        engine.unload()
        exit_code = engine.exit_code()
        print(f"[{label.upper()}] Engine unloaded, exit code {exit_code}.", flush=True)

    return {
        "label": label,
        "load_time_s": load_time,
        "vram_init_gib": vram_init,
        "vram_loaded_gib": vram_loaded,
        "cases": suite_results
    }

def main():
    out_dir = Path("/data/strata-gfx906/eval-v0139/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for label in ["fork", "upstream"]:
        bin_path = BINARIES[label]
        if not bin_path.exists():
            print(f"Error: binary {bin_path} not found!", file=sys.stderr)
            sys.exit(1)
        res = run_suite(label, bin_path, out_dir)
        results[label] = res
        print(f"\nCooldown pause between {label} and next run...\n", flush=True)
        time.sleep(5)

    summary_file = out_dir / "comparison_summary.json"
    summary_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nEvaluation complete! Summary written to {summary_file}")

if __name__ == "__main__":
    main()
