#!/usr/bin/env python3
"""Single-GPU real-model CLI baseline without MTP; no API/listener."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate
from tools.strata_tokenizer import Tokenizer

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.gpu < 0 or a.out.exists():
    ap.error("GPU must be nonnegative; output evidence must not already exist")
shard = ROOT / "models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
ple = shard.with_name(shard.name.replace("00001", "00002"))
pack = ROOT / "packs/iq2_xs"
tk = Tokenizer.from_gguf(shard)
tpl = ChatTemplate(pack / "tokenizer/chat_template.jinja")
text = tpl.render([{"role": "user", "content": "只回答一个数字：7+5等于几？"}], enable_thinking=False)
ids = tk.encode(text, parse_special=True)
cmd = [str(ROOT / "build-hip/strata"), "--pack", str(pack), "--native", str(shard), "--ple-gguf", str(ple),
       "--expert-profile", str(ROOT / "data/expert-profile.bin"), "--expert-cache", "auto",
       # Native packs require the verifier window >=2 and a prefill chunk,
       # even without --mtp. This selects the existing no-MTP verifier path.
       "--tokens", ",".join(map(str, ids)), "--max-new", "3", "--spec", "2", "--prefill", "128",
       "--max-context", "4096", "--kv", "int8", "--pool-workers", "6",
       "--adapt-every", "0", "--pcie-frac", "0", "--vram-reserve-mib", "1024"]
env = dict(os.environ)
env.pop("ROCR_VISIBLE_DEVICES", None)
env.pop("CUDA_VISIBLE_DEVICES", None)
env["HIP_VISIBLE_DEVICES"] = str(a.gpu)
a.out.parent.mkdir(parents=True, exist_ok=True)
result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
a.out.with_suffix(".stdout.log").write_text(result.stdout)
a.out.with_suffix(".stderr.log").write_text(result.stderr)
m = re.search(r"^output\s*:\s*([\d\s]+)$", result.stdout, re.MULTILINE)
generated = [int(x) for x in m.group(1).split()] if m else []
report = {"model": "GSQ-RCO IQ2_XS", "physical_gpu": a.gpu, "mtp": False, "command": cmd,
          "prompt_ids": ids, "ids": generated, "text": tk.decode(generated) if generated else "",
          "exit_code": result.returncode}
a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False), flush=True)
if result.returncode or not generated:
    raise SystemExit("CLI baseline failed; see the recorded logs")
