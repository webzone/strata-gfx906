#!/usr/bin/env python3
"""Validate and summarize archived gfx906 short-model evidence (CPU-only)."""
import argparse
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise SystemExit("incomplete/failed evidence: " + message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    a = ap.parse_args()
    root = a.directory
    log = (root / "ctest-final.log").read_text()
    counts = {word: len(re.findall(r"Test\s+#\d+:.*?\.\.\..*?\b" + word + r"\b", log))
              for word in ("Passed", "Skipped", "Failed")}
    require(counts == {"Passed": 40, "Skipped": 1, "Failed": 0}, "CTest result counts")
    python_log = (root / "python-tests-final.log").read_text()
    require("Ran 18 tests" in python_log and "\nOK\n" in python_log, "18 Python regressions")
    handoff = (root / "hip-layer-handoff.log").read_text()
    require("137980416 exact float/guard checks" in handoff and "hip_layer_handoff OK" in handoff,
            "dual-GPU transport probe")
    baseline = json.loads((root / "cli-no-mtp-v2.json").read_text())
    require(baseline["exit_code"] == 0 and baseline["mtp"] is False, "CLI no-MTP baseline")
    models = {}
    reference = None
    for name in ("smoke-dual", "smoke-single", "smoke-dual-threshold1", "smoke-reverse"):
        report = json.loads((root / (name + ".json")).read_text())
        require(report["status"] == "passed_transport_and_generation" and report["exit_code"] == 0, name)
        cases = report["cases"]
        require([c["name"] for c in cases] == ["arithmetic", "literal", "code", "sequence"], name+" cases")
        if reference is None:
            reference = cases
        for ref, case in zip(reference, cases):
            require(ref["prompt_ids"] == case["prompt_ids"] and ref["ids"] == case["ids"],
                    name+" token parity for "+case["name"])
        models[name] = [{"name": c["name"], "tokens": len(c["ids"]),
                         "decode_tok_s": len(c["ids"])*1000/c["engine"]["decode_ms"],
                         "drafts_accepted": c["engine"]["drafts_accepted"],
                         "drafts_offered": c["engine"]["drafts_offered"],
                         "cache_hit_fraction": c["engine"]["hits"]/c["engine"]["lookups"]}
                        for c in cases]
    require(baseline["ids"] == reference[0]["ids"], "no-MTP arithmetic token parity")
    require((root / "reference.exit").read_text().strip() == "0", "pinned CPU implementation exit")
    cpu_text = (root / "reference.stdout.log").read_text().strip()
    require(cpu_text == "12 [end of text]", "pinned CPU implementation arithmetic output")
    health = json.loads((root / "environment-final.json").read_text())
    edac = json.loads((root / "edac-readonly.json").read_text())
    kernel = health["kernel_recent"]["stdout"]
    hardware_warning = "MCE MEMORY ERROR" in kernel or "CMCI storm" in kernel
    print(json.dumps({"ctest": counts, "python_passed": 18, "handoff_exact_checks": 137980416,
                      "all_four_model_modes_token_parity": True, "cli_no_mtp_arithmetic_parity": True,
                      "cpu_reference_arithmetic_text_match": True, "models": models,
                      "hardware_warning_detected": hardware_warning,
                      "host_dram_ce_cumulative": int(edac["counters"]["/sys/devices/system/edac/mc/mc0/ce_count"]),
                      "host_dram_ue_cumulative": int(edac["counters"]["/sys/devices/system/edac/mc/mc0/ue_count"]),
                      "production_or_stability_acceptance": False,
                      "limitations": ["short prompts, 4K configured context; no long-soak/full-logit parity",
                                      "timings are individual smoke cases, not an isolated comparative benchmark",
                                      "threshold1 still offers some drafts; never label it MTP-off",
                                      "CPU reference matches one arithmetic text, not all model logits",
                                      "host DRAM MCE/CE/CMCI storm: further stress paused; cumulative CE is not this-run delta"]}, indent=2))


if __name__ == "__main__":
    main()
