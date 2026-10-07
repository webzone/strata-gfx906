#!/usr/bin/env python3
"""Single-shot vs N-concurrent probe against a live Strata service.

Sends N mutually-unique, same-shape prompts (either one at a time or released together),
records every response's engine-reported `usage`/`timings` verbatim, and snapshots `/metrics`
while the batch is in flight. Read-only towards the service: it starts nothing, stops nothing
and changes no setting.

  python3 tools/gfx906_concurrency_probe.py --url http://127.0.0.1:8082 --streams 1 \
      --items 150 --max-tokens 256 --output logs/probe-1x
  python3 tools/gfx906_concurrency_probe.py --url http://127.0.0.1:8082 --streams 4 \
      --items 150 --max-tokens 256 --output logs/probe-4x

Every rate in the summary is the engine's own number, except `aggregate_*`, which is
completion tokens divided by the client-measured in-flight window and is labelled as such.
"""
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

FILLER_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
                "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
                "x-ray yankee zulu")


def prompt_text(salt: int, items: int) -> str:
    """`items` lines, each carrying a unique index+salt so no K/V can be reused.

    The task is long-form on purpose: a short answer would stop after a few tokens and give
    no decode sample. `finish_reason` must be `length` for the decode rate to cover
    `--max-tokens` tokens.
    """
    lines = [f"{n}. ledger {salt:03d}-{n:06d} quota {salt * 7919 + n} {FILLER_WORDS}"
             for n in range(1, items + 1)]
    return ("Context log (unique per probe, read it, then do the task at the end):\n"
            + "\n".join(lines)
            + "\n\nTask: first explain, in detail and in prose, how a B-tree insert works "
              "including node splitting and rebalancing. Then list the integers from 1 to 1000, "
              "one per line. Keep going until you are stopped; do not summarize.")


def get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


class MetricsMonitor(threading.Thread):
    """Poll /metrics while the batch is in flight; keep the raw snapshots."""

    def __init__(self, url: str, interval: float = 1.0):
        super().__init__(daemon=True)
        self.url = url.rstrip("/") + "/metrics"
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                m = get_json(self.url)
                live = m.get("live", {})
                self.samples.append({
                    "t": round(time.time(), 3),
                    "state": live.get("state"),
                    "phase": live.get("phase"),
                    "running": live.get("running"),
                    "queued": live.get("queued"),
                    "parallel": live.get("parallel"),
                    "generated": live.get("generated"),
                    "tok_s": live.get("tok_s"),
                    "prefill_tok_s_mean": live.get("prefill_tok_s_mean"),
                    "slots": [{"slot": s.get("slot"), "state": s.get("state")}
                              for s in live.get("slots", [])],
                })
            except Exception as exc:  # a dropped poll must not break the probe
                self.samples.append({"t": round(time.time(), 3), "error": str(exc)})
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--streams", type=int, default=1, help="requests in this phase")
    ap.add_argument("--items", type=int, default=150, help="filler lines per prompt")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--salt-base", type=int, default=1, help="unique salt per stream")
    ap.add_argument("--stagger-ms", type=float, default=0.0,
                    help="0 releases all streams together; >0 spaces their starts")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--wait-idle", type=int, default=0,
                    help="seconds to wait for live.running==0 and queued==0 before starting")
    ap.add_argument("--engine-log", default=None,
                    help="engine log file; the lines written during the phase are captured verbatim")
    a = ap.parse_args()

    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=False)
    url = a.url.rstrip("/")
    model = get_json(url + "/v1/models")["data"][0]["id"]

    def run_stream(idx: int, out: dict):
        body = {
            "model": model, "messages": [{"role": "user", "content": prompt_text(
                a.salt_base + idx, a.items)}],
            "max_tokens": a.max_tokens, "temperature": 0.0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                payload = json.loads(r.read().decode())
            t1 = time.time()
            choice = payload.get("choices", [{}])[0]
            out.update({
                "stream_index": idx,
                "client_start": round(t0, 3),
                "client_end": round(t1, 3),
                "client_wall_s": round(t1 - t0, 2),
                "finish_reason": choice.get("finish_reason"),
                "content": choice.get("message", {}).get("content"),
                "usage": payload.get("usage"),
                "timings": payload.get("timings"),
            })
        except Exception as exc:
            out.update({"stream_index": idx, "error": repr(exc)})

    # Optional: wait for the service to be free, so the phase is not mixed with other traffic.
    idle_wait = {"requested_s": a.wait_idle, "waited_s": 0.0, "idle_at_start": None}
    if a.wait_idle:
        waited_from = time.time()
        deadline = waited_from + a.wait_idle
        while time.time() < deadline:
            live = get_json(url + "/metrics").get("live", {})
            if not live.get("running") and not live.get("queued"):
                idle_wait["idle_at_start"] = True
                break
            time.sleep(1.0)
            idle_wait["waited_s"] = round(time.time() - waited_from, 1)
        else:
            live = get_json(url + "/metrics").get("live", {})
            idle_wait["idle_at_start"] = not live.get("running") and not live.get("queued")

    log_size_before = Path(a.engine_log).stat().st_size if a.engine_log else None

    before = get_json(url + "/metrics")
    monitor = MetricsMonitor(url)
    monitor.start()

    results = [{} for _ in range(a.streams)]
    threads = []
    start = time.time()
    for i in range(a.streams):
        t = threading.Thread(target=run_stream, args=(i, results[i]))
        threads.append(t)
        t.start()
        if a.stagger_ms:
            time.sleep(a.stagger_ms / 1000.0)
    for t in threads:
        t.join()
    end = time.time()

    monitor.stop()
    monitor.join(timeout=5)
    after = get_json(url + "/metrics")

    # Engine log delta: the authoritative per-request record (reused vs read tokens, ms, rates).
    engine_log_delta = None
    if a.engine_log:
        raw = Path(a.engine_log).open("rb")
        raw.seek(log_size_before)
        engine_log_delta = raw.read().decode("utf-8", "replace")
        raw.close()

    ok = [r for r in results if "usage" in r]
    usage_sum = sum((r["usage"] or {}).get("completion_tokens", 0) for r in ok)
    window = max((r["client_end"] for r in ok), default=0) - min((r["client_start"] for r in ok), default=0)
    decode = [((r["timings"] or {}).get("predicted_per_second")) for r in ok]
    prefill = [((r["timings"] or {}).get("prompt_per_second")) for r in ok]
    summary = {
        "streams": a.streams,
        "ok": len(ok),
        "prompt_tokens": [r["usage"].get("prompt_tokens") for r in ok],
        "cached_tokens": [((r["timings"] or {}).get("cache_n")) for r in ok],
        "finish_reason": [r.get("finish_reason") for r in ok],
        "completion_tokens": [r["usage"].get("completion_tokens") for r in ok],
        "engine_decode_tok_s": decode,
        "engine_prefill_tok_s": prefill,
        "decode_tok_s_min": min([d for d in decode if d], default=None),
        "decode_tok_s_max": max([d for d in decode if d], default=None),
        "aggregate_completion_tok_s_client_window":
            round(usage_sum / window, 2) if window and usage_sum else None,
        "client_window_s": round(window, 2),
        "max_live_running_observed": max([s.get("running") or 0 for s in monitor.samples], default=None),
        "client_estimated_prefill_tok_s": [
            round((r["usage"].get("prompt_tokens", 0) - ((r["timings"] or {}).get("cache_n") or 0))
                  / max(r["client_wall_s"] - ((r["timings"] or {}).get("predicted_ms") or 0) / 1000.0, 0.001), 1)
            for r in ok],
        "contaminated_by_other_traffic": bool(
            max([s.get("running") or 0 for s in monitor.samples], default=0) > a.streams),
        "max_slots_busy_observed": max(
            [sum(1 for s in (s.get("slots") or []) if s.get("state") != "idle")
             for s in monitor.samples], default=None),
    }

    record = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        "url": url, "model": model,
        "request": {"streams": a.streams, "items": a.items, "max_tokens": a.max_tokens,
                    "temperature": 0.0, "stream": False, "stagger_ms": a.stagger_ms,
                    "salt_base": a.salt_base, "chat_template_kwargs": {"enable_thinking": False}},
        "elapsed_s": round(end - start, 2),
        "summary": summary,
        "streams": results,
        "idle_wait": idle_wait,
        "metrics_before": before,
        "engine_log_delta": engine_log_delta,
        "metrics_during": monitor.samples,
        "metrics_after": after,
    }
    (folder / "probe-results.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
