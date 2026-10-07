#!/usr/bin/env python3
"""Real HTTP regression: a long SSE request must finish despite three short concurrent bursts.

Run only against an explicitly reserved test server. Saves raw requests/responses; does not start/stop it.
"""
import argparse
import json
from pathlib import Path
import threading
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=False)
    url = a.url.rstrip("/")
    headers = {"Content-Type": "application/json", "Origin": "https://arbitrary-origin.example"}
    body = {"model": "m", "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content":
                "Context log:\n" + "alpha beta gamma delta epsilon\n" * 6000 +
                "\nNow write a numbered list from 1 to 2000, with each number on its own line. "
                "Continue until the limit; do not summarize."}]}
    (folder / "long.request.json").write_text(json.dumps(body, indent=2) + "\n")
    state = {"chunks": 0, "usage": None, "error": None}
    cv = threading.Condition()

    def stream():
        try:
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r, (folder / "long.response.sse").open("wb") as raw:
                assert r.headers.get("Access-Control-Allow-Origin") == "*", dict(r.headers)
                for line in r:
                    raw.write(line)
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if "error" in ev:
                        raise RuntimeError(ev["error"])
                    with cv:
                        if ev.get("usage"):
                            state["usage"] = ev["usage"]
                        if any(c.get("delta", {}).get("content") for c in ev.get("choices", [])):
                            state["chunks"] += 1
                        cv.notify_all()
        except Exception as e:
            with cv:
                state["error"] = repr(e)
                cv.notify_all()

    started = time.monotonic()
    worker = threading.Thread(target=stream, daemon=True)
    worker.start()
    bursts = []
    try:
        with cv:
            assert cv.wait_for(lambda: state["chunks"] >= 8 or state["error"], 300), "no initial SSE progress"
            assert state["error"] is None, state
        for i in range(3):
            before = state["chunks"]
            short = {"model": "m", "max_tokens": 16, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False},
                     "messages": [{"role": "user", "content": f"Interruption {i}: what is 17 + 25? Reply with only the number."}]}
            (folder / f"short-{i}.request.json").write_text(json.dumps(short, indent=2) + "\n")
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(short).encode(), headers=headers)
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=120) as r:
                assert r.headers.get("Access-Control-Allow-Origin") == "*", dict(r.headers)
                raw = r.read()
            (folder / f"short-{i}.response.json").write_bytes(raw)
            answer = json.loads(raw)
            assert answer["choices"][0]["message"]["content"].strip() == "42", answer
            with cv:
                after_short = state["chunks"]
                assert cv.wait_for(lambda: state["chunks"] >= after_short + 8 or state["error"], 120), "long stream stalled"
                assert state["error"] is None, state
                bursts.append({"seconds": time.monotonic() - t0, "before": before,
                               "after_short": after_short, "after": state["chunks"]})
        worker.join(180)
        assert not worker.is_alive(), "long stream did not finish"
        assert state["error"] is None, state
        assert state["usage"] and state["usage"]["completion_tokens"] >= 512, state
        summary = {"ok": True, "seconds": time.monotonic() - started, "bursts": bursts, **state}
    except Exception as e:
        summary = {"ok": False, "seconds": time.monotonic() - started, "bursts": bursts, **state, "failure": repr(e)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
