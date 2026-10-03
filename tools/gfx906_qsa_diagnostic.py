#!/usr/bin/env python3
"""Analyze private real-QSA captures/logits; sampled observations are not global model parity."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path


def raw(path, code, count):
    path = Path(path)
    a = array.array(code)
    if path.stat().st_size != count * a.itemsize:
        raise ValueError(f"wrong byte count: {path}")
    with path.open("rb") as f:
        a.fromfile(f, count)
    return a


def metric(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("incompatible/empty vectors")
    if any(not math.isfinite(x) for x in a) or any(not math.isfinite(x) for x in b):
        raise ValueError("nonfinite vectors")
    delta = [float(x) - float(y) for x, y in zip(a, b)]
    err2 = math.fsum(x * x for x in delta)
    ref2 = math.fsum(float(x) * float(x) for x in b)
    return dict(elements=len(a), unequal=sum(x != y for x, y in zip(a, b)),
                max_abs=max(map(abs, delta)), relative_l2=math.sqrt(err2 / max(1e-30, ref2)))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def operators(control, online):
    records = []
    names = {p.name for p in control.glob("device*-layer*-pos*")}
    other = {p.name for p in online.glob("device*-layer*-pos*")}
    if not names or names != other:
        raise ValueError("missing or incompatible operator capture coverage")
    inputs = ["geometry.i64", "query-rows.i64", "q.f32", "k.i8", "v.i8",
              "k-scale.f16", "v-scale.f16", "ids.i32", "steps.i32", "pages.i32"]
    for name in sorted(names):
        a, b = control / name, online / name
        ga, gb = raw(a / "geometry.i64", "q", 9), raw(b / "geometry.i64", "q", 9)
        if ga != gb or ga[0] != 1 or not 1 <= ga[1] <= 3 or not 1 <= ga[2] <= 32768:
            raise ValueError("capture geometry differs or is invalid")
        n = ga[1] * 24 * 256
        record = dict(capture=name, geometry=list(ga), sampled_queries=ga[1],
                      input_equal={key: digest(a / key) == digest(b / key) for key in inputs},
                      query=metric(raw(a / "q.f32", "f", n), raw(b / "q.f32", "f", n)),
                      output=metric(raw(a / "actual.f32", "f", n), raw(b / "actual.f32", "f", n)))
        record["identical_inputs"] = all(record["input_equal"].values())
        # Different upstream inputs must never be labelled an isolated attention-operator comparison.
        record["isolated_operator_comparison"] = record["identical_inputs"]
        records.append(record)
    return records


def predictions(control, online):
    names = {p.name for p in control.glob("prediction*.i64")}
    other = {p.name for p in online.glob("prediction*.i64")}
    if not names or names != other:
        raise ValueError("missing or incompatible committed-prediction coverage")
    records, common_prefix = [], True
    first = None
    indices = []
    for name in sorted(names, key=lambda v: int(v.removeprefix("prediction").removesuffix(".i64"))):
        a, b = raw(control / name, "q", 5), raw(online / name, "q", 5)
        index = a[1]
        if a[0] != 1 or b[0] != 1 or a[1:3] != b[1:3] or a[4] != b[4] or not 1 <= a[4] <= 1000000:
            raise ValueError("prediction coordinate/vocabulary mismatch")
        indices.append(index)
        filename = name.removesuffix(".i64") + ".f32"
        va, vb = raw(control / filename, "f", a[4]), raw(online / filename, "f", b[4])
        m = metric(va, vb)
        ta = sorted(range(len(va)), key=lambda i: (-va[i], i))[:2]
        tb = sorted(range(len(vb)), key=lambda i: (-vb[i], i))[:2]
        if a[3] != ta[0] or b[3] != tb[0]:
            raise ValueError("diagnostic requires verified top-k1 greedy predictions")
        equal = a[3] == b[3]
        record = dict(index=index, position=a[2], same_prior_ids=common_prefix,
                      control_token=a[3], online_token=b[3], equal=equal, difference=m,
                      control_top2=[dict(id=i, logit=va[i]) for i in ta],
                      online_top2=[dict(id=i, logit=vb[i]) for i in tb],
                      control_margin=va[ta[0]] - va[ta[1]] if len(ta) > 1 else None,
                      online_margin=vb[tb[0]] - vb[tb[1]] if len(tb) > 1 else None)
        if common_prefix and not equal:
            first = record
            x, y = a[3], b[3]
            record["flip_pair"] = dict(control_gap=va[x] - va[y], online_gap=vb[x] - vb[y],
                                       control_id_delta=vb[x] - va[x], online_id_delta=vb[y] - va[y])
        records.append(record)
        common_prefix = common_prefix and equal
    if indices != list(range(len(indices))):
        raise ValueError("noncontiguous prediction sequence")
    return dict(generated=len(records), identical=common_prefix, first_divergence=first, predictions=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--online", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = dict(schema=1, scope="sampled real-input attention and greedy committed-logit diagnostics; NOT performance or global quality",
                  operators=operators(args.control / "attention", args.online / "attention"),
                  generation=predictions(args.control / "logits", args.online / "logits"))
    with args.output.open("x") as f:
        args.output.chmod(0o600)
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(operator_captures=len(result["operators"]),
                          identical_input_captures=sum(v["identical_inputs"] for v in result["operators"]),
                          generated=result["generation"]["generated"],
                          first_divergence=result["generation"]["first_divergence"]), allow_nan=False))


if __name__ == "__main__":
    main()
