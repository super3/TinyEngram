"""Compare eval_domain_tables.py result files as a markdown table.

    python eval_scripts/compare_arms.py results/base.json results/table.json \
        results/random_table.json results/lora.json [--baseline base] [--out summary.md]

Rows are suite entries plus per-suite averages, columns are arms, cells show the score
and the change against the baseline arm. The ± next to a task is the baseline's standard
error from lm-eval: treat changes smaller than about twice that as noise. If both a trained ``table`` arm and a
``random_table`` arm are present, a trained-minus-random row is added: the part of the
gain that comes from what the table learned rather than from the extra pathway.
"""

import argparse
import json
from collections import OrderedDict


def load(paths):
    runs = OrderedDict()
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        runs[r["name"]] = r
    return runs


def fmt(x, ppl=False):
    if x is None:
        return "-"
    return f"{x:.2f}" if ppl else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", default=None, help="run name to diff against (default: the base arm)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = load(args.files)
    names = list(runs)
    base = args.baseline or next((n for n, r in runs.items() if r["arm"] == "base"), names[0])

    entries = OrderedDict()
    for r in runs.values():
        for e, v in r["results"].items():
            entries.setdefault(e, (v["suite"], v["metric"]))

    lines = ["| suite | task | metric | " + " | ".join(names) + " |",
             "|---|---|---|" + "---|" * len(names)]
    suite_scores = OrderedDict()
    for e, (suite, metric) in entries.items():
        ppl = "perplexity" in (metric or "")
        cells = []
        b = runs[base]["results"].get(e, {}).get("score")
        for n in names:
            s = runs[n]["results"].get(e, {}).get("score")
            if s is not None and not ppl:
                suite_scores.setdefault((suite, n), []).append(s)
            if n == base or s is None or b is None:
                cells.append(fmt(s, ppl))
            else:
                cells.append(f"{fmt(s, ppl)} ({s - b:+.4f})")
        se = runs[base]["results"].get(e, {}).get("stderr")
        label = f"{e} (±{se:.3f})" if isinstance(se, (int, float)) else e
        lines.append(f"| {suite} | {label} | {metric} | " + " | ".join(cells) + " |")

    for suite in OrderedDict((s, None) for s, _ in entries.values()):
        avgs = {n: (sum(v) / len(v) if (v := suite_scores.get((suite, n))) else None) for n in names}
        b = avgs.get(base)
        cells = [fmt(avgs[n]) if (n == base or avgs[n] is None or b is None)
                 else f"{fmt(avgs[n])} ({avgs[n] - b:+.4f})" for n in names]
        lines.append(f"| **{suite} avg** | | | " + " | ".join(cells) + " |")

    trained = next((n for n, r in runs.items() if r.get("table_config", {}).get("arm") == "table"), None)
    random_ = next((n for n, r in runs.items() if r.get("table_config", {}).get("arm") == "random_table"), None)
    out = "\n".join(lines)
    if trained and random_:
        gap = []
        for suite in OrderedDict((s, None) for s, _ in entries.values()):
            a, c = suite_scores.get((suite, trained)), suite_scores.get((suite, random_))
            if a and c:
                gap.append(f"{suite}: {sum(a) / len(a) - sum(c) / len(c):+.4f}")
        out += f"\n\nTrained minus random-frozen table ({trained} vs {random_}): " + ", ".join(gap)

    tele_lines = []
    for n, r in runs.items():
        for e, layers in r.get("telemetry", {}).items():
            if layers:
                fr = sum(t["fire_rate"] for t in layers) / len(layers)
                g = sum(t["mean_gate_fired"] for t in layers) / len(layers)
                inj = sum(t["mean_inject_ratio_fired"] for t in layers) / len(layers)
                tele_lines.append(f"| {n} | {e} | {fr:.3f} | {g:.3f} | {inj:.4f} |")
    if tele_lines:
        out += ("\n\n| run | task | fire rate | mean gate (fired) | inject ratio (fired) |\n|---|---|---|---|---|\n"
                + "\n".join(tele_lines))

    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
