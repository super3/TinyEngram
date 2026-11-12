"""Wall-clock speculative-decoding test with llama.cpp's llama-server (test #2).

For each configuration, start llama-server, send the same prompts with greedy decoding,
and record generation speed from the server's own timings. Also checks that every
configuration produces exactly the same text as plain decoding (speculative decoding must
not change greedy output).

    python speedtest/llama_speed.py --server ./llama-server --model target.gguf \
        --draft draft.gguf --med_cache med.bin --gen_cache gen.bin \
        --prompts prompts.json --out results.json
"""

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_healthy(base, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_config(name, flags, args, prompts, logdir):
    base = f"http://127.0.0.1:{args.port}"
    cmd = [args.server, "-m", args.model, "--port", str(args.port), "-ngl", "99", "-c", "4096",
           "-np", "1", "--no-webui"] + flags
    log_path = os.path.join(logdir, f"server_{name}.log")
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        if not wait_healthy(base, proc):
            return {"error": "server did not start", "log": log_path}
        post(base + "/completion", {"prompt": "Hello", "n_predict": 8, "temperature": 0})  # warm-up
        for p in prompts:
            r = post(base + "/completion", {"prompt": p["text"], "n_predict": args.n_predict, "temperature": 0,
                                            "top_k": 1, "cache_prompt": False, "ignore_eos": True})
            t = r.get("timings", {})
            rows.append({"domain": p["domain"], "id": p["id"], "content": r.get("content", ""),
                         "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                         "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")})
    except Exception as exc:
        return {"error": repr(exc), "log": log_path, "rows": rows}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    stats = [l.strip() for l in open(log_path, errors="replace") if re.search(r"statistics|acceptance", l)]
    return {"rows": rows, "stats": stats[-6:], "log": log_path}


def summarize(results, prompts):
    domains = sorted({p["domain"] for p in prompts})
    ref = {(r["domain"], r["id"]): r["content"] for r in results.get("none", {}).get("rows", [])}
    table = {}
    for name, res in results.items():
        if "rows" not in res or not res["rows"]:
            table[name] = {"error": res.get("error")}
            continue
        entry = {}
        for d in domains:
            rs = [r for r in res["rows"] if r["domain"] == d and r["predicted_ms"]]
            n = sum(r["predicted_n"] for r in rs)
            ms = sum(r["predicted_ms"] for r in rs)
            same = sum(1 for r in rs if ref.get((d, r["id"])) == r["content"])
            entry[d] = {"tokens_per_s": 1000 * n / ms if ms else None, "prompts": len(rs),
                        "same_text_as_plain": same}
        table[name] = entry
    base = table.get("none", {})
    for name, entry in table.items():
        for d in domains:
            if d in entry and base.get(d, {}).get("tokens_per_s") and entry[d]["tokens_per_s"]:
                entry[d]["speedup"] = entry[d]["tokens_per_s"] / base[d]["tokens_per_s"]
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--med_cache", default=None)
    ap.add_argument("--gen_cache", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n_predict", type=int, default=256)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    draft = ["-md", args.draft, "-ngld", "99"] if args.draft else []
    configs = {"none": []}
    if args.draft:
        configs["draft"] = draft + ["--spec-type", "draft-simple"]
    if args.med_cache:
        configs["cache_med"] = ["--spec-type", "ngram-cache", "-lcs", args.med_cache]
    if args.gen_cache:
        configs["cache_gen"] = ["--spec-type", "ngram-cache", "-lcs", args.gen_cache]
    if args.draft and args.med_cache:
        configs["draft+cache_med"] = draft + ["--spec-type", "draft-simple,ngram-cache", "-lcs", args.med_cache]
    if args.draft and args.gen_cache:
        configs["draft+cache_gen"] = draft + ["--spec-type", "draft-simple,ngram-cache", "-lcs", args.gen_cache]
    configs["ngram_mod"] = ["--spec-type", "ngram-mod"]
    if args.only:
        keep = set(args.only.split(","))
        configs = {k: v for k, v in configs.items() if k in keep or k == "none"}

    logdir = os.path.dirname(os.path.abspath(args.out))
    results = {}
    for name, flags in configs.items():
        t0 = time.time()
        results[name] = run_config(name, flags, args, prompts, logdir)
        print(f"{name}: {'ERROR ' + results[name]['error'] if 'error' in results[name] else 'ok'} "
              f"({time.time() - t0:.0f}s)", flush=True)
        summary = summarize(results, prompts)
        with open(args.out, "w") as f:
            json.dump({"configs": configs, "summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summarize(results, prompts), indent=1))


if __name__ == "__main__":
    main()
