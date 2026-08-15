#!/usr/bin/env python3
"""Relentless judging: run cardinal over (corpus x attributes), land every
pairwise judgment durably in scry ClickHouse (ratiometer.judgments on colo2).

Production posture: pairwise cache ON (measurement batteries elsewhere use
--no-cache), every judgment traced, traces denormalized (entity texts resolved
by corpus line index) and inserted content-addressed — ReplacingMergeTree on
cache_key_hash makes replays free.

Usage:
  judge_and_land.py --corpus manifund.txt --attributes batteries/manifund_attributes.txt \
      --model gemma4-26b-a4b --budget 240 --seed 1 --run-tag manifund-2026-08-15 \
      --outdir /tmp/judge-runs [--start-at N]

Env: OPENROUTER_BASE_URL (local judges), CARDINAL_PAIRWISE_MAX_OUTPUT_TOKENS
(tight-context serves). Landing goes over ssh to colo2's clickhouse
(/data/clickhouse-twitter-lab/bin/clickhouse client --port 19000) with
JSONEachRow on stdin (no shell interpolation of data).
"""
import argparse, json, os, subprocess, sys, time

CARDINAL = os.path.expanduser("~/projects/llmsorting/target/release/cardinal")
SSH = "/usr/bin/ssh"


def run_cell(corpus, attr, model, budget, seed, outdir, idx):
    slug = f"a{idx:03d}"
    out = os.path.join(outdir, f"{slug}.json")
    trace = os.path.join(outdir, f"{slug}.trace.jsonl")
    errf = os.path.join(outdir, f"{slug}.err")
    env = dict(os.environ)
    env.setdefault("OPENROUTER_API_KEY", "local")
    cmd = [CARDINAL, "sort", corpus, "--by", attr, "--model", model,
           "--budget", str(budget), "--seed", str(seed),
           "--trace", trace, "--format", "json"]
    with open(out, "w") as fo, open(errf, "w") as fe:
        subprocess.run(cmd, stdout=fo, stderr=fe, env=env, check=True, timeout=3600)
    return out, trace


def land(trace_path, corpus_lines, corpus_name, attr, seed, run_tag):
    rows = []
    for line in open(trace_path):
        d = json.loads(line)
        if d.get("error"):
            err = str(d["error"])
        else:
            err = ""
        rows.append({
            "ts": d["timestamp_ms"] / 1000.0,
            "run_tag": run_tag,
            "corpus": corpus_name,
            "model": d["model"],
            "served_model": d.get("served_model") or d["model"],
            "template": d["prompt_template_slug"],
            "attribute": attr,
            "attribute_prompt_hash": d["attribute_prompt_hash"],
            "seed": seed,
            "entity_a": corpus_lines[d["entity_a_index"]],
            "entity_b": corpus_lines[d["entity_b_index"]],
            "entity_a_hash": d["entity_a_hash"],
            "entity_b_hash": d["entity_b_hash"],
            "cache_key_hash": d["cache_key_hash"],
            "higher_ranked": d.get("higher_ranked") or "",
            "ratio": d.get("ratio") if d.get("ratio") is not None else 0.0,
            "confidence": d.get("confidence") if d.get("confidence") is not None else 0.0,
            "swapped": bool(d.get("swapped")),
            "cached": bool(d.get("cached")),
            "refused": bool(d.get("refused")),
            "input_tokens": d.get("input_tokens") or 0,
            "output_tokens": d.get("output_tokens") or 0,
            "error": err,
        })
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
    proc = subprocess.run(
        [SSH, "colo2", "/data/clickhouse-twitter-lab/bin/clickhouse", "client",
         "--port", "19000", "--query",
         "INSERT INTO ratiometer.judgments FORMAT JSONEachRow"],
        input=payload, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"land failed: {proc.stderr.decode()[:500]}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--start-at", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    corpus_lines = [l.rstrip("\n") for l in open(a.corpus) if l.strip()]
    corpus_name = os.path.basename(a.corpus)
    attrs = [l.strip() for l in open(a.attributes) if l.strip()]
    landed_total = 0
    t0 = time.time()
    for i, attr in enumerate(attrs):
        if i < a.start_at:
            continue
        t = time.time()
        out, trace = run_cell(a.corpus, attr, a.model, a.budget, a.seed, a.outdir, i)
        n = land(trace, corpus_lines, corpus_name, attr, a.seed, a.run_tag)
        landed_total += n
        print(f"[{i+1}/{len(attrs)}] {attr!r}: {n} judgments landed "
              f"({time.time()-t:.1f}s)", flush=True)
    print(f"done: {landed_total} judgments in {time.time()-t0:.0f}s -> ratiometer.judgments")


if __name__ == "__main__":
    main()
