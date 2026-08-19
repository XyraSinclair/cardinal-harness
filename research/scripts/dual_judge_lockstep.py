#!/usr/bin/env python3
"""Two dense judges, in lockstep, landing every judgment durably.

The standing judging pair is two DENSE models — qwen3.8:27b and gemma4:31b —
because a dense model has the stronger theoretical claim to a single coherent
latent behind a subtle attribute. This harness keeps them from getting far
ahead of each other on any set: it runs each attribute on BOTH judges
concurrently and only advances when both have landed, so the lag between the
two judges on a set never exceeds one attribute's wall (~30-60s with the
logprob template — inside the 3-minute bar, near the 30s ideal).

Requirement: both judge endpoints must be reachable AT THE SAME TIME. On the
one 96GB card that means both dense models co-resident (two vLLM ports); the
two dense fp8 weights (~27GB + ~34GB) plus KV need the card roughly dedicated
(see notes: the production rerankers currently hold ~36GB of it). If only one
slot is available, run judge_and_land.py per model instead — but then the two
judges are a full sweep apart, not in lockstep.

Usage:
  dual_judge_lockstep.py \
    --a qwen38-27b@http://127.0.0.1:18023/v1 \
    --b gemma4-31b@http://127.0.0.1:18024/v1 \
    --corpus manifund.txt --attributes batteries/highdim_attributes.txt \
    --template canonical_bucket_v1 --budget 240 --seed 1 \
    --run-tag highdim-lockstep --outdir /tmp/lockstep
"""
import argparse, os, sys, time, concurrent.futures as cf
import judge_and_land as jl


def parse_endpoint(spec):
    model, base = spec.split("@", 1)
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="modelA@baseurlA")
    ap.add_argument("--b", required=True, help="modelB@baseurlB")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--attributes", required=True)
    ap.add_argument("--template", default="canonical_bucket_v1")
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-lag-seconds", type=float, default=180.0)
    a = ap.parse_args()
    ja, jb = parse_endpoint(a.a), parse_endpoint(a.b)
    for j in (ja, jb):
        os.makedirs(os.path.join(a.outdir, j[0]), exist_ok=True)
    corpus_lines = [l.rstrip("\n") for l in open(a.corpus) if l.strip()]
    corpus_name = os.path.basename(a.corpus)
    attrs = [l.strip() for l in open(a.attributes) if l.strip()]
    t0 = time.time()
    worst_lag = 0.0
    for i, attr in enumerate(attrs):
        # Each judge spawns its cardinal in its own OPENROUTER_BASE_URL env, so
        # the two endpoints run truly concurrently and advance together.
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(_run_one, j, a, attr, i, corpus_lines, corpus_name): j for j in (ja, jb)}
            times = {}
            for f in cf.as_completed(futs):
                model, n, dt = f.result()
                times[model] = (n, dt)
        lag = abs(times[ja[0]][1] - times[jb[0]][1])
        worst_lag = max(worst_lag, lag)
        print(f"[{i+1}/{len(attrs)}] {attr[:40]!r}: "
              f"{ja[0]} {times[ja[0]][0]}j/{times[ja[0]][1]:.0f}s | "
              f"{jb[0]} {times[jb[0]][0]}j/{times[jb[0]][1]:.0f}s | lag {lag:.0f}s", flush=True)
        if lag > a.max_lag_seconds:
            print(f"  WARNING: per-attribute lag {lag:.0f}s exceeds "
                  f"{a.max_lag_seconds:.0f}s bound", flush=True)
    print(f"done: {len(attrs)} attributes x2 judges in {time.time()-t0:.0f}s, "
          f"worst per-attribute lag {worst_lag:.0f}s -> ratiometer.judgments")


def _run_one(judge, args, attr, idx, corpus_lines, corpus_name):
    """Spawn one judge's cardinal run with its own OPENROUTER_BASE_URL."""
    model, base = judge
    import subprocess, json
    slug = f"a{idx:03d}"
    trace = os.path.join(args.outdir, model, f"{slug}.trace.jsonl")
    out = os.path.join(args.outdir, model, f"{slug}.json")
    env = dict(os.environ)
    env["OPENROUTER_BASE_URL"] = base
    env.setdefault("OPENROUTER_API_KEY", "local")
    cmd = [jl.CARDINAL, "sort", args.corpus, "--by", attr, "--model", model,
           "--budget", str(args.budget), "--seed", str(args.seed),
           "--template", args.template, "--trace", trace, "--format", "json"]
    t = time.time()
    with open(out, "w") as fo, open(out + ".err", "w") as fe:
        subprocess.run(cmd, stdout=fo, stderr=fe, env=env, check=True, timeout=3600)
    n = jl.land(trace, corpus_lines, corpus_name, attr, args.seed, args.run_tag)
    return model, n, time.time() - t


if __name__ == "__main__":
    main()
