#!/usr/bin/env python3
"""Per-attribute cross-judge agreement from ratiometer.judgments.

For every attribute in a run_tag, reconstruct each judge's directional verdict
per unordered pair (net higher_ranked vote, folding the swapped presentation)
and score how often two judges agree on direction over pairs both judged.

agreement = matching-direction pairs / co-judged decisive pairs

Low agreement on a stable-per-judge attribute = the criterion is
model-idiosyncratic (each judge reified a different construct) — the signal
that an attribute needs sharper phrasing before its ranking is trusted.

Usage: cross_judge_agreement.py <run_tag> [judgeA judgeB]
"""
import subprocess, sys, json, collections, itertools

SSH = "/usr/bin/ssh"
CH = "/data/clickhouse-twitter-lab/bin/clickhouse client --port 19000 --query"


def q(sql):
    p = subprocess.run([SSH, "colo2", f"{CH} {json.dumps(sql)}"],
                       capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr[:400])
    return [l.split("\t") for l in p.stdout.splitlines() if l]


def main():
    run_tag = sys.argv[1]
    # pull decisive judgments: attribute, model, unordered-pair key, direction
    # (winner entity hash). swapped is already folded into higher_ranked=A/B by
    # the judge; we map to the actual winning entity hash.
    rows = q("SELECT attribute, model, entity_a_hash, entity_b_hash, higher_ranked "
             "FROM ratiometer.judgments WHERE run_tag = '" + run_tag +
             "' AND refused = 0 AND higher_ranked IN ('A','B')")
    models = sorted({r[1] for r in rows})
    if len(sys.argv) >= 4:
        ja, jb = sys.argv[2], sys.argv[3]
    elif len(models) >= 2:
        ja, jb = models[0], models[1]
    else:
        print(f"only one judge present ({models}); need two. run the sweep on a second model first.")
        return
    # attribute -> model -> pairkey -> winner_hash
    win = collections.defaultdict(lambda: collections.defaultdict(dict))
    for attr, model, ah, bh, hr in rows:
        key = tuple(sorted((ah, bh)))
        winner = ah if hr == "A" else bh
        win[attr][model][key] = winner
    print(f"judges: {ja}  vs  {jb}   (run_tag={run_tag})\n")
    print(f"| {'attribute':52} | co-pairs | agreement |")
    print("|" + "-" * 54 + "|----------|-----------|")
    out = []
    for attr in sorted(win):
        A, B = win[attr].get(ja, {}), win[attr].get(jb, {})
        shared = set(A) & set(B)
        if not shared:
            continue
        agree = sum(1 for k in shared if A[k] == B[k])
        frac = agree / len(shared)
        out.append((frac, attr, len(shared)))
    for frac, attr, n in sorted(out):
        print(f"| {attr[:52]:52} | {n:8} | {frac:9.3f} |")
    if out:
        mean = sum(f for f, _, _ in out) / len(out)
        print(f"\nmean cross-judge agreement over {len(out)} attributes: {mean:.3f}")
        print("lowest (most model-idiosyncratic, rephrase candidates):")
        for frac, attr, n in sorted(out)[:5]:
            print(f"  {frac:.3f}  {attr}")


if __name__ == "__main__":
    main()
