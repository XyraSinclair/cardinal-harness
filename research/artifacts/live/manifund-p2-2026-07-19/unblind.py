#!/usr/bin/env python3
"""P2 unblinding: registered predictions vs ACX 2024 ground truth.

Run ONLY after the prediction commit. Reads the committed rerank response
and data/manifund/ground_truth.csv; reports AUC (funded vs not), Spearman
(combined score vs dollars raised), per-attribute AUCs, and the
both-directions disagreement shortlist. Prints denominators everywhere.
"""

import csv
import json
import math
import sys
from pathlib import Path

PACK = Path(__file__).parent
ROOT = PACK.parent.parent.parent


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney), ties get half credit."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def main():
    response = json.load(open(PACK / "acx2024-response.json"))
    results = response["entities"]
    gt = {}
    for row in csv.DictReader(open(ROOT / "data/manifund/ground_truth.csv")):
        if "acx2024" in row["cohorts"]:
            # Judgment items carry the (possibly truncated) slug as their id.
            gt[row["slug"]] = {
                "funded": "funded" in row["cohorts"],
                "raised": float(row["raised"] or 0),
            }
    scored = [r for r in results if r["id"] in gt]
    print(f"matched {len(scored)} of {len(results)} scored items to ACX 2024 ground truth")
    combined = [r["u_mean"] for r in scored]
    funded = [gt[r["id"]]["funded"] for r in scored]
    raised = [gt[r["id"]]["raised"] for r in scored]
    n_funded = sum(funded)
    print(f"cohort: n={len(scored)}, funded={n_funded}")
    a = auc(combined, funded)
    print(f"AUC (combined score vs funded): {a:.3f}")
    s = spearman(combined, raised)
    print(f"Spearman (combined score vs dollars raised): {s:+.3f}")
    s_funded_only = spearman(
        [c for c, f in zip(combined, funded) if f],
        [d for d, f in zip(raised, funded) if f],
    )
    print(f"Spearman among funded only (score vs dollars): {s_funded_only:+.3f} (n={n_funded})")

    # Per-attribute AUCs.
    attr_ids = [a["id"] for a in json.load(
        open(ROOT / "data/manifund/requests/p2-acx2024-4attr.json"))["attributes"]]
    for attr in attr_ids:
        vals = [r["attribute_scores"][attr]["latent_mean"] for r in scored]
        print(f"  AUC[{attr}]: {auc(vals, funded):.3f}")

    # Disagreement shortlists, both directions.
    order = sorted(scored, key=lambda r: -r["u_mean"])
    rank = {r["id"]: i + 1 for i, r in enumerate(order)}
    underrated = sorted(
        (r for r in scored if not gt[r["id"]]["funded"]),
        key=lambda r: rank[r["id"]],
    )[:5]
    overrated = sorted(
        (r for r in scored if gt[r["id"]]["funded"]),
        key=lambda r: -rank[r["id"]],
    )[:5]
    print("\nmodel-ranked HIGH but not funded (underrated by realized funding?):")
    for r in underrated:
        print(f"  #{rank[r['id']]:>2} {r['id']}  score {r['u_mean']:+.3f} ± {r['u_std']:.3f}")
    print("model-ranked LOW but funded (overrated by realized funding?):")
    for r in overrated:
        print(f"  #{rank[r['id']]:>2} {r['id']}  score {r['u_mean']:+.3f} ± {r['u_std']:.3f}  raised ${gt[r['id']]['raised']:,.0f}")


if __name__ == "__main__":
    sys.exit(main())
