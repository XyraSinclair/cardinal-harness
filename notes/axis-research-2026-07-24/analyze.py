#!/usr/bin/env python3
"""Score the pre-registered predictions in AXIS_RESEARCH.md against the runs."""
import itertools
import json
from pathlib import Path

HERE = Path(__file__).parent
AXES = ["end_of_time", "ultrawattagedness", "clarity"]
MODELS = ["opus46", "gpt56sol", "mini54"]
LABEL = {
    0: "allotted-hydrogen", 1: "cosmic-SLOP", 2: "kernel-lock",
    3: "performed-4:47am", 4: "reed-solomon", 5: "pump-manual",
    6: "euclid-primes", 7: "couch-payroll", 8: "archive-seance",
    9: "corporate-slop", 10: "proton-decay", 11: "porch-light",
}

def ranks(axis, model):
    data = json.loads((HERE / f"sort-{axis}-{model}.json").read_text())
    out = {}
    for it in data["items"]:
        idx = int(it["id"].split("-")[1])
        out[idx] = it["rank"]
    return out

def spearman(ra, rb):
    n = len(ra)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in ra)
    return 1 - 6 * d2 / (n * (n * n - 1))

R = {(a, m): ranks(a, m) for a in AXES for m in MODELS}

print("== Spearman rank agreement per axis ==")
for a in AXES:
    for m1, m2 in itertools.combinations(MODELS, 2):
        print(f"  {a:20s} {m1:9s} vs {m2:9s}  rho={spearman(R[a,m1], R[a,m2]):+.3f}")

print("\n== Decoy rank shifts (rank 1 = top; positive shift = mini ranks it HIGHER) ==")
for a, decoy in (("end_of_time", 1), ("ultrawattagedness", 3)):
    fr = min(R[a, "opus46"][decoy], R[a, "gpt56sol"][decoy])
    mi = R[a, "mini54"][decoy]
    print(f"  {a:20s} {LABEL[decoy]:16s} frontier-best rank {fr:2d}  mini rank {mi:2d}  shift {fr - mi:+d}")

print("\n== Orthogonality: frontier end_of_time vs clarity ==")
for m in ("opus46", "gpt56sol"):
    print(f"  {m:9s} rho(end_of_time, clarity) = {spearman(R['end_of_time', m], R['clarity', m]):+.3f}")

print("\n== Full rankings ==")
for a in AXES:
    print(f"-- {a} --")
    for m in MODELS:
        order = sorted(R[a, m], key=lambda i: R[a, m][i])
        print(f"  {m:9s}: " + " > ".join(LABEL[i] for i in order))
