#!/usr/bin/env python3
"""Replication verdicts against the frozen rules in REPLICATION_SPEC.md.

Arm 1: test-retest Spearman per original cell (24 cells); power gate =
median >= 0.80. Arm 2: per admitted axis x new small model, the wave-2
signature statistic with the small model substituted for mini54 —
T3-substitute (decoy >= rank 7 for BOTH original frontier runs, small
model places it >= 3 ranks above the frontiers' best) or T2-analog
(fr-fr rho minus mean fr-small rho >= 0.20) — using whichever criterion
admitted the axis in wave 2 (RESULTS-WAVE2.md; T2+T3 axes accept either).
TIER-GENERAL = >= 2 of 3 small models show the signature.
Program verdict: power gate AND >= 3 of 6 admitted axes TIER-GENERAL.
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
W2 = HERE / "wave2"
REP = W2 / "replication"
ORIG_MODELS = ["opus46", "gpt56sol", "mini54"]
NEW_MODELS = ["haiku45", "dsflash", "gpt5mini"]
DECOY = 1  # 0-based index of the primary decoy in every probe file
# axis -> admitting criteria from RESULTS-WAVE2.md
ADMITTED = {
    "eschatological_seriousness": {"T3"},
    "rosetta_load": {"T2"},
    "scar_tissue_density": {"T2", "T3"},
    "live_wire_prose": {"T2", "T3"},
    "antimemetic_payload": {"T2", "T3"},
    "hostile_paraphrase_invariance": {"T2"},
}

def ranks(path):
    data = json.loads(path.read_text())
    return {int(it["id"].split("-")[1]): it["rank"] for it in data["items"]}

def spearman(ra, rb):
    n = len(ra)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in ra)
    return 1 - 6 * d2 / (n * (n * n - 1))

def main() -> None:
    prompts = json.loads((W2 / "prompts.json").read_text())

    print("== Arm 1: test-retest (original vs --no-cache repeat) ==")
    retests = {}
    for axis in prompts:
        for m in ORIG_MODELS:
            orig = ranks(W2 / f"sort-{axis}-{m}.json")
            rep = ranks(REP / f"sort-{axis}-{m}-rep.json")
            retests[(axis, m)] = spearman(orig, rep)
    for m in ORIG_MODELS:
        vals = [retests[(a, m)] for a in prompts]
        print(f"  {m:10s} median {statistics.median(vals):+.3f} "
              f"range [{min(vals):+.3f}, {max(vals):+.3f}]")
    med = statistics.median(retests.values())
    power = med >= 0.80
    print(f"  ALL CELLS median {med:+.3f} -> power gate "
          f"{'PASS' if power else 'FAIL (tier claims suspended)'}")
    worst = sorted(retests.items(), key=lambda kv: kv[1])[:4]
    for (a, m), v in worst:
        print(f"    worst: {a} x {m} {v:+.3f}")

    print("\n== Arm 2: tier signature with new small models ==")
    print(f"{'axis':32s} {'model':9s} {'gap':>6s} {'decoy fr/sm':>11s} T2a T3s sig")
    tier_general = {}
    for axis, criteria in ADMITTED.items():
        Rop = ranks(W2 / f"sort-{axis}-opus46.json")
        Rso = ranks(W2 / f"sort-{axis}-gpt56sol.json")
        frfr = spearman(Rop, Rso)
        d_fr_best = min(Rop[DECOY], Rso[DECOY])
        d_fr_worst_ok = min(Rop[DECOY], Rso[DECOY]) >= 7
        sigs = 0
        for m in NEW_MODELS:
            Rs = ranks(REP / f"sort-{axis}-{m}.json")
            gap = frfr - (spearman(Rop, Rs) + spearman(Rso, Rs)) / 2
            t2a = gap >= 0.20
            t3s = d_fr_worst_ok and (d_fr_best - Rs[DECOY] >= 3)
            sig = (("T2" in criteria and t2a) or ("T3" in criteria and t3s))
            sigs += int(sig)
            print(f"{axis:32s} {m:9s} {gap:+.3f} "
                  f"{d_fr_best:>5d}/{Rs[DECOY]:<5d} {int(t2a)}   {int(t3s)}   "
                  f"{'YES' if sig else 'no'}")
        tier_general[axis] = sigs >= 2
        print(f"{axis:32s} -> {'TIER-GENERAL' if tier_general[axis] else 'MINI-SPECIFIC'} "
              f"({sigs}/3)")

    n_general = sum(tier_general.values())
    print(f"\n== Program verdict ==")
    print(f"  power gate: {'PASS' if power else 'FAIL'} (median retest {med:+.3f})")
    print(f"  tier-general axes: {n_general}/6")
    if power and n_general >= 3:
        print("  VERDICT: tier-divergent axes UPGRADED TO FINDING")
    else:
        print("  VERDICT: NOT upgraded — erratum on top of RESULTS-WAVE2.md; "
              "relabel per REPLICATION_SPEC rule 4")

if __name__ == "__main__":
    main()
