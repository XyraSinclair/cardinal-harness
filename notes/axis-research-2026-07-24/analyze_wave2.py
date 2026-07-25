#!/usr/bin/env python3
"""Wave-2 verdicts against the frozen thresholds in WAVE2_SPEC.md.

Primary decoy is always file item index 1 (0-based; the second line) ->
cardinal id item-0001. T1: opus<->sol rho >= 0.60. T2: fr-fr rho minus mean
fr-mini rho >= 0.20. T3: decoy rank >= 7 for both frontiers AND mini places
it >= 3 ranks higher than the frontiers' best (lowest) rank for it.
PASS = T1 and (T2 or T3); WEAK = T1 only; FAIL = not T1.
"""
import itertools
import json
from pathlib import Path

HERE = Path(__file__).parent
W2 = HERE / "wave2"
MODELS = ["opus46", "gpt56sol", "mini54"]
DECOY = 1  # 0-based index of the primary decoy in every probe file

def ranks(axis, model):
    data = json.loads((W2 / f"sort-{axis}-{model}.json").read_text())
    return {int(it["id"].split("-")[1]): it["rank"] for it in data["items"]}

def spearman(ra, rb):
    n = len(ra)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in ra)
    return 1 - 6 * d2 / (n * (n * n - 1))

def main() -> None:
    prompts = json.loads((W2 / "prompts.json").read_text())
    print(f"{'axis':32s} {'fr-fr':>6s} {'op-mi':>6s} {'so-mi':>6s} "
          f"{'gap':>6s} {'decoy fr/mi':>11s}  verdict")
    for axis in prompts:
        try:
            R = {m: ranks(axis, m) for m in MODELS}
        except FileNotFoundError:
            print(f"{axis:32s} (runs missing)")
            continue
        frfr = spearman(R["opus46"], R["gpt56sol"])
        opmi = spearman(R["opus46"], R["mini54"])
        somi = spearman(R["gpt56sol"], R["mini54"])
        gap = frfr - (opmi + somi) / 2
        d_fr = max(R["opus46"][DECOY], R["gpt56sol"][DECOY])  # worst=deepest
        d_fr_best = min(R["opus46"][DECOY], R["gpt56sol"][DECOY])
        d_mi = R["mini54"][DECOY]
        t1 = frfr >= 0.60
        t2 = gap >= 0.20
        t3 = (min(R["opus46"][DECOY], R["gpt56sol"][DECOY]) >= 7
              and d_fr_best - d_mi >= 3)
        verdict = "PASS" if (t1 and (t2 or t3)) else ("WEAK" if t1 else "FAIL")
        flags = f"T1={int(t1)} T2={int(t2)} T3={int(t3)}"
        print(f"{axis:32s} {frfr:+.3f} {opmi:+.3f} {somi:+.3f} "
              f"{gap:+.3f} {d_fr_best:>4d}/{d_mi:<4d}  {verdict} ({flags})")

    print("\nPer-axis frontier top-3 / bottom-3 (opus46):")
    for axis in prompts:
        try:
            r = ranks(axis, "opus46")
        except FileNotFoundError:
            continue
        order = sorted(r, key=lambda i: r[i])
        top = ", ".join(str(i + 1) for i in order[:3])
        bot = ", ".join(str(i + 1) for i in order[-3:])
        print(f"  {axis:32s} top: items {top}   bottom: items {bot}")

if __name__ == "__main__":
    main()
