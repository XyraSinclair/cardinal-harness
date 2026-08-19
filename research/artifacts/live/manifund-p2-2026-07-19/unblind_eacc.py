#!/usr/bin/env python3
"""EA Community Choice replication unblinding — same registered analysis as
unblind.py, EA CC cohort (78 items, crowd quadratic-matching truth signal).
Written before the EA CC response was compared to ground truth."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from unblind import auc, spearman  # noqa: E402  (registered helpers)

PACK = Path(__file__).parent
ROOT = PACK.parent.parent.parent


def main():
    results = json.load(open(PACK / "eacc-response.json"))["entities"]
    gt = {}
    for row in csv.DictReader(open(ROOT / "data/manifund/ground_truth.csv")):
        if "eacc" in row["cohorts"]:
            gt[row["slug"]] = {
                "funded": "funded" in row["cohorts"],
                "raised": float(row["raised"] or 0),
            }
    scored = [r for r in results if r["id"] in gt]
    print(f"matched {len(scored)} of {len(results)} scored items to EA CC ground truth")
    combined = [r["u_mean"] for r in scored]
    funded = [gt[r["id"]]["funded"] for r in scored]
    raised = [gt[r["id"]]["raised"] for r in scored]
    print(f"cohort: n={len(scored)}, funded={sum(funded)}")
    print(f"AUC (combined score vs funded): {auc(combined, funded):.3f}")
    print(f"Spearman (combined score vs dollars raised): {spearman(combined, raised):+.3f}")
    attr_ids = [a["id"] for a in json.load(open(PACK / "p2-eacc-4attr.json"))["attributes"]]
    for attr in attr_ids:
        vals = [r["attribute_scores"][attr]["latent_mean"] for r in scored]
        print(f"  AUC[{attr}]: {auc(vals, funded):.3f}")


if __name__ == "__main__":
    sys.exit(main())
