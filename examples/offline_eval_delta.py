#!/usr/bin/env python3
"""Compare offline cardinal synthetic eval receipts against Likert receipts.

This script intentionally uses only Python's standard library. It does not call a
provider and does not require an API key. The output is a receipt aid, not a
statistical significance test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METRIC_DIRECTIONS: dict[str, bool] = {
    "topk_precision": True,
    "topk_recall": True,
    "kendall_tau_b": True,
    "coverage_95ci": True,
    "comparisons_used": False,
    "gate_precision": True,
    "gate_recall": True,
}


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_name = row.get("case_name")
            if not isinstance(case_name, str):
                raise SystemExit(f"{path}:{line_no}: missing string case_name")
            if case_name in rows:
                raise SystemExit(f"{path}:{line_no}: duplicate case_name {case_name!r}")
            rows[case_name] = row
    return rows


def metric(row: dict[str, Any], name: str) -> float | None:
    metrics = row.get("metrics", {})
    value = metrics.get(name)
    if value is None:
        value = metrics.get("rank_quality", {}).get(name)
    if value is None and name == "kendall_tau_b":
        value = metrics.get("kendall_tau")
    if value is None and name == "comparisons_used":
        value = metrics.get("ratings_used")
    if value is None:
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def winner_label(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 1e-12:
        return "tie"
    cardinal_is_better = delta > 0 if higher_is_better else delta < 0
    return "cardinal" if cardinal_is_better else "likert"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write CSV and text deltas between cardinal and Likert offline eval receipts."
    )
    parser.add_argument("--cardinal", required=True, type=Path)
    parser.add_argument("--likert", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    cardinal = load_jsonl(args.cardinal)
    likert = load_jsonl(args.likert)
    cases = sorted(set(cardinal) & set(likert))
    if not cases:
        raise SystemExit("no overlapping case_name values")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    wins = {"cardinal": 0, "likert": 0, "tie": 0}
    topk_wins = {"cardinal": 0, "likert": 0, "tie": 0}
    decisive_metric = "topk_precision"

    for case in cases:
        c_row = cardinal[case]
        l_row = likert[case]
        c_metrics = c_row.get("metrics", {})
        l_metrics = l_row.get("metrics", {})
        for name, higher_is_better in METRIC_DIRECTIONS.items():
            c_value = metric(c_row, name)
            l_value = metric(l_row, name)
            if c_value is None or l_value is None:
                continue
            delta = c_value - l_value
            winner = winner_label(delta, higher_is_better)
            wins[winner] += 1
            if name == decisive_metric:
                topk_wins[winner] += 1
            rows.append(
                {
                    "case": case,
                    "metric": name,
                    "higher_is_better": str(higher_is_better).lower(),
                    "cardinal": fmt(c_value),
                    "likert": fmt(l_value),
                    "delta_cardinal_minus_likert": fmt(delta),
                    "winner": winner,
                    "cardinal_calls": str(c_metrics.get("comparisons_used", "")),
                    "likert_calls": str(l_metrics.get("ratings_used", "")),
                    "cardinal_stop": str(c_metrics.get("stop_reason", "")),
                }
            )

    if not rows:
        raise SystemExit("overlapping cases had no comparable metrics")

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    missing_cardinal = sorted(set(likert) - set(cardinal))
    missing_likert = sorted(set(cardinal) - set(likert))
    topk_lines = [row for row in rows if row["metric"] == decisive_metric]
    with args.summary.open("w", encoding="utf-8") as handle:
        handle.write("Offline cardinal-vs-Likert synthetic comparison\n")
        handle.write(f"Cardinal receipt: {args.cardinal}\n")
        handle.write(f"Likert receipt:   {args.likert}\n")
        handle.write(f"Delta CSV:        {args.csv}\n")
        handle.write(f"Compared cases:   {len(cases)}\n")
        handle.write(f"Compared metrics: {len(rows)}\n")
        if missing_cardinal:
            handle.write(f"Likert-only cases skipped: {', '.join(missing_cardinal)}\n")
        if missing_likert:
            handle.write(f"Cardinal-only cases skipped: {', '.join(missing_likert)}\n")
        handle.write("\n")
        handle.write(
            f"Across comparable metric rows: cardinal wins {wins['cardinal']}, "
            f"Likert wins {wins['likert']}, ties {wins['tie']}.\n"
        )
        handle.write(
            f"By {decisive_metric}: cardinal wins {topk_wins['cardinal']} case(s), "
            f"Likert wins {topk_wins['likert']} case(s), ties {topk_wins['tie']} case(s).\n"
        )
        handle.write(
            "Positive deltas mean cardinal is numerically higher; use higher_is_better "
            "or winner for metrics where lower is better.\n\n"
        )
        for row in topk_lines:
            handle.write(
                f"- {row['case']}: cardinal {row['cardinal']} vs Likert {row['likert']} "
                f"(delta {row['delta_cardinal_minus_likert']}; {row['winner']})\n"
            )
        handle.write(
            "\nThis is a mechanical comparison of deterministic synthetic receipts. "
            "It is not a live-LLM benchmark and does not establish universal superiority.\n"
        )


if __name__ == "__main__":
    main()
