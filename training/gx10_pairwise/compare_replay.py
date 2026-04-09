#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and cache-replayed rerank responses."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--k", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def top_k_ids(response: dict[str, Any], k: int) -> list[str]:
    entities = sorted(
        response["entities"],
        key=lambda entity: (
            entity["rank"] is None,
            entity["rank"] if entity["rank"] is not None else 10**9,
        ),
    )
    return [entity["id"] for entity in entities[:k]]


def rank_map(response: dict[str, Any]) -> dict[str, int]:
    return {
        entity["id"]: int(entity["rank"])
        for entity in response["entities"]
        if entity["rank"] is not None
    }


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    baseline_meta = baseline["meta"]
    candidate_meta = candidate["meta"]
    k = args.k or int(candidate_meta["k"])

    baseline_topk = top_k_ids(baseline, k)
    candidate_topk = top_k_ids(candidate, k)
    baseline_topk_set = set(baseline_topk)
    candidate_topk_set = set(candidate_topk)
    shared = baseline_topk_set & candidate_topk_set
    union = baseline_topk_set | candidate_topk_set

    baseline_ranks = rank_map(baseline)
    candidate_ranks = rank_map(candidate)
    common_ids = sorted(set(baseline_ranks) & set(candidate_ranks))
    mean_abs_rank_delta = (
        sum(abs(baseline_ranks[item_id] - candidate_ranks[item_id]) for item_id in common_ids)
        / len(common_ids)
        if common_ids
        else 0.0
    )

    baseline_attempted = int(baseline_meta["comparisons_attempted"])
    candidate_attempted = int(candidate_meta["comparisons_attempted"])
    attempted_ratio = (
        candidate_attempted / baseline_attempted if baseline_attempted > 0 else 0.0
    )

    baseline_cost = int(baseline_meta["provider_cost_nanodollars"])
    candidate_cost = int(candidate_meta["provider_cost_nanodollars"])
    cost_ratio = candidate_cost / baseline_cost if baseline_cost > 0 else None

    print(f"k={k}")
    print(
        f"topk_overlap={len(shared) / max(1, k):.3f} "
        f"jaccard={len(shared) / max(1, len(union)):.3f}"
    )
    print(
        f"top1_match={baseline_topk[:1] == candidate_topk[:1]} "
        f"baseline_top1={baseline_topk[0] if baseline_topk else '-'} "
        f"candidate_top1={candidate_topk[0] if candidate_topk else '-'}"
    )
    print(f"mean_abs_rank_delta={mean_abs_rank_delta:.3f}")
    print(
        "comparisons_attempted "
        f"{baseline_attempted} -> {candidate_attempted} "
        f"(ratio={attempted_ratio:.3f}, win_20pct={attempted_ratio <= 0.8})"
    )
    print(
        "comparisons_used "
        f"{baseline_meta['comparisons_used']} -> {candidate_meta['comparisons_used']}"
    )
    print(
        "comparisons_refused "
        f"{baseline_meta['comparisons_refused']} -> {candidate_meta['comparisons_refused']}"
    )
    print(
        "global_topk_error "
        f"{baseline_meta['global_topk_error']:.4f} -> {candidate_meta['global_topk_error']:.4f}"
    )
    if cost_ratio is not None:
        print(
            "provider_cost_nanodollars "
            f"{baseline_cost} -> {candidate_cost} "
            f"(ratio={cost_ratio:.3f}, win_25pct={cost_ratio <= 0.75})"
        )
    print(f"stop_reason {baseline_meta['stop_reason']} -> {candidate_meta['stop_reason']}")


if __name__ == "__main__":
    main()
