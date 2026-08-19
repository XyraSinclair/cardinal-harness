#!/usr/bin/env python3
"""Disk-only warranty pricing backtest for issue #50.

Inputs are checked-in SQLite caches. Outputs are deterministic JSON.
"""

import json
import math
import os
import sqlite3
from collections import defaultdict


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS_PATH = os.path.join(HERE, "results.json")

INPUTS = {
    "judge_bench_order_split": os.path.join(
        REPO, "artifacts", "live", "judge-bench-2026-07-05", "bench-cache.sqlite"
    ),
    "corpus_map_repeat_split": os.path.join(
        REPO, "artifacts", "live", "corpus-map-500-2026-07-08", "map-cache.sqlite"
    ),
}

RETEST_DIR = os.path.join(REPO, "artifacts", "live", "judge-bench-retest-2026-07-05")
EPS = 1e-12
SENSITIVITY_SIGMA_B_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0]


def mean(values):
    return sum(values) / len(values) if values else None


def sign(value):
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def phi(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def finite(value):
    return value is not None and math.isfinite(value)


def sorted_pair(a, b):
    return (a, b) if a <= b else (b, a)


def group_key_to_string(key):
    return "|".join(key)


def pair_key_to_string(key):
    model, attribute, lo, hi = key
    return "|".join([model, attribute, lo, hi])


def connect_readonly(path):
    uri = "file:" + path + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def load_cache(path):
    stats = {
        "path": os.path.relpath(path, REPO),
        "rows_total": 0,
        "usable_draws": 0,
        "exclusions": defaultdict(int),
    }
    draws = []
    conn = connect_readonly(path)
    try:
        rows = conn.execute(
            """
            select
              key_hash, model, prompt_template_slug, template_hash,
              attribute_id, attribute_prompt_hash,
              entity_a_id, entity_b_id, higher_ranked, ratio, confidence,
              refused, created_at, updated_at, hit_count
            from pairwise_cache
            order by created_at, key_hash
            """
        )
        for row in rows:
            stats["rows_total"] += 1
            if row["refused"]:
                stats["exclusions"]["refused"] += 1
                continue
            if row["higher_ranked"] is None:
                stats["exclusions"]["missing_higher_ranked"] += 1
                continue
            higher = str(row["higher_ranked"]).upper()
            if higher not in ("A", "B"):
                stats["exclusions"]["invalid_higher_ranked"] += 1
                continue
            if row["ratio"] is None:
                stats["exclusions"]["missing_ratio"] += 1
                continue
            ratio = float(row["ratio"])
            if not finite(ratio) or ratio < 1.0 or ratio > 26.0:
                stats["exclusions"]["invalid_ratio"] += 1
                continue

            a = str(row["entity_a_id"])
            b = str(row["entity_b_id"])
            lo, hi = sorted_pair(a, b)
            first_in_slot_a = a == lo
            toward_slot_a = 1.0 if higher == "A" else -1.0
            toward_first = toward_slot_a if first_in_slot_a else -toward_slot_a
            signed_log_ratio = toward_first * math.log(max(1.0, ratio))
            order = "canonical_first_in_slot_a" if first_in_slot_a else "canonical_first_in_slot_b"

            draws.append(
                {
                    "key_hash": str(row["key_hash"]),
                    "model": str(row["model"]),
                    "prompt_template_slug": str(row["prompt_template_slug"]),
                    "template_hash": str(row["template_hash"]),
                    "attribute_id": str(row["attribute_id"]),
                    "attribute_prompt_hash": str(row["attribute_prompt_hash"]),
                    "entity_a_id": a,
                    "entity_b_id": b,
                    "lo": lo,
                    "hi": hi,
                    "higher_ranked": higher,
                    "ratio": ratio,
                    "confidence": None if row["confidence"] is None else float(row["confidence"]),
                    "created_at": int(row["created_at"]),
                    "updated_at": int(row["updated_at"]),
                    "hit_count": int(row["hit_count"]),
                    "signed_log_ratio": signed_log_ratio,
                    "order": order,
                }
            )
            stats["usable_draws"] += 1
    finally:
        conn.close()
    stats["exclusions"] = dict(sorted(stats["exclusions"].items()))
    return draws, stats


def group_draws(draws):
    groups = defaultdict(list)
    for draw in draws:
        key = (draw["model"], draw["attribute_id"], draw["lo"], draw["hi"])
        groups[key].append(draw)
    for values in groups.values():
        values.sort(key=lambda d: (d["created_at"], d["key_hash"]))
    return groups


def probe_groups(groups):
    histogram = defaultdict(int)
    pairs_ge_4 = 0
    pairs_both_orders = 0
    max_draws = 0
    prompt_hash_histogram = defaultdict(int)
    for draws in groups.values():
        n = len(draws)
        orders = len({d["order"] for d in draws})
        prompt_hashes = len({d["attribute_prompt_hash"] for d in draws})
        histogram[(n, orders)] += 1
        prompt_hash_histogram[prompt_hashes] += 1
        max_draws = max(max_draws, n)
        if n >= 4:
            pairs_ge_4 += 1
        if orders == 2:
            pairs_both_orders += 1
    return {
        "unordered_pairs": len(groups),
        "pairs_with_both_orders": pairs_both_orders,
        "pairs_with_at_least_4_draws": pairs_ge_4,
        "max_draws_per_pair": max_draws,
        "multiplicity_histogram": [
            {"draws": k[0], "orders": k[1], "pairs": v}
            for k, v in sorted(histogram.items())
        ],
        "attribute_prompt_hashes_per_pair": [
            {"hashes": k, "pairs": v} for k, v in sorted(prompt_hash_histogram.items())
        ],
    }


def build_warranty(
    path_name,
    pair_key,
    pricing_draws,
    claim_draws,
    all_draws,
    split_kind,
):
    price_values = [d["signed_log_ratio"] for d in pricing_draws]
    claim_values = [d["signed_log_ratio"] for d in claim_draws]
    delta_hat = mean(price_values)
    claim_mean = mean(claim_values)
    model, attribute, lo, hi = pair_key
    return {
        "path": path_name,
        "split_kind": split_kind,
        "group_key": (path_name, model, attribute),
        "pair_key": pair_key,
        "model": model,
        "attribute_id": attribute,
        "lo": lo,
        "hi": hi,
        "delta_hat": delta_hat,
        "claim_mean": claim_mean,
        "k_price": len(pricing_draws),
        "k_claim": len(claim_draws),
        "k_all": len(all_draws),
        "pricing_orders": dict(sorted(count_by(pricing_draws, "order").items())),
        "claim_orders": dict(sorted(count_by(claim_draws, "order").items())),
        "attribute_prompt_hashes_all": sorted({d["attribute_prompt_hash"] for d in all_draws}),
        "created_at_min": min(d["created_at"] for d in all_draws),
        "created_at_max": max(d["created_at"] for d in all_draws),
    }


def count_by(rows, field):
    out = defaultdict(int)
    for row in rows:
        out[row[field]] += 1
    return out


def select_order_split(path_name, groups):
    exclusions = defaultdict(int)
    warranties = []
    variance_pairs = defaultdict(dict)
    for pair_key, draws in groups.items():
        by_order = defaultdict(list)
        for draw in draws:
            by_order[draw["order"]].append(draw)
        pricing = by_order.get("canonical_first_in_slot_a", [])
        claim = by_order.get("canonical_first_in_slot_b", [])
        if not pricing or not claim:
            exclusions["missing_counter_order"] += 1
            continue
        group_key = (path_name, pair_key[0], pair_key[1])
        variance_pairs[group_key][(pair_key[2], pair_key[3])] = draws
        warranty = build_warranty(path_name, pair_key, pricing, claim, draws, "order_counterbalance")
        if sign(warranty["delta_hat"]) == 0:
            exclusions["zero_pricing_mean"] += 1
            continue
        warranties.append(warranty)
    return warranties, variance_pairs, dict(sorted(exclusions.items()))


def select_repeat_split(path_name, groups):
    exclusions = defaultdict(int)
    warranties = []
    variance_pairs = defaultdict(dict)
    for pair_key, draws in groups.items():
        if len(draws) < 4:
            exclusions["fewer_than_4_draws"] += 1
            continue
        midpoint = len(draws) // 2
        pricing = draws[:midpoint]
        claim = draws[midpoint:]
        if not pricing or not claim:
            exclusions["empty_split_side"] += 1
            continue
        group_key = (path_name, pair_key[0], pair_key[1])
        variance_pairs[group_key][(pair_key[2], pair_key[3])] = draws
        warranty = build_warranty(path_name, pair_key, pricing, claim, draws, "created_at_half_split")
        if sign(warranty["delta_hat"]) == 0:
            exclusions["zero_pricing_mean"] += 1
            continue
        warranties.append(warranty)
    return warranties, variance_pairs, dict(sorted(exclusions.items()))


def union_find_components(vertices, edges):
    parent = {v: v for v in vertices}

    def find(v):
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != v:
            nxt = parent[v]
            parent[v] = root
            v = nxt
        return root

    for i, j, _mean_value, _k in edges:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj

    components = defaultdict(list)
    for vertex in vertices:
        components[find(vertex)].append(vertex)
    return [sorted(values) for values in components.values()]


def solve_linear_system(matrix, vector):
    n = len(vector)
    if n == 0:
        return []
    a = [row[:] for row in matrix]
    b = vector[:]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-14:
            return None
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]
        pivot_value = a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] / pivot_value
            if factor == 0.0:
                continue
            a[row][col] = 0.0
            for inner in range(col + 1, n):
                a[row][inner] -= factor * a[col][inner]
            b[row] -= factor * b[col]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        rhs = b[row]
        for col in range(row + 1, n):
            rhs -= a[row][col] * x[col]
        if abs(a[row][row]) < 1e-14:
            return None
        x[row] = rhs / a[row][row]
    return x


def weighted_graph_residuals(edges, sigma_w2):
    vertices = sorted({v for edge in edges for v in edge[:2]})
    components = union_find_components(vertices, edges)
    refs = {component[0] for component in components}
    index = {}
    for vertex in vertices:
        if vertex not in refs:
            index[vertex] = len(index)

    n = len(index)
    normal = [[0.0 for _ in range(n)] for _ in range(n)]
    rhs = [0.0 for _ in range(n)]
    weights = []
    for i, j, edge_mean, k in edges:
        weight = k / sigma_w2
        weights.append(weight)
        row_terms = []
        if i in index:
            row_terms.append((index[i], 1.0))
        if j in index:
            row_terms.append((index[j], -1.0))
        for a_idx, a_value in row_terms:
            rhs[a_idx] += weight * edge_mean * a_value
            for b_idx, b_value in row_terms:
                normal[a_idx][b_idx] += weight * a_value * b_value

    solution = solve_linear_system(normal, rhs)
    if solution is None:
        return None

    scores = {vertex: 0.0 for vertex in refs}
    for vertex, idx in index.items():
        scores[vertex] = solution[idx]

    residuals = []
    for i, j, edge_mean, _k in edges:
        residuals.append(edge_mean - (scores[i] - scores[j]))
    return scores, residuals, weights, len(components)


def estimate_variance_components(variance_pairs):
    results = {}
    for group_key, pair_draws in sorted(variance_pairs.items()):
        ss = 0.0
        dof = 0
        edges = []
        vertices = set()
        for (lo, hi), draws in sorted(pair_draws.items()):
            values = [d["signed_log_ratio"] for d in draws]
            edge_mean = mean(values)
            if len(values) >= 2:
                ss += sum((value - edge_mean) ** 2 for value in values)
                dof += len(values) - 1
            edges.append((lo, hi, edge_mean, len(values)))
            vertices.add(lo)
            vertices.add(hi)

        raw_sigma_w2 = None if dof == 0 else ss / dof
        sigma_w2 = None if raw_sigma_w2 is None else max(raw_sigma_w2, EPS)
        component_result = {
            "group_key": group_key_to_string(group_key),
            "pairs_for_variance": len(pair_draws),
            "draws_for_variance": sum(len(draws) for draws in pair_draws.values()),
            "within_pair_dof": dof,
            "sigma_w2_raw": raw_sigma_w2,
            "sigma_w2": sigma_w2,
            "sigma_w": None if sigma_w2 is None else math.sqrt(sigma_w2),
            "sigma_b2": None,
            "sigma_b": None,
            "sigma_b_identifiable": False,
            "q_statistic": None,
            "degrees_of_freedom": None,
            "c": None,
            "dl_reason": None,
        }
        if sigma_w2 is None:
            component_result["dl_reason"] = "sigma_w_unidentified"
            results[group_key] = component_result
            continue

        residual_result = weighted_graph_residuals(edges, sigma_w2)
        if residual_result is None:
            component_result["dl_reason"] = "singular_graph_solve"
            results[group_key] = component_result
            continue
        _scores, residuals, weights, component_count = residual_result
        q_statistic = sum(w * r * r for w, r in zip(weights, residuals))
        df = len(edges) + component_count - len(vertices)
        sum_w = sum(weights)
        sum_w2 = sum(w * w for w in weights)
        c = sum_w - (sum_w2 / max(sum_w, EPS))
        component_result["q_statistic"] = q_statistic
        component_result["degrees_of_freedom"] = df
        component_result["c"] = c
        if df <= 0:
            component_result["dl_reason"] = "zero_cycle_degrees_of_freedom"
        elif c <= 0.0:
            component_result["dl_reason"] = "nonpositive_c"
        else:
            sigma_b2 = max(0.0, (q_statistic - df) / c)
            component_result["sigma_b2"] = sigma_b2
            component_result["sigma_b"] = math.sqrt(sigma_b2)
            component_result["sigma_b_identifiable"] = True
            component_result["dl_reason"] = "identified"
        results[group_key] = component_result
    return results


def price_probability(delta_hat, sigma_w2, sigma_b2, k_price, k_claim):
    se_delta2 = sigma_w2 / k_price
    claim_var = sigma_w2 / k_claim
    denom2 = sigma_b2 + claim_var + se_delta2
    if denom2 <= 0.0:
        return 0.0
    return phi(-abs(delta_hat) / math.sqrt(denom2))


def attach_prices(warranties, variance_components):
    records = []
    exclusions = defaultdict(int)
    for warranty in warranties:
        group_key = warranty["group_key"]
        components = variance_components.get(group_key)
        if components is None or components["sigma_w2"] is None:
            exclusions["missing_variance_components"] += 1
            continue
        sigma_w2 = components["sigma_w2"]
        sigma_b2 = components["sigma_b2"] if components["sigma_b_identifiable"] else 0.0
        probability = price_probability(
            warranty["delta_hat"],
            sigma_w2,
            sigma_b2,
            warranty["k_price"],
            warranty["k_claim"],
        )
        claim_se = math.sqrt(sigma_w2 / warranty["k_claim"])
        raw_flip = sign(warranty["delta_hat"]) * sign(warranty["claim_mean"]) < 0
        certified_flip = raw_flip and abs(warranty["claim_mean"]) > 2.0 * claim_se
        records.append(
            {
                **warranty,
                "predicted_p_reversal": probability,
                "premium": probability,
                "sigma_w": math.sqrt(sigma_w2),
                "sigma_b": math.sqrt(sigma_b2),
                "sigma_b_source": "der_simonian_laird"
                if components["sigma_b_identifiable"]
                else "unidentifiable_zero_floor",
                "se_delta": math.sqrt(sigma_w2 / warranty["k_price"]),
                "se_claim": claim_se,
                "raw_flip": raw_flip,
                "certified_flip": certified_flip,
            }
        )
    return records, dict(sorted(exclusions.items()))


def reliability_bins(records, event_field):
    if not records:
        return []
    ordered = sorted(records, key=lambda r: (r["predicted_p_reversal"], pair_key_to_string(r["pair_key"])))
    n = len(ordered)
    bins = []
    for idx in range(10):
        start = (idx * n) // 10
        end = ((idx + 1) * n) // 10
        if end <= start:
            continue
        subset = ordered[start:end]
        observed = sum(1 for row in subset if row[event_field])
        bins.append(
            {
                "decile": idx + 1,
                "n": len(subset),
                "p_min": subset[0]["predicted_p_reversal"],
                "p_max": subset[-1]["predicted_p_reversal"],
                "predicted_mean": mean([row["predicted_p_reversal"] for row in subset]),
                "observed_frequency": observed / len(subset),
                "events": observed,
            }
        )
    return bins


def reliability_slope(bins):
    if len(bins) < 2:
        return None, None
    total_weight = sum(bin_row["n"] for bin_row in bins)
    x_bar = sum(bin_row["n"] * bin_row["predicted_mean"] for bin_row in bins) / total_weight
    y_bar = sum(bin_row["n"] * bin_row["observed_frequency"] for bin_row in bins) / total_weight
    denom = sum(bin_row["n"] * (bin_row["predicted_mean"] - x_bar) ** 2 for bin_row in bins)
    if denom <= 0.0:
        return None, None
    slope = (
        sum(
            bin_row["n"]
            * (bin_row["predicted_mean"] - x_bar)
            * (bin_row["observed_frequency"] - y_bar)
            for bin_row in bins
        )
        / denom
    )
    intercept = y_bar - slope * x_bar
    return slope, intercept


def calibration(records, event_field):
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "events": 0,
            "base_rate": None,
            "brier": None,
            "base_rate_brier": None,
            "brier_improvement": None,
            "reliability_slope": None,
            "reliability_intercept": None,
            "loss": True,
            "loss_reasons": ["no_records"],
            "deciles": [],
        }
    ys = [1.0 if row[event_field] else 0.0 for row in records]
    ps = [row["predicted_p_reversal"] for row in records]
    base_rate = mean(ys)
    brier = mean([(p - y) ** 2 for p, y in zip(ps, ys)])
    base_rate_brier = mean([(base_rate - y) ** 2 for y in ys])
    bins = reliability_bins(records, event_field)
    slope, intercept = reliability_slope(bins)
    loss_reasons = []
    if brier >= base_rate_brier:
        loss_reasons.append("brier_not_better_than_base_rate")
    if slope is None:
        loss_reasons.append("reliability_slope_unidentifiable")
    elif slope < 0.5 or slope > 1.5:
        loss_reasons.append("reliability_slope_outside_[0.5,1.5]")
    return {
        "n": n,
        "events": int(sum(ys)),
        "base_rate": base_rate,
        "brier": brier,
        "base_rate_brier": base_rate_brier,
        "brier_improvement": base_rate_brier - brier,
        "reliability_slope": slope,
        "reliability_intercept": intercept,
        "loss": bool(loss_reasons),
        "loss_reasons": loss_reasons,
        "deciles": bins,
    }


def aggregate_records(records):
    premium = sum(row["premium"] for row in records)
    raw_claims = sum(1 for row in records if row["raw_flip"])
    certified_claims = sum(1 for row in records if row["certified_flip"])
    return {
        "pairs_priced": len(records),
        "pricing_draws": sum(row["k_price"] for row in records),
        "claim_draws": sum(row["k_claim"] for row in records),
        "all_draws_on_priced_pairs": sum(row["k_all"] for row in records),
        "collected_premium_unit_payout": premium,
        "raw_claims": raw_claims,
        "certified_claims": certified_claims,
        "raw_loss_ratio": None if premium <= 0.0 else raw_claims / premium,
        "certified_loss_ratio": None if premium <= 0.0 else certified_claims / premium,
        "raw": calibration(records, "raw_flip"),
        "certified": calibration(records, "certified_flip"),
    }


def aggregate_with_sensitivity(records, variance_components):
    unidentifiable = [
        row
        for row in records
        if not variance_components[row["group_key"]]["sigma_b_identifiable"]
    ]
    if not unidentifiable:
        return []
    out = []
    for multiplier in SENSITIVITY_SIGMA_B_MULTIPLIERS:
        premium = 0.0
        for row in records:
            components = variance_components[row["group_key"]]
            sigma_w2 = components["sigma_w2"]
            if components["sigma_b_identifiable"]:
                sigma_b2 = components["sigma_b2"]
            else:
                sigma_b = multiplier * math.sqrt(sigma_w2)
                sigma_b2 = sigma_b * sigma_b
            premium += price_probability(
                row["delta_hat"], sigma_w2, sigma_b2, row["k_price"], row["k_claim"]
            )
        raw_claims = sum(1 for row in records if row["raw_flip"])
        certified_claims = sum(1 for row in records if row["certified_flip"])
        out.append(
            {
                "unidentifiable_sigma_b_multiplier": multiplier,
                "collected_premium_unit_payout": premium,
                "raw_loss_ratio": None if premium <= 0.0 else raw_claims / premium,
                "certified_loss_ratio": None if premium <= 0.0 else certified_claims / premium,
            }
        )
    return out


def summarize_path(path_name, noise_class, cache_stats, probe, warranties, variance_pairs, selection_exclusions):
    variance_components = estimate_variance_components(variance_pairs)
    records, pricing_exclusions = attach_prices(warranties, variance_components)
    aggregate = aggregate_records(records)
    aggregate["sensitivity_when_sigma_b_unidentifiable"] = aggregate_with_sensitivity(
        records, variance_components
    )
    aggregate["primary_certified_gate_loses"] = aggregate["certified"]["loss"]
    serializable_records = []
    for row in records:
        serializable = dict(row)
        serializable["group_key"] = group_key_to_string(row["group_key"])
        serializable["pair_key"] = pair_key_to_string(row["pair_key"])
        serializable_records.append(serializable)
    return {
        "path_name": path_name,
        "noise_class_priced": noise_class,
        "input": cache_stats,
        "probe": probe,
        "selection_exclusions": selection_exclusions,
        "pricing_exclusions": pricing_exclusions,
        "variance_components": [
            variance_components[key] for key in sorted(variance_components)
        ],
        "aggregate": aggregate,
        "pairs": serializable_records,
    }


def retest_pair_stream_status():
    files = []
    if os.path.isdir(RETEST_DIR):
        for name in sorted(os.listdir(RETEST_DIR)):
            path = os.path.join(RETEST_DIR, name)
            if os.path.isfile(path):
                files.append(os.path.relpath(path, REPO))
    sqlite_files = [name for name in files if name.endswith(".sqlite")]
    return {
        "usable": False,
        "path": os.path.relpath(RETEST_DIR, REPO),
        "files": files,
        "sqlite_files": sqlite_files,
        "reason": "retest pack has aggregate reports but no pair-level SQLite cache or per-pair claim stream",
        "estimated_cost_to_create_usd": 0.46,
    }


def main():
    loaded = {}
    for path_name, path in INPUTS.items():
        draws, stats = load_cache(path)
        groups = group_draws(draws)
        loaded[path_name] = {
            "draws": draws,
            "stats": stats,
            "groups": groups,
            "probe": probe_groups(groups),
        }

    order_warranties, order_variance_pairs, order_exclusions = select_order_split(
        "judge_bench_order_split", loaded["judge_bench_order_split"]["groups"]
    )
    repeat_warranties, repeat_variance_pairs, repeat_exclusions = select_repeat_split(
        "corpus_map_repeat_split", loaded["corpus_map_repeat_split"]["groups"]
    )

    paths = {
        "corpus_map_repeat_split": summarize_path(
            "corpus_map_repeat_split",
            "attribute-prompt/repeat-draw noise on k>=4 map-cache pairs; presentation order is counterbalanced inside the supported subset where present",
            loaded["corpus_map_repeat_split"]["stats"],
            loaded["corpus_map_repeat_split"]["probe"],
            repeat_warranties,
            repeat_variance_pairs,
            repeat_exclusions,
        ),
        "judge_bench_order_split": summarize_path(
            "judge_bench_order_split",
            "within-run presentation-order noise; not session noise",
            loaded["judge_bench_order_split"]["stats"],
            loaded["judge_bench_order_split"]["probe"],
            order_warranties,
            order_variance_pairs,
            order_exclusions,
        ),
    }

    selected = (
        "corpus_map_repeat_split"
        if paths["corpus_map_repeat_split"]["aggregate"]["pairs_priced"] > 0
        else "judge_bench_order_split"
        if paths["judge_bench_order_split"]["aggregate"]["pairs_priced"] > 0
        else None
    )

    results = {
        "receipt": "warranty-backtest-2026-07-09",
        "unit_payout": 1.0,
        "selected_primary_path": selected,
        "loss_criteria": {
            "brier": "loss if model Brier is not strictly lower than base-rate-only Brier",
            "reliability_slope": "loss if weighted decile reliability slope is outside [0.5, 1.5] or unidentifiable",
            "primary_gate": "certified_flip",
        },
        "ideal_issue_50_v1_to_retest_stream": retest_pair_stream_status(),
        "paths": paths,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

