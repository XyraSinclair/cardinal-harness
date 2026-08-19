#!/usr/bin/env python3
"""Confidence-map calibration receipt.

Rerunnable with only the Python standard library:

    python3 artifacts/live/confidence-knob-2026-07-09/analysis.py

The analysis intentionally uses cached point judgements, not network calls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EPS = 1.0e-3
DEFAULT_GAMMA = 2.0
EPS_MIN = 1.0e-9
EPS_MAX = 1.0 - 1.0e-9
GAMMA_MIN = 0.0
GAMMA_MAX = 20.0
DEGENERATE_MASS_THRESHOLD = 0.90


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: Path


@dataclass
class Observation:
    source: str
    model: str
    template: str
    attribute: str
    entity_a: str
    entity_b: str
    confidence: float
    signed_log_ratio_toward_first: float
    first_in_slot_a: bool


@dataclass
class ResidualDraw:
    confidence: float
    residual: float


@dataclass
class FitResult:
    eps: float
    gamma: float
    sigma0: float
    nll: float
    n: int
    weighted_sse: float
    sum_log_g: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_specs(root: Path) -> List[SourceSpec]:
    return [
        SourceSpec(
            "corpus-map-500-2026-07-08",
            root / "artifacts/live/corpus-map-500-2026-07-08/map-cache.sqlite",
        ),
        SourceSpec(
            "judge-bench-2026-07-05",
            root / "artifacts/live/judge-bench-2026-07-05/bench-cache.sqlite",
        ),
    ]


def sqlite_uri(path: Path) -> str:
    # immutable=1 keeps untracked WAL/SHM sidecars out of this committed-data
    # receipt. These paths contain no URI-special characters in this repo.
    return f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def confidence_weight(confidence: float, eps: float, gamma: float) -> float:
    c = clamp01(confidence)
    if gamma == 0.0:
        c_to_gamma = 1.0
    else:
        c_to_gamma = c**gamma
    return eps + (1.0 - eps) * c_to_gamma


def confidence_key(confidence: float) -> str:
    return f"{confidence:.12g}"


def parse_higher(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in ("A", "B"):
        return text
    return None


def signed_log_ratio_toward_first(
    entity_a: str,
    entity_b: str,
    higher_ranked: str,
    ratio: float,
) -> Tuple[float, bool]:
    first = min(entity_a, entity_b)
    first_in_slot_a = entity_a == first
    toward_slot_a = 1.0 if higher_ranked == "A" else -1.0
    toward_first = toward_slot_a if first_in_slot_a else -toward_slot_a
    return toward_first * math.log(max(1.0, ratio)), first_in_slot_a


def load_source(spec: SourceSpec) -> Tuple[List[Observation], Dict[str, object]]:
    observations: List[Observation] = []
    summary: Dict[str, object] = {
        "source": spec.label,
        "path": str(spec.path),
        "total_rows": 0,
        "refused_rows": 0,
        "malformed_nonrefused_rows": 0,
        "confidence_clamped_rows": 0,
    }
    query = """
        SELECT
            model,
            prompt_template_slug,
            attribute_id,
            entity_a_id,
            entity_b_id,
            higher_ranked,
            ratio,
            confidence,
            refused
        FROM pairwise_cache
    """
    with sqlite3.connect(sqlite_uri(spec.path), uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    summary["total_rows"] = len(rows)
    for row in rows:
        if int(row["refused"] or 0) != 0:
            summary["refused_rows"] += 1
            continue
        higher = parse_higher(row["higher_ranked"])
        ratio = row["ratio"]
        confidence = row["confidence"]
        entity_a = row["entity_a_id"]
        entity_b = row["entity_b_id"]
        if (
            higher is None
            or ratio is None
            or confidence is None
            or entity_a is None
            or entity_b is None
        ):
            summary["malformed_nonrefused_rows"] += 1
            continue
        try:
            ratio_f = float(ratio)
            confidence_f = float(confidence)
        except (TypeError, ValueError):
            summary["malformed_nonrefused_rows"] += 1
            continue
        if not (
            math.isfinite(ratio_f)
            and math.isfinite(confidence_f)
            and ratio_f > 0.0
        ):
            summary["malformed_nonrefused_rows"] += 1
            continue
        clamped_confidence = clamp01(confidence_f)
        if clamped_confidence != confidence_f:
            summary["confidence_clamped_rows"] += 1
        signed, first_in_slot_a = signed_log_ratio_toward_first(
            str(entity_a), str(entity_b), higher, ratio_f
        )
        observations.append(
            Observation(
                source=spec.label,
                model=str(row["model"]),
                template=str(row["prompt_template_slug"]),
                attribute=str(row["attribute_id"]),
                entity_a=str(entity_a),
                entity_b=str(entity_b),
                confidence=clamped_confidence,
                signed_log_ratio_toward_first=signed,
                first_in_slot_a=first_in_slot_a,
            )
        )
    summary["valid_observation_rows"] = len(observations)
    return observations, summary


def group_key(obs: Observation) -> Tuple[str, str, str, str, str, str]:
    first = min(obs.entity_a, obs.entity_b)
    second = max(obs.entity_a, obs.entity_b)
    return (obs.source, obs.model, obs.template, obs.attribute, first, second)


def histogram_key(obs: Observation) -> Tuple[str, str, str]:
    return (obs.source, obs.model, obs.template)


def bin_index(confidence: float) -> int:
    c = clamp01(confidence)
    if c >= 1.0:
        return 9
    idx = int(c * 10.0)
    return max(0, min(9, idx))


def bin_label(index: int) -> str:
    lower = index / 10.0
    upper = (index + 1) / 10.0
    if index == 9:
        return "[0.9,1.0]"
    return f"[{lower:.1f},{upper:.1f})"


def decile_histogram(confidences: Iterable[float]) -> Dict[str, int]:
    counts = Counter(bin_index(c) for c in confidences)
    return {bin_label(i): counts.get(i, 0) for i in range(10)}


def confidence_histograms(
    observations: Sequence[Observation],
    source_summaries: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    refused_by_key: Counter = Counter()
    malformed_by_source: Dict[str, int] = {}
    for summary in source_summaries:
        malformed_by_source[str(summary["source"])] = int(
            summary["malformed_nonrefused_rows"]
        )

    # Refusal rows do not always have confidence, but they do have model/template.
    # Count them separately by querying only the refusal dimensions.
    root = repo_root()
    source_by_label = {spec.label: spec for spec in source_specs(root)}
    for source, spec in source_by_label.items():
        with sqlite3.connect(sqlite_uri(spec.path), uri=True) as conn:
            for model, template, count in conn.execute(
                """
                SELECT model, prompt_template_slug, COUNT(*)
                FROM pairwise_cache
                WHERE refused != 0
                GROUP BY model, prompt_template_slug
                """
            ):
                refused_by_key[(source, str(model), str(template))] += int(count)

    confidence_by_key: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for obs in observations:
        confidence_by_key[histogram_key(obs)].append(obs.confidence)

    result: List[Dict[str, object]] = []
    for key in sorted(confidence_by_key):
        confidences = confidence_by_key[key]
        exact = Counter(confidence_key(c) for c in confidences)
        valid_n = len(confidences)
        top_value, top_count = exact.most_common(1)[0]
        result.append(
            {
                "source": key[0],
                "model": key[1],
                "prompt_template_slug": key[2],
                "valid_observations": valid_n,
                "refused_rows": refused_by_key.get(key, 0),
                "top_exact_confidence": top_value,
                "top_exact_confidence_count": top_count,
                "top_exact_confidence_mass": top_count / valid_n if valid_n else None,
                "degenerate_gt_90pct": (top_count / valid_n) > DEGENERATE_MASS_THRESHOLD
                if valid_n
                else False,
                "decile_histogram": decile_histogram(confidences),
                "exact_histogram": {
                    value: count for value, count in sorted(exact.items(), key=lambda item: float(item[0]))
                },
            }
        )
    return result


def residual_groups(
    observations: Sequence[Observation],
) -> Tuple[List[List[ResidualDraw]], Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Observation]] = defaultdict(list)
    for obs in observations:
        grouped[group_key(obs)].append(obs)

    residual_group_list: List[List[ResidualDraw]] = []
    singleton_groups = 0
    singleton_draws = 0
    both_order_groups = 0
    single_order_groups = 0
    group_size_counts: Counter = Counter()
    residual_draws = 0
    for _key, draws in grouped.items():
        group_size_counts[len(draws)] += 1
        if len(draws) < 2:
            singleton_groups += 1
            singleton_draws += len(draws)
            continue
        orientations = {draw.first_in_slot_a for draw in draws}
        if len(orientations) == 2:
            both_order_groups += 1
        else:
            single_order_groups += 1
        mean_y = sum(draw.signed_log_ratio_toward_first for draw in draws) / len(draws)
        residual_group = [
            ResidualDraw(
                confidence=draw.confidence,
                residual=draw.signed_log_ratio_toward_first - mean_y,
            )
            for draw in draws
        ]
        residual_draws += len(residual_group)
        residual_group_list.append(residual_group)

    summary: Dict[str, object] = {
        "valid_pair_groups_total": len(grouped),
        "residual_pair_groups_n_ge_2": len(residual_group_list),
        "residual_draws": residual_draws,
        "residual_degrees_of_freedom_pair_mean_removed": residual_draws
        - len(residual_group_list),
        "singleton_groups_excluded": singleton_groups,
        "singleton_draws_excluded": singleton_draws,
        "residual_groups_with_both_presentation_orders": both_order_groups,
        "residual_groups_with_single_presentation_order": single_order_groups,
        "group_size_histogram": {
            str(size): count for size, count in sorted(group_size_counts.items())
        },
    }
    return residual_group_list, summary


def flatten_residual_groups(groups: Sequence[Sequence[ResidualDraw]]) -> List[ResidualDraw]:
    return [draw for group in groups for draw in group]


def aggregate_residuals(draws: Sequence[ResidualDraw]) -> List[Tuple[float, int, float]]:
    by_confidence: Dict[float, List[float]] = defaultdict(lambda: [0, 0.0])
    for draw in draws:
        entry = by_confidence[draw.confidence]
        entry[0] += 1
        entry[1] += draw.residual * draw.residual
    return [
        (confidence, int(values[0]), float(values[1]))
        for confidence, values in sorted(by_confidence.items())
    ]


def fit_stats_for_params(
    aggregate: Sequence[Tuple[float, int, float]],
    eps: float,
    gamma: float,
) -> Tuple[float, float, float, int, float]:
    n = sum(count for _confidence, count, _sumsq in aggregate)
    if n <= 0:
        return math.inf, math.nan, math.nan, 0, math.nan
    weighted_sse = 0.0
    sum_log_g = 0.0
    eps = max(EPS_MIN, min(EPS_MAX, eps))
    gamma = max(GAMMA_MIN, min(GAMMA_MAX, gamma))
    for confidence, count, sumsq in aggregate:
        g = confidence_weight(confidence, eps, gamma)
        if not (g > 0.0 and math.isfinite(g)):
            return math.inf, math.nan, math.nan, n, math.nan
        weighted_sse += g * sumsq
        sum_log_g += count * math.log(g)
    if not (weighted_sse > 0.0 and math.isfinite(weighted_sse)):
        return math.inf, math.nan, math.nan, n, weighted_sse
    sigma2 = weighted_sse / n
    nll = 0.5 * (n * (math.log(2.0 * math.pi * sigma2) + 1.0) - sum_log_g)
    return nll, math.sqrt(sigma2), sum_log_g, n, weighted_sse


def logit(p: float) -> float:
    p = max(EPS_MIN, min(EPS_MAX, p))
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_confidence_curve(
    aggregate: Sequence[Tuple[float, int, float]]
) -> FitResult:
    eps_grid = [
        EPS_MIN,
        1.0e-7,
        1.0e-6,
        1.0e-5,
        1.0e-4,
        DEFAULT_EPS,
        3.0e-3,
        1.0e-2,
        3.0e-2,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.99,
        EPS_MAX,
    ]
    gamma_grid = [
        0.0,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        DEFAULT_GAMMA,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        GAMMA_MAX,
    ]

    best_eps = DEFAULT_EPS
    best_gamma = DEFAULT_GAMMA
    best_nll = math.inf
    best_sigma0 = math.nan
    best_sum_log_g = math.nan
    best_n = 0
    best_weighted_sse = math.nan

    def consider(eps: float, gamma: float) -> bool:
        nonlocal best_eps, best_gamma, best_nll, best_sigma0
        nonlocal best_sum_log_g, best_n, best_weighted_sse
        eps = max(EPS_MIN, min(EPS_MAX, eps))
        gamma = max(GAMMA_MIN, min(GAMMA_MAX, gamma))
        nll, sigma0, sum_log_g, n, weighted_sse = fit_stats_for_params(
            aggregate, eps, gamma
        )
        if nll < best_nll - 1.0e-10:
            best_eps = eps
            best_gamma = gamma
            best_nll = nll
            best_sigma0 = sigma0
            best_sum_log_g = sum_log_g
            best_n = n
            best_weighted_sse = weighted_sse
            return True
        return False

    for eps in eps_grid:
        for gamma in gamma_grid:
            consider(eps, gamma)

    x = logit(best_eps)
    gamma = best_gamma
    step_x = 2.0
    step_gamma = 2.0
    while step_x > 1.0e-5 or step_gamma > 1.0e-5:
        improved = False
        candidates = []
        for dx in (-step_x, 0.0, step_x):
            for dg in (-step_gamma, 0.0, step_gamma):
                if dx == 0.0 and dg == 0.0:
                    continue
                candidates.append((x + dx, gamma + dg))
        candidates.append((x, 0.0))
        candidates.append((x, GAMMA_MAX))
        for cand_x, cand_gamma in candidates:
            cand_eps = inv_logit(cand_x)
            cand_gamma = max(GAMMA_MIN, min(GAMMA_MAX, cand_gamma))
            if consider(cand_eps, cand_gamma):
                x = logit(best_eps)
                gamma = best_gamma
                improved = True
                break
        if not improved:
            step_x *= 0.5
            step_gamma *= 0.5

    return FitResult(
        eps=best_eps,
        gamma=best_gamma,
        sigma0=best_sigma0,
        nll=best_nll,
        n=best_n,
        weighted_sse=best_weighted_sse,
        sum_log_g=best_sum_log_g,
    )


def fit_fixed(
    aggregate: Sequence[Tuple[float, int, float]], eps: float, gamma: float
) -> FitResult:
    nll, sigma0, sum_log_g, n, weighted_sse = fit_stats_for_params(
        aggregate, eps, gamma
    )
    return FitResult(
        eps=eps,
        gamma=gamma,
        sigma0=sigma0,
        nll=nll,
        n=n,
        weighted_sse=weighted_sse,
        sum_log_g=sum_log_g,
    )


def percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(sorted_values[lo])
    frac = position - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def percentile_interval(values: Sequence[float]) -> Dict[str, Optional[float]]:
    finite = sorted(v for v in values if math.isfinite(v))
    return {
        "p2_5": percentile(finite, 0.025),
        "p50": percentile(finite, 0.5),
        "p97_5": percentile(finite, 0.975),
    }


def bootstrap_fits(
    groups: Sequence[Sequence[ResidualDraw]], reps: int, seed: int
) -> Dict[str, object]:
    rng = random.Random(seed)
    fit_values: Dict[str, List[float]] = {
        "eps": [],
        "gamma": [],
        "sigma0": [],
        "nll": [],
        "lrt_stat_vs_flat": [],
    }
    if not groups or reps <= 0:
        return {"reps": 0, "seed": seed, "intervals": {}}

    group_count = len(groups)
    for _ in range(reps):
        sampled_draws: List[ResidualDraw] = []
        for _idx in range(group_count):
            sampled_draws.extend(groups[rng.randrange(group_count)])
        aggregate = aggregate_residuals(sampled_draws)
        fitted = fit_confidence_curve(aggregate)
        flat = fit_fixed(aggregate, 1.0e-9, 0.0)
        lrt = max(0.0, 2.0 * (flat.nll - fitted.nll))
        fit_values["eps"].append(fitted.eps)
        fit_values["gamma"].append(fitted.gamma)
        fit_values["sigma0"].append(fitted.sigma0)
        fit_values["nll"].append(fitted.nll)
        fit_values["lrt_stat_vs_flat"].append(lrt)

    intervals = {name: percentile_interval(values) for name, values in fit_values.items()}
    return {
        "reps": reps,
        "seed": seed,
        "intervals": intervals,
    }


def empirical_bins(
    residuals: Sequence[ResidualDraw],
    fitted: FitResult,
) -> List[Dict[str, object]]:
    accum: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "sum_residual": 0.0,
            "sum_residual_sq": 0.0,
            "sum_confidence": 0.0,
            "sum_default_g": 0.0,
            "sum_fitted_g": 0.0,
        }
    )
    for draw in residuals:
        idx = bin_index(draw.confidence)
        entry = accum[idx]
        entry["n"] += 1.0
        entry["sum_residual"] += draw.residual
        entry["sum_residual_sq"] += draw.residual * draw.residual
        entry["sum_confidence"] += draw.confidence
        entry["sum_default_g"] += confidence_weight(
            draw.confidence, DEFAULT_EPS, DEFAULT_GAMMA
        )
        entry["sum_fitted_g"] += confidence_weight(
            draw.confidence, fitted.eps, fitted.gamma
        )

    rows: List[Dict[str, object]] = []
    top_idx = max((idx for idx, values in accum.items() if values["n"] > 0), default=None)
    top_mse = None
    top_default_g = None
    top_fitted_g = None
    if top_idx is not None:
        top = accum[top_idx]
        top_mse = top["sum_residual_sq"] / top["n"] if top["n"] else None
        top_default_g = top["sum_default_g"] / top["n"] if top["n"] else None
        top_fitted_g = top["sum_fitted_g"] / top["n"] if top["n"] else None

    for idx in range(10):
        values = accum.get(idx)
        if not values or values["n"] <= 0:
            rows.append(
                {
                    "bin": bin_label(idx),
                    "n": 0,
                    "mean_confidence": None,
                    "mean_residual": None,
                    "residual_mse": None,
                    "residual_sample_variance": None,
                    "empirical_relative_precision_to_top_bin": None,
                    "default_relative_weight_to_top_bin": None,
                    "fitted_relative_weight_to_top_bin": None,
                }
            )
            continue
        n = int(values["n"])
        mean_residual = values["sum_residual"] / values["n"]
        mse = values["sum_residual_sq"] / values["n"]
        sample_variance = None
        if n > 1:
            sample_variance = (
                values["sum_residual_sq"]
                - values["sum_residual"] * values["sum_residual"] / values["n"]
            ) / (values["n"] - 1.0)
        mean_default_g = values["sum_default_g"] / values["n"]
        mean_fitted_g = values["sum_fitted_g"] / values["n"]
        rows.append(
            {
                "bin": bin_label(idx),
                "n": n,
                "mean_confidence": values["sum_confidence"] / values["n"],
                "mean_residual": mean_residual,
                "residual_mse": mse,
                "residual_sample_variance": sample_variance,
                "empirical_relative_precision_to_top_bin": (top_mse / mse)
                if top_mse is not None and mse > 0.0
                else None,
                "default_relative_weight_to_top_bin": (mean_default_g / top_default_g)
                if top_default_g
                else None,
                "fitted_relative_weight_to_top_bin": (mean_fitted_g / top_fitted_g)
                if top_fitted_g
                else None,
            }
        )
    return rows


def fit_report(
    aggregate: Sequence[Tuple[float, int, float]],
    groups: Sequence[Sequence[ResidualDraw]],
    bootstrap_reps: int,
    seed: int,
) -> Dict[str, object]:
    fitted = fit_confidence_curve(aggregate)
    flat = fit_fixed(aggregate, 1.0e-9, 0.0)
    default = fit_fixed(aggregate, DEFAULT_EPS, DEFAULT_GAMMA)

    lrt = max(0.0, 2.0 * (flat.nll - fitted.nll))
    # Chi-square survival function for df=2.
    lrt_p_df2 = math.exp(-0.5 * lrt)
    aic_flat = 2.0 * 1.0 + 2.0 * flat.nll
    aic_fitted = 2.0 * 3.0 + 2.0 * fitted.nll
    aic_default_fixed = 2.0 * 1.0 + 2.0 * default.nll

    bootstrap = bootstrap_fits(groups, bootstrap_reps, seed)
    gamma_ci = bootstrap.get("intervals", {}).get("gamma", {})
    eps_ci = bootstrap.get("intervals", {}).get("eps", {})
    gamma_covers_zero = (
        gamma_ci.get("p2_5") is not None
        and gamma_ci.get("p97_5") is not None
        and gamma_ci["p2_5"] <= 0.0 <= gamma_ci["p97_5"]
    )
    eps_default_in_ci = (
        eps_ci.get("p2_5") is not None
        and eps_ci.get("p97_5") is not None
        and eps_ci["p2_5"] <= DEFAULT_EPS <= eps_ci["p97_5"]
    )
    gamma_default_in_ci = (
        gamma_ci.get("p2_5") is not None
        and gamma_ci.get("p97_5") is not None
        and gamma_ci["p2_5"] <= DEFAULT_GAMMA <= gamma_ci["p97_5"]
    )
    default_delta_aic = aic_default_fixed - aic_fitted
    defaults_consistent = (
        eps_default_in_ci and gamma_default_in_ci and default_delta_aic <= 10.0
    )
    if lrt_p_df2 >= 0.05 or gamma_covers_zero:
        confidence_map_verdict = "DELETE THE KNOB supported"
    elif not defaults_consistent:
        confidence_map_verdict = "knob wrong: refit or estimate jointly"
    else:
        confidence_map_verdict = "knob survives"

    return {
        "fitted_confidence_curve": {
            "eps": fitted.eps,
            "gamma": fitted.gamma,
            "sigma0": fitted.sigma0,
            "nll": fitted.nll,
            "n_residual_draws": fitted.n,
            "weighted_sse": fitted.weighted_sse,
            "sum_log_g": fitted.sum_log_g,
        },
        "flat_model": {
            "eps": flat.eps,
            "gamma": flat.gamma,
            "sigma0": flat.sigma0,
            "nll": flat.nll,
            "aic": aic_flat,
        },
        "default_curve_fixed_eps_gamma": {
            "eps": default.eps,
            "gamma": default.gamma,
            "sigma0": default.sigma0,
            "nll": default.nll,
            "aic_fixed_curve": aic_default_fixed,
            "delta_aic_vs_best_fitted_curve": default_delta_aic,
        },
        "likelihood_ratio_test_vs_flat": {
            "statistic": lrt,
            "df": 2,
            "p_value_chi_square_df2": lrt_p_df2,
            "flat_rejected_at_0_05": lrt_p_df2 < 0.05,
            "note": "df=2 chi-square approximation; null has boundary/non-identifiability caveat",
        },
        "aic": {
            "flat_k1": aic_flat,
            "fitted_confidence_k3": aic_fitted,
            "default_fixed_curve_k1": aic_default_fixed,
            "delta_fitted_minus_flat": aic_fitted - aic_flat,
            "delta_default_fixed_minus_flat": aic_default_fixed - aic_flat,
        },
        "bootstrap": bootstrap,
        "verdict_inputs": {
            "gamma_ci_covers_zero": gamma_covers_zero,
            "eps_default_inside_ci": eps_default_in_ci,
            "gamma_default_inside_ci": gamma_default_in_ci,
            "default_delta_aic_lte_10": default_delta_aic <= 10.0,
            "defaults_consistent": defaults_consistent,
        },
        "confidence_map_verdict": confidence_map_verdict,
    }


def subgroup_fit_reports(
    groups_by_name: Dict[str, List[List[ResidualDraw]]]
) -> Dict[str, object]:
    reports: Dict[str, object] = {}
    for name, groups in sorted(groups_by_name.items()):
        residuals = flatten_residual_groups(groups)
        if len(residuals) < 20:
            continue
        aggregate = aggregate_residuals(residuals)
        fitted = fit_confidence_curve(aggregate)
        flat = fit_fixed(aggregate, 1.0e-9, 0.0)
        lrt = max(0.0, 2.0 * (flat.nll - fitted.nll))
        reports[name] = {
            "n_residual_draws": len(residuals),
            "n_pair_groups": len(groups),
            "eps": fitted.eps,
            "gamma": fitted.gamma,
            "sigma0": fitted.sigma0,
            "nll": fitted.nll,
            "flat_nll": flat.nll,
            "lrt_stat_vs_flat": lrt,
            "p_value_chi_square_df2": math.exp(-0.5 * lrt),
        }
    return reports


def build_subgroup_groups(
    observations: Sequence[Observation],
) -> Dict[str, List[List[ResidualDraw]]]:
    raw_groups: Dict[
        Tuple[str, str, str, str, str, str], List[Observation]
    ] = defaultdict(list)
    for obs in observations:
        raw_groups[group_key(obs)].append(obs)

    result: Dict[str, List[List[ResidualDraw]]] = defaultdict(list)
    for key, draws in raw_groups.items():
        if len(draws) < 2:
            continue
        mean_y = sum(draw.signed_log_ratio_toward_first for draw in draws) / len(draws)
        residual_group = [
            ResidualDraw(draw.confidence, draw.signed_log_ratio_toward_first - mean_y)
            for draw in draws
        ]
        source, model, template, _attribute, _first, _second = key
        result[f"{source} | {model} | {template}"].append(residual_group)
    return result


def primary_verdict(
    histograms: Sequence[Dict[str, object]],
    confidence_map_verdict: str,
) -> Dict[str, object]:
    degenerate = [hist for hist in histograms if hist["degenerate_gt_90pct"]]
    if degenerate:
        return {
            "primary_verdict": "knob unexercised on live data",
            "reason": "At least one source/model/template has >90% mass on one exact confidence value.",
            "degenerate_histograms": [
                {
                    "source": hist["source"],
                    "model": hist["model"],
                    "prompt_template_slug": hist["prompt_template_slug"],
                    "top_exact_confidence": hist["top_exact_confidence"],
                    "top_exact_confidence_mass": hist["top_exact_confidence_mass"],
                    "valid_observations": hist["valid_observations"],
                }
                for hist in degenerate
            ],
            "secondary_confidence_map_verdict": confidence_map_verdict,
        }
    return {
        "primary_verdict": confidence_map_verdict,
        "reason": "No source/model/template confidence histogram crossed the >90% single-value degeneracy gate.",
        "degenerate_histograms": [],
    }


def write_results(output_path: Path, results: Dict[str, object]) -> None:
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    root = repo_root()
    sources = source_specs(root)
    all_observations: List[Observation] = []
    source_summaries: List[Dict[str, object]] = []
    for spec in sources:
        observations, summary = load_source(spec)
        all_observations.extend(observations)
        source_summaries.append(summary)

    histograms = confidence_histograms(all_observations, source_summaries)
    groups, residual_summary = residual_groups(all_observations)
    residuals = flatten_residual_groups(groups)
    aggregate = aggregate_residuals(residuals)
    fit = fit_report(aggregate, groups, args.bootstrap_reps, args.seed)
    fitted_curve = fit["fitted_confidence_curve"]
    fitted_for_bins = FitResult(
        eps=fitted_curve["eps"],
        gamma=fitted_curve["gamma"],
        sigma0=fitted_curve["sigma0"],
        nll=fitted_curve["nll"],
        n=fitted_curve["n_residual_draws"],
        weighted_sse=fitted_curve["weighted_sse"],
        sum_log_g=fitted_curve["sum_log_g"],
    )
    bins = empirical_bins(residuals, fitted_for_bins)
    subgroup_reports = subgroup_fit_reports(build_subgroup_groups(all_observations))

    results: Dict[str, object] = {
        "receipt": "confidence-knob-2026-07-09",
        "script": str(Path(__file__).relative_to(root)),
        "configuration": {
            "default_eps_confidence": DEFAULT_EPS,
            "default_gamma_confidence": DEFAULT_GAMMA,
            "degenerate_mass_threshold": DEGENERATE_MASS_THRESHOLD,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.seed,
            "sqlite_open_mode": "mode=ro&immutable=1",
        },
        "sources": source_summaries,
        "totals": {
            "sqlite_rows": sum(int(summary["total_rows"]) for summary in source_summaries),
            "refused_rows": sum(int(summary["refused_rows"]) for summary in source_summaries),
            "malformed_nonrefused_rows": sum(
                int(summary["malformed_nonrefused_rows"])
                for summary in source_summaries
            ),
            "valid_observation_rows": len(all_observations),
        },
        "confidence_histograms_by_source_model_template": histograms,
        "residual_summary": residual_summary,
        "empirical_precision_bins": bins,
        "maximum_likelihood": fit,
        "subgroup_fit_sensitivity_by_source_model_template": subgroup_reports,
    }
    results["verdict"] = primary_verdict(
        histograms, fit["confidence_map_verdict"]
    )

    output_path = Path(__file__).resolve().parent / "results.json"
    write_results(output_path, results)

    print(json.dumps(results["verdict"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
