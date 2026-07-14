#!/usr/bin/env python3
"""Run a real OpenRouter-backed method comparison for structured list judgments.

This script compares live judging regimes on the same frozen,
attribute-weighted cases:

- scalar_matrix: one structured score matrix call per case
- list_sort: one structured whole-list sorting call per case
- ordinal_pairwise: pairwise direction-only judgments per attribute
- cardinal_pairwise_ratio: pairwise direction + magnitude judgments per attribute

A reference_pairwise_ratio regime is run with a configurable reference model
and used as the comparison target for metrics. Cases are loaded from
examples/live-method-suite.json by default so the benchmark prompt surface is
versioned separately from the runner. Every call writes a key-free request body,
raw OpenRouter response body, parsed judgment, usage, and cost accounting into
the output directory.
"""

from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=text"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_SUITE_PATH = Path(__file__).with_name("live-method-suite.json")
SUMMARY_SCHEMA_VERSION = "live_method_comparison_summary_v1"

@dataclass(frozen=True)
class Attribute:
    id: str
    prompt: str
    weight: float


@dataclass(frozen=True)
class Item:
    id: str
    text: str


@dataclass(frozen=True)
class LiveJudgmentCase:
    name: str
    description: str
    items: tuple[Item, ...]
    attributes: tuple[Attribute, ...]


def expect_str(row: dict[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{context} missing non-empty string field {key!r}")
    return value


def parse_items(rows: Any, case_name: str) -> tuple[Item, ...]:
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"case {case_name!r} must contain a non-empty items list")
    items: list[Item] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"case {case_name!r} item {index} must be an object")
        item = Item(expect_str(row, "id", f"case {case_name!r} item {index}"), expect_str(row, "text", f"case {case_name!r} item {index}"))
        if item.id in seen:
            raise SystemExit(f"case {case_name!r} has duplicate item id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if len(items) < 2:
        raise SystemExit(f"case {case_name!r} must contain at least two items")
    return tuple(items)


def parse_attributes(rows: Any, case_name: str) -> tuple[Attribute, ...]:
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"case {case_name!r} must contain a non-empty attributes list")
    attributes: list[Attribute] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"case {case_name!r} attribute {index} must be an object")
        attr_id = expect_str(row, "id", f"case {case_name!r} attribute {index}")
        if attr_id in seen:
            raise SystemExit(f"case {case_name!r} has duplicate attribute id {attr_id!r}")
        seen.add(attr_id)
        try:
            weight = float(row.get("weight"))
        except (TypeError, ValueError) as err:
            raise SystemExit(f"case {case_name!r} attribute {attr_id!r} has invalid weight") from err
        if not math.isfinite(weight) or weight <= 0.0:
            raise SystemExit(f"case {case_name!r} attribute {attr_id!r} weight must be positive and finite")
        attributes.append(Attribute(attr_id, expect_str(row, "prompt", f"case {case_name!r} attribute {attr_id!r}"), weight))
    return tuple(attributes)


def load_suite(path: Path) -> tuple[str, str, tuple[LiveJudgmentCase, ...]]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict):
        raise SystemExit(f"suite file must contain a JSON object: {path}")
    suite_name = expect_str(suite, "name", "suite")
    suite_description = expect_str(suite, "description", f"suite {suite_name!r}")
    raw_cases = suite.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SystemExit(f"suite {suite_name!r} must contain a non-empty cases list")
    cases: list[LiveJudgmentCase] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_cases):
        if not isinstance(row, dict):
            raise SystemExit(f"suite {suite_name!r} case {index} must be an object")
        case_name = expect_str(row, "name", f"suite {suite_name!r} case {index}")
        if case_name in seen:
            raise SystemExit(f"suite {suite_name!r} has duplicate case name {case_name!r}")
        seen.add(case_name)
        cases.append(
            LiveJudgmentCase(
                name=case_name,
                description=expect_str(row, "description", f"case {case_name!r}"),
                items=parse_items(row.get("items"), case_name),
                attributes=parse_attributes(row.get("attributes"), case_name),
            )
        )
    return suite_name, suite_description, tuple(cases)


class OpenRouterClient:
    def __init__(self, api_key: str, out_dir: Path, max_usd: float, temperature: float) -> None:
        if not api_key.startswith("sk-or-"):
            raise SystemExit("OPENROUTER_API_KEY must look like an OpenRouter key")
        self.api_key = api_key
        self.out_dir = out_dir
        self.max_nanodollars = int(max_usd * 1_000_000_000)
        self.temperature = temperature
        self.models = self._load_models()
        self.total_cost_nanodollars = 0
        self.call_count = 0

    def _load_models(self) -> dict[str, dict[str, Any]]:
        with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = data.get("data")
        if not isinstance(models, list):
            raise SystemExit("OpenRouter models response did not contain a data list")
        return {str(model.get("id")): model for model in models if isinstance(model, dict)}

    def assert_model(self, model: str) -> None:
        meta = self.models.get(model)
        if meta is None:
            raise SystemExit(f"model not present in live OpenRouter metadata: {model}")
        params = set(meta.get("supported_parameters") or [])
        if "response_format" not in params and "structured_outputs" not in params:
            raise SystemExit(f"model does not advertise structured JSON support: {model}")

    def pricing(self, model: str) -> dict[str, float]:
        raw = (self.models.get(model) or {}).get("pricing") or {}
        return {
            "prompt": float(raw.get("prompt") or 0.0),
            "completion": float(raw.get("completion") or 0.0),
        }

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        call_dir: Path,
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.assert_model(model)
        if self.total_cost_nanodollars >= self.max_nanodollars:
            raise SystemExit(
                f"max live benchmark spend reached before call {self.call_count + 1}: "
                f"${self.total_cost_nanodollars / 1_000_000_000:.6f}"
            )

        body = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        call_dir.mkdir(parents=True, exist_ok=True)
        write_json(call_dir / "request.json", body)

        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/XyraSinclair/cardinal-harness",
                "X-Title": "cardinal-harness live method comparison",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw_body = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as err:
            error_body = err.read().decode("utf-8", errors="replace")
            write_json(
                call_dir / "error.json",
                {"status": err.code, "body": error_body[:4000], "model": model},
            )
            raise SystemExit(f"OpenRouter HTTP {err.code} for {model}; see {call_dir / 'error.json'}") from err

        latency_ms = int((time.perf_counter() - started) * 1000)
        response_body = json.loads(raw_body)
        write_json(call_dir / "response.json", response_body)
        content = extract_content(response_body)
        parsed = parse_json_object(content)
        usage = summarize_usage(response_body, self.pricing(model), latency_ms)
        self.total_cost_nanodollars += usage["cost_nanodollars"]
        self.call_count += 1
        write_json(call_dir / "parsed.json", parsed)
        write_json(call_dir / "usage.json", usage)
        if self.total_cost_nanodollars > self.max_nanodollars:
            raise SystemExit(
                f"max live benchmark spend exceeded after call {self.call_count}: "
                f"${self.total_cost_nanodollars / 1_000_000_000:.6f}"
            )
        return parsed, usage


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_content(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit("OpenRouter response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise SystemExit("OpenRouter response choice has no message")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str) and arguments.strip():
                return arguments
    raise SystemExit("OpenRouter response did not contain JSON content")


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    decoder = json.JSONDecoder()

    candidates: list[str] = [stripped]
    lines = stripped.splitlines()
    in_fence = False
    fence_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            if in_fence:
                candidates.insert(0, "\n".join(fence_lines).strip())
                fence_lines = []
                in_fence = False
            else:
                in_fence = True
            continue
        if in_fence:
            fence_lines.append(line)

    if stripped.startswith("```") and stripped.endswith("```") and len(lines) >= 3:
        candidates.insert(0, "\n".join(lines[1:-1]).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        raise SystemExit("model output JSON was not an object")

    for start, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise SystemExit(f"model output did not contain a JSON object: {stripped[:300]}")


def summarize_usage(response_body: dict[str, Any], pricing: dict[str, float], latency_ms: int) -> dict[str, Any]:
    usage = response_body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("cost_details") or {}
    upstream_cost = details.get("upstream_inference_cost")
    if isinstance(upstream_cost, (int, float)):
        cost_nanodollars = max(0, int(round(float(upstream_cost) * 1_000_000_000)))
        is_estimate = False
    else:
        cost_nanodollars = max(
            0,
            int(round((prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]) * 1_000_000_000)),
        )
        is_estimate = True
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_nanodollars": cost_nanodollars,
        "cost_usd": cost_nanodollars / 1_000_000_000,
        "cost_is_estimate": is_estimate,
        "latency_ms": latency_ms,
        "calls": 1,
    }


def system_prompt() -> str:
    return (
        "You are a rigorous structured-judgment engine. Judge only the stated attribute. "
        "Use the item text, not outside popularity. Return strict JSON only. "
        "Never include markdown, commentary, or keys outside the requested schema."
    )


def case_context(case: LiveJudgmentCase) -> str:
    attrs = "\n".join(f"- {a.id} (weight {a.weight}): {a.prompt}" for a in case.attributes)
    items = "\n".join(f"- {item.id}: {item.text}" for item in case.items)
    return f"Case: {case.name}\nDescription: {case.description}\n\nAttributes:\n{attrs}\n\nItems:\n{items}"


def weighted_utility(attribute_scores: dict[str, dict[str, float]], case: LiveJudgmentCase) -> dict[str, float]:
    weights = {a.id: a.weight for a in case.attributes}
    return {
        item_id: sum(weights[attr_id] * scores.get(attr_id, 0.0) for attr_id in weights)
        for item_id, scores in attribute_scores.items()
    }


def ranked_ids_from_utility(utility: dict[str, float]) -> list[str]:
    return [item_id for item_id, _score in sorted(utility.items(), key=lambda kv: (-kv[1], kv[0]))]


def normalize_scores(raw: dict[str, float]) -> dict[str, float]:
    values = list(raw.values())
    if not values:
        return {}
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return {key: 0.5 for key in raw}
    return {key: (value - lo) / (hi - lo) for key, value in raw.items()}


def validate_item_id(item_id: str, case: LiveJudgmentCase) -> None:
    valid = {item.id for item in case.items}
    if item_id not in valid:
        raise SystemExit(f"model returned unknown item id {item_id!r}; expected one of {sorted(valid)}")


def pair_higher_id(raw_value: Any, case: LiveJudgmentCase, left: Item, right: Item, method: str) -> str:
    raw = str(raw_value).strip()
    normalized = raw.lower()
    if normalized in {"a", "item a", "option a", "left"}:
        return left.id
    if normalized in {"b", "item b", "option b", "right"}:
        return right.id
    validate_item_id(raw, case)
    if raw not in {left.id, right.id}:
        raise SystemExit(f"{method} pair returned {raw}, not in pair {left.id}/{right.id}")
    return raw


def run_scalar_matrix(client: OpenRouterClient, case: LiveJudgmentCase, model: str, out_dir: Path) -> dict[str, Any]:
    schema = {
        "scores": [
            {
                "id": "item id",
                "attributes": {
                    "attribute_id": {"score": "number from 0 to 100", "confidence": "number from 0 to 1"}
                },
            }
        ]
    }
    parsed, usage = client.chat_json(
        model=model,
        max_tokens=2200,
        call_dir=out_dir / "calls" / "scalar_matrix",
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": case_context(case)
                + "\n\nTask: Score every item independently on every attribute. "
                + "Output JSON exactly like this schema: "
                + json.dumps(schema),
            },
        ],
    )
    attribute_scores: dict[str, dict[str, float]] = {item.id: {} for item in case.items}
    confidence_by_item: dict[str, dict[str, float]] = {item.id: {} for item in case.items}
    rows = parsed.get("scores")
    if not isinstance(rows, list):
        raise SystemExit("scalar_matrix response missing scores list")
    for row in rows:
        item_id = str(row.get("id"))
        validate_item_id(item_id, case)
        attrs = row.get("attributes")
        if not isinstance(attrs, dict):
            raise SystemExit(f"scalar_matrix row for {item_id} missing attributes object")
        for attr in case.attributes:
            cell = attrs.get(attr.id)
            if not isinstance(cell, dict):
                raise SystemExit(f"scalar_matrix row for {item_id} missing attribute {attr.id}")
            attribute_scores[item_id][attr.id] = clamp(float(cell.get("score")) / 100.0, 0.0, 1.0)
            confidence_by_item[item_id][attr.id] = clamp(float(cell.get("confidence", 0.5)), 0.0, 1.0)
    utility = weighted_utility(attribute_scores, case)
    return method_result("scalar_matrix", model, attribute_scores, utility, usage, {"confidence": confidence_by_item})


def run_list_sort(client: OpenRouterClient, case: LiveJudgmentCase, model: str, out_dir: Path) -> dict[str, Any]:
    schema = {"ranked_ids": ["best item id", "next item id"], "confidence": "number from 0 to 1"}
    parsed, usage = client.chat_json(
        model=model,
        max_tokens=1200,
        call_dir=out_dir / "calls" / "list_sort",
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": case_context(case)
                + "\n\nTask: Return one global ranking that maximizes the weighted attributes. "
                + "Output JSON exactly like this schema: "
                + json.dumps(schema),
            },
        ],
    )
    ranked = parsed.get("ranked_ids")
    if not isinstance(ranked, list):
        raise SystemExit("list_sort response missing ranked_ids")
    ranked_ids = [str(item_id) for item_id in ranked]
    expected = [item.id for item in case.items]
    if sorted(ranked_ids) != sorted(expected):
        raise SystemExit(f"list_sort ranked_ids must contain each item exactly once: got {ranked_ids}")
    n = len(ranked_ids)
    base_scores = {item_id: (n - idx - 1) / max(n - 1, 1) for idx, item_id in enumerate(ranked_ids)}
    attribute_scores = {item.id: {attr.id: base_scores[item.id] for attr in case.attributes} for item in case.items}
    utility = weighted_utility(attribute_scores, case)
    return method_result(
        "list_sort",
        model,
        attribute_scores,
        utility,
        usage,
        {"ranked_ids_raw": ranked_ids, "confidence": clamp(float(parsed.get("confidence", 0.5)), 0.0, 1.0)},
    )


def run_ordinal_pairwise(client: OpenRouterClient, case: LiveJudgmentCase, model: str, out_dir: Path) -> dict[str, Any]:
    attribute_scores: dict[str, dict[str, float]] = {item.id: {} for item in case.items}
    pair_rows: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    for attr in case.attributes:
        wins = {item.id: 0.0 for item in case.items}
        counts = {item.id: 0 for item in case.items}
        for pair_index, (left, right) in enumerate(itertools.combinations(case.items, 2), 1):
            parsed, usage = client.chat_json(
                model=model,
                max_tokens=700,
                call_dir=out_dir / "calls" / "ordinal_pairwise" / attr.id / f"pair_{pair_index:02d}_{left.id}_vs_{right.id}",
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            f"Case: {case.name}\nAttribute {attr.id}: {attr.prompt}\n\n"
                            f"A ({left.id}): {left.text}\n\nB ({right.id}): {right.text}\n\n"
                            "Task: Which item has more of this attribute? Choose one item ID. "
                            "Output JSON exactly: {\"higher_id\":\"item id\",\"confidence\":0.0}."
                        ),
                    },
                ],
            )
            higher_id = pair_higher_id(parsed.get("higher_id"), case, left, right, "ordinal")
            confidence = clamp(float(parsed.get("confidence", 0.5)), 0.0, 1.0)
            wins[higher_id] += confidence
            counts[left.id] += 1
            counts[right.id] += 1
            pair_rows.append({"attribute": attr.id, "left": left.id, "right": right.id, "higher_id": higher_id, "confidence": confidence})
            usages.append(usage)
        normalized = normalize_scores({item_id: wins[item_id] / max(counts[item_id], 1) for item_id in wins})
        for item_id, score in normalized.items():
            attribute_scores[item_id][attr.id] = score
    usage = aggregate_usage(usages)
    utility = weighted_utility(attribute_scores, case)
    return method_result("ordinal_pairwise", model, attribute_scores, utility, usage, {"pairs": pair_rows})


def run_cardinal_pairwise_ratio(client: OpenRouterClient, case: LiveJudgmentCase, model: str, out_dir: Path, method_name: str) -> dict[str, Any]:
    attribute_scores: dict[str, dict[str, float]] = {item.id: {} for item in case.items}
    pair_rows: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    for attr in case.attributes:
        log_scores = {item.id: 0.0 for item in case.items}
        counts = {item.id: 0 for item in case.items}
        for pair_index, (left, right) in enumerate(itertools.combinations(case.items, 2), 1):
            parsed, usage = client.chat_json(
                model=model,
                max_tokens=800,
                call_dir=out_dir / "calls" / method_name / attr.id / f"pair_{pair_index:02d}_{left.id}_vs_{right.id}",
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {
                        "role": "user",
                        "content": (
                            f"Case: {case.name}\nAttribute {attr.id}: {attr.prompt}\n\n"
                            f"A ({left.id}): {left.text}\n\nB ({right.id}): {right.text}\n\n"
                            "Task: Which item has more of this attribute, and how many times more? "
                            "Use ratio=1 only for near ties; cap large gaps at 26. "
                            "Output JSON exactly: {\"higher_id\":\"item id\",\"ratio\":1.0,\"confidence\":0.0}."
                        ),
                    },
                ],
            )
            higher_id = pair_higher_id(parsed.get("higher_id"), case, left, right, "cardinal")
            ratio = clamp(float(parsed.get("ratio")), 1.0, 26.0)
            confidence = clamp(float(parsed.get("confidence", 0.5)), 0.0, 1.0)
            magnitude = math.log(ratio) * max(confidence, 0.05)
            if higher_id == left.id:
                log_scores[left.id] += magnitude
                log_scores[right.id] -= magnitude
            else:
                log_scores[right.id] += magnitude
                log_scores[left.id] -= magnitude
            counts[left.id] += 1
            counts[right.id] += 1
            pair_rows.append(
                {
                    "attribute": attr.id,
                    "left": left.id,
                    "right": right.id,
                    "higher_id": higher_id,
                    "ratio": ratio,
                    "confidence": confidence,
                }
            )
            usages.append(usage)
        normalized = normalize_scores({item_id: log_scores[item_id] / max(counts[item_id], 1) for item_id in log_scores})
        for item_id, score in normalized.items():
            attribute_scores[item_id][attr.id] = score
    usage = aggregate_usage(usages)
    utility = weighted_utility(attribute_scores, case)
    return method_result(method_name, model, attribute_scores, utility, usage, {"pairs": pair_rows})


def method_result(
    method: str,
    model: str,
    attribute_scores: dict[str, dict[str, float]],
    utility: dict[str, float],
    usage: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    ranked_ids = ranked_ids_from_utility(utility)
    return {
        "method": method,
        "model": model,
        "ranked_ids": ranked_ids,
        "utility": utility,
        "attribute_scores": attribute_scores,
        "usage": usage,
        "extra": extra,
    }


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in rows)
    cost_nanodollars = sum(int(row.get("cost_nanodollars", 0)) for row in rows)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_nanodollars": cost_nanodollars,
        "cost_usd": cost_nanodollars / 1_000_000_000,
        "cost_is_estimate": any(bool(row.get("cost_is_estimate")) for row in rows),
        "latency_ms": sum(int(row.get("latency_ms", 0)) for row in rows),
        "calls": sum(int(row.get("calls", 1)) for row in rows),
    }


def clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        raise SystemExit("model returned a non-finite numeric value")
    return min(max(value, lo), hi)


def kendall_tau(reference: list[str], candidate: list[str]) -> float:
    if sorted(reference) != sorted(candidate):
        raise SystemExit("cannot compare rankings with different item sets")
    pos_ref = {item_id: idx for idx, item_id in enumerate(reference)}
    pos_cand = {item_id: idx for idx, item_id in enumerate(candidate)}
    concordant = 0
    discordant = 0
    for a, b in itertools.combinations(reference, 2):
        ref_order = pos_ref[a] < pos_ref[b]
        cand_order = pos_cand[a] < pos_cand[b]
        if ref_order == cand_order:
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant
    return 1.0 if total == 0 else (concordant - discordant) / total


def topk_jaccard(reference: list[str], candidate: list[str], k: int) -> float:
    a = set(reference[:k])
    b = set(candidate[:k])
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare_to_reference(reference: dict[str, Any], result: dict[str, Any], k: int) -> dict[str, Any]:
    return {
        "kendall_tau": kendall_tau(reference["ranked_ids"], result["ranked_ids"]),
        "topk_jaccard": topk_jaccard(reference["ranked_ids"], result["ranked_ids"], k),
        "reference_ranked_ids": reference["ranked_ids"],
        "candidate_ranked_ids": result["ranked_ids"],
    }


def agreement_score(metrics: dict[str, Any]) -> float:
    tau_unit = (float(metrics["kendall_tau"]) + 1.0) / 2.0
    topk = float(metrics["topk_jaccard"])
    return (tau_unit + topk) / 2.0


def budget_normalized_methods(cases_out: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for case in cases_out:
        rows = [case["reference"], *case["methods"]]
        for result in rows:
            method = result["method"]
            bucket = buckets.setdefault(
                method,
                {
                    "method": method,
                    "model": result["model"],
                    "case_count": 0,
                    "agreement_score_total": 0.0,
                    "kendall_tau_total": 0.0,
                    "topk_jaccard_total": 0.0,
                    "usages": [],
                },
            )
            metrics = result["metrics_vs_reference"]
            bucket["case_count"] += 1
            bucket["agreement_score_total"] += agreement_score(metrics)
            bucket["kendall_tau_total"] += float(metrics["kendall_tau"])
            bucket["topk_jaccard_total"] += float(metrics["topk_jaccard"])
            bucket["usages"].append(result["usage"])
    normalized: list[dict[str, Any]] = []
    for bucket in buckets.values():
        usage = aggregate_usage(bucket["usages"])
        case_count = int(bucket["case_count"])
        agreement_total = float(bucket["agreement_score_total"])
        calls = int(usage["calls"])
        total_tokens = int(usage["total_tokens"])
        cost_usd = float(usage["cost_usd"])
        normalized.append(
            {
                "method": bucket["method"],
                "model": bucket["model"],
                "case_count": case_count,
                "mean_kendall_tau": bucket["kendall_tau_total"] / case_count,
                "mean_topk_jaccard": bucket["topk_jaccard_total"] / case_count,
                "mean_agreement_score": agreement_total / case_count,
                "agreement_score_total": agreement_total,
                "usage": usage,
                "agreement_score_per_call": agreement_total / calls if calls else None,
                "agreement_score_per_1k_tokens": agreement_total / (total_tokens / 1000.0) if total_tokens else None,
                "agreement_score_per_usd": agreement_total / cost_usd if cost_usd else None,
            }
        )
    return sorted(normalized, key=lambda row: (-row["mean_agreement_score"], row["method"]))


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Live Structured-Judgment Method Comparison",
        "",
        "This study record is generated from real OpenRouter chat-completion calls.",
        "It compares structured judgment regimes on the same frozen attribute-weighted item lists.",
        "The reference is another live LLM regime, not human ground truth.",
        "",
        f"Schema: `{summary['schema_version']}`",
        f"Suite: `{summary['suite']['name']}`",
        f"Suite path: `{summary['suite']['path']}`",
        f"Suite SHA-256: `{summary['suite']['sha256']}`",
        f"Candidate model: `{summary['candidate_model']}`",
        f"Reference model: `{summary['reference_model']}`",
        f"Cases: {summary['case_count']}",
        f"OpenRouter calls: {summary['totals']['calls']}",
        f"Prompt tokens: {summary['totals']['prompt_tokens']}",
        f"Completion tokens: {summary['totals']['completion_tokens']}",
        f"Total tokens: {summary['totals']['total_tokens']}",
        f"Provider cost: ${summary['totals']['cost_usd']:.6f}",
        f"Any estimated cost rows: {summary['totals']['cost_is_estimate']}",
        "",
        "| Case | Method | Model | Kendall tau vs reference | Top-k Jaccard | Cost USD | Ranking |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for case in summary["cases"]:
        for method in case["methods"]:
            metrics = method["metrics_vs_reference"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{case['case']}`",
                        f"`{method['method']}`",
                        f"`{method['model']}`",
                        f"{metrics['kendall_tau']:.3f}",
                        f"{metrics['topk_jaccard']:.3f}",
                        f"${method['usage']['cost_usd']:.6f}",
                        ", ".join(method["ranked_ids"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Budget-normalized aggregate",
            "",
            "`agreement_score` averages normalized Kendall tau `((tau + 1) / 2)` and top-k Jaccard.",
            "The per-call/token/dollar columns are descriptive normalizations of this run, not a substitute for a separately equalized-budget experiment.",
            "",
            "| Method | Mean tau | Mean top-k | Mean agreement | Calls | Total tokens | Cost USD | Agreement/call | Agreement/1k tokens | Agreement/USD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["budget_normalized_methods"]:
        usage = row["usage"]
        per_usd = row["agreement_score_per_usd"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['method']}`",
                    f"{row['mean_kendall_tau']:.3f}",
                    f"{row['mean_topk_jaccard']:.3f}",
                    f"{row['mean_agreement_score']:.3f}",
                    str(usage["calls"]),
                    str(usage["total_tokens"]),
                    f"${usage['cost_usd']:.6f}",
                    f"{row['agreement_score_per_call']:.4f}",
                    f"{row['agreement_score_per_1k_tokens']:.4f}",
                    "n/a" if per_usd is None else f"{per_usd:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- These are real provider calls, but the reference is still an LLM reference, not an external oracle.",
            "- High agreement with the reference means a regime recovered this live reference ordering on this task family and budget.",
            "- Low agreement is useful evidence of prompt/regime brittleness, not proof that the model is incapable.",
            "- Budget-normalized columns are useful for comparing this run's efficiency; they do not prove equal-token or equal-dollar dominance.",
            "- Cost comparisons are only headline-safe when `cost_is_estimate` is false for every row being compared.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def selected_cases(cases: tuple[LiveJudgmentCase, ...], names: list[str] | None) -> list[LiveJudgmentCase]:
    if not names:
        return list(cases)
    wanted = set(names)
    selected = [case for case in cases if case.name in wanted]
    missing = wanted - {case.name for case in selected}
    if missing:
        available = ", ".join(case.name for case in cases)
        raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}; available cases: {available}")
    return selected


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real OpenRouter method comparisons for structured list judgments.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument("--candidate-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--reference-model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--case", action="append")
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient(api_key, out_dir, max_usd=args.max_usd, temperature=args.temperature)
    client.assert_model(args.candidate_model)
    client.assert_model(args.reference_model)
    suite_name, suite_description, cases = load_suite(args.suite)

    cases_out: list[dict[str, Any]] = []
    for case in selected_cases(cases, args.case):
        case_dir = out_dir / case.name
        case_manifest = {
            "name": case.name,
            "description": case.description,
            "items": [item.__dict__ for item in case.items],
            "attributes": [attr.__dict__ for attr in case.attributes],
        }
        write_json(case_dir / "case.json", case_manifest)

        reference = run_cardinal_pairwise_ratio(
            client,
            case,
            args.reference_model,
            case_dir,
            "reference_pairwise_ratio",
        )
        methods = [
            run_scalar_matrix(client, case, args.candidate_model, case_dir),
            run_list_sort(client, case, args.candidate_model, case_dir),
            run_ordinal_pairwise(client, case, args.candidate_model, case_dir),
            run_cardinal_pairwise_ratio(client, case, args.candidate_model, case_dir, "cardinal_pairwise_ratio"),
        ]
        reference["metrics_vs_reference"] = compare_to_reference(reference, reference, min(3, len(case.items)))
        for method in methods:
            method["metrics_vs_reference"] = compare_to_reference(reference, method, min(3, len(case.items)))

        write_json(case_dir / "reference_pairwise_ratio.json", reference)
        for method in methods:
            write_json(case_dir / f"{method['method']}.json", method)
        cases_out.append({"case": case.name, "reference": reference, "methods": methods})

    all_results = [case["reference"] for case in cases_out] + [method for case in cases_out for method in case["methods"]]
    total_usage = aggregate_usage([result["usage"] for result in all_results])
    total_usage["calls"] = client.call_count
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "suite": {
            "name": suite_name,
            "description": suite_description,
            "path": portable_path(args.suite),
            "sha256": file_sha256(args.suite),
        },
        "candidate_model": args.candidate_model,
        "reference_model": args.reference_model,
        "case_count": len(cases_out),
        "totals": total_usage,
        "budget_normalized_methods": budget_normalized_methods(cases_out),
        "cases": cases_out,
        "metadata": {
            "models_url": OPENROUTER_MODELS_URL,
            "generated_unix_ms": int(time.time() * 1000),
            "max_usd": args.max_usd,
            "temperature": args.temperature,
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_markdown(summary, out_dir / "summary.md")
    write_markdown(summary, out_dir / "README.md")
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'} and {out_dir / 'README.md'}")
    print(f"calls {client.call_count}; cost ${client.total_cost_nanodollars / 1_000_000_000:.6f}")


if __name__ == "__main__":
    main()
