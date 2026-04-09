#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

RATIO_LADDER = [
    1.0,
    1.05,
    1.1,
    1.2,
    1.3,
    1.5,
    1.75,
    2.1,
    2.5,
    3.1,
    3.9,
    5.1,
    6.8,
    9.2,
    12.7,
    18.0,
    26.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local model over a prompt grid and seed cardinal's SQLite cache."
    )
    parser.add_argument("--prompt-grid", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HF model or PEFT adapter path for inference.")
    parser.add_argument("--cache-db", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-snap-to-ladder", action="store_true")
    parser.add_argument("--input-cost-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def render_chat_text(
    messages: list[dict[str, str]],
    tokenizer: Any | None = None,
    *,
    add_generation_prompt: bool = False,
) -> str:
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    parts: list[str] = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"<{role}>\n{message['content']}\n</{role}>")
    if add_generation_prompt:
        parts.append("<ASSISTANT>\n")
    return "\n\n".join(parts)


def preview(records: list[dict[str, Any]]) -> None:
    attributes = sorted({record["attribute"]["id"] for record in records})
    swapped = sum(1 for record in records if record["presentation"]["swapped"])
    print(f"records={len(records)} swapped={swapped}")
    print(f"attributes={', '.join(attributes)}")
    if records:
        sample = records[0]
        print("\n--- sample prompt ---")
        print(render_chat_text(sample["messages"], add_generation_prompt=True))


def extract_json(raw: str) -> str:
    trimmed = raw.strip()
    if trimmed.startswith("{"):
        depth = 0
        for idx, char in enumerate(trimmed):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return trimmed[: idx + 1]
    start = trimmed.find("{")
    if start >= 0:
        depth = 0
        remainder = trimmed[start:]
        for idx, char in enumerate(remainder):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return remainder[: idx + 1]
    return trimmed


def snap_ratio_to_ladder(ratio: float) -> float:
    return min(RATIO_LADDER, key=lambda candidate: abs(candidate - ratio))


def parse_pairwise_response(
    raw: str,
    *,
    snap_to_ladder: bool,
) -> tuple[dict[str, Any], str | None, bool]:
    try:
        parsed = json.loads(extract_json(raw))
        if parsed.get("refused", False):
            return {"refused": True}, None, False

        higher_ranked = str(parsed["higher_ranked"]).upper()
        if higher_ranked not in {"A", "B"}:
            raise ValueError(f"invalid higher_ranked: {higher_ranked}")

        ratio = float(parsed["ratio"])
        if not 1.0 <= ratio <= 26.0:
            raise ValueError(f"ratio out of allowed range [1,26]: {ratio}")

        confidence = max(0.0, min(1.0, float(parsed["confidence"])))
        snapped = False
        if snap_to_ladder:
            snapped_ratio = snap_ratio_to_ladder(ratio)
            snapped = abs(snapped_ratio - ratio) > 1e-9
            ratio = snapped_ratio
        if abs(ratio - 1.0) < 1e-9:
            higher_ranked = "A"

        return {
            "refused": False,
            "higher_ranked": higher_ranked,
            "ratio": ratio,
            "confidence": confidence,
        }, None, snapped
    except Exception as exc:
        return {"refused": True}, str(exc), False


def canonical_signature(prediction: dict[str, Any], swapped: bool) -> tuple[Any, ...]:
    if prediction["refused"]:
        return ("refused",)

    higher_ranked = prediction["higher_ranked"]
    ratio = prediction["ratio"]
    if abs(ratio - 1.0) < 1e-9:
        higher_ranked = "A"
    elif swapped:
        higher_ranked = "B" if higher_ranked == "A" else "A"
    return ("observation", higher_ranked, ratio)


def estimate_cost_nanodollars(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million_tokens: float,
    output_cost_per_million_tokens: float,
) -> int:
    return int(
        round(
            input_tokens * input_cost_per_million_tokens * 1000.0
            + output_tokens * output_cost_per_million_tokens * 1000.0
        )
    )


def init_cache(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pairwise_cache (
          key_hash TEXT PRIMARY KEY,
          model TEXT NOT NULL,
          prompt_template_slug TEXT NOT NULL,
          template_hash TEXT NOT NULL,
          attribute_id TEXT NOT NULL,
          attribute_prompt_hash TEXT NOT NULL,
          entity_a_id TEXT NOT NULL,
          entity_b_id TEXT NOT NULL,
          entity_a_hash TEXT NOT NULL,
          entity_b_hash TEXT NOT NULL,
          higher_ranked TEXT,
          ratio REAL,
          confidence REAL,
          refused INTEGER NOT NULL,
          input_tokens INTEGER,
          output_tokens INTEGER,
          provider_cost_nanodollars INTEGER,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          hit_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def upsert_cache_row(
    conn: sqlite3.Connection,
    record: dict[str, Any],
    prediction: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    provider_cost_nanodollars: int,
) -> None:
    now = int(time.time())
    cache = record["cache"]
    conn.execute(
        """
        INSERT INTO pairwise_cache (
          key_hash, model, prompt_template_slug, template_hash, attribute_id, attribute_prompt_hash,
          entity_a_id, entity_b_id, entity_a_hash, entity_b_hash,
          higher_ranked, ratio, confidence, refused,
          input_tokens, output_tokens, provider_cost_nanodollars,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(key_hash) DO UPDATE SET
          higher_ranked = excluded.higher_ranked,
          ratio = excluded.ratio,
          confidence = excluded.confidence,
          refused = excluded.refused,
          input_tokens = excluded.input_tokens,
          output_tokens = excluded.output_tokens,
          provider_cost_nanodollars = excluded.provider_cost_nanodollars,
          updated_at = excluded.updated_at
        """,
        (
            cache["cache_key_hash"],
            cache["model"],
            cache["prompt_template_slug"],
            cache["template_hash"],
            record["attribute"]["id"],
            cache["attribute_prompt_hash"],
            record["entity_a"]["id"],
            record["entity_b"]["id"],
            cache["entity_a_hash"],
            cache["entity_b_hash"],
            prediction.get("higher_ranked"),
            prediction.get("ratio"),
            prediction.get("confidence"),
            1 if prediction["refused"] else 0,
            input_tokens,
            output_tokens,
            provider_cost_nanodollars,
            now,
            now,
        ),
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install training/gx10_pairwise/pyproject.toml first."
        ) from exc

    try:
        from peft import AutoPeftModelForCausalLM
    except ImportError:
        AutoPeftModelForCausalLM = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    adapter_config = Path(args.model) / "adapter_config.json"
    if adapter_config.exists() and AutoPeftModelForCausalLM is not None:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    return model, tokenizer, torch


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.prompt_grid)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No prompt records found.")

    preview(records)
    if args.dry_run:
        return

    model, tokenizer, torch = load_model_and_tokenizer(args)
    conn = sqlite3.connect(args.cache_db)
    init_cache(conn)

    predictions_handle = None
    if args.predictions_out is not None:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = args.predictions_out.open("w", encoding="utf-8")

    metrics = {
        "records": 0,
        "refusals": 0,
        "parse_refusals": 0,
        "snapped_to_ladder": 0,
        "winner_exact": 0,
        "ratio_exact": 0,
        "exact_match": 0,
        "labeled_observations": 0,
        "labeled_refusals": 0,
    }
    pair_signatures: dict[str, list[tuple[Any, ...]]] = {}

    try:
        for idx, record in enumerate(records, start=1):
            prompt_text = render_chat_text(
                record["messages"],
                tokenizer,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(prompt_text, return_tensors="pt")
            device = next(model.parameters()).device
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
            input_tokens = int(model_inputs["input_ids"].shape[1])

            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_tokens = generated[0][input_tokens:]
            output_tokens = int(generated_tokens.shape[0])
            raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            prediction, parse_error, snapped = parse_pairwise_response(
                raw_output,
                snap_to_ladder=not args.no_snap_to_ladder,
            )
            provider_cost_nanodollars = estimate_cost_nanodollars(
                input_tokens,
                output_tokens,
                args.input_cost_per_million_tokens,
                args.output_cost_per_million_tokens,
            )
            upsert_cache_row(
                conn,
                record,
                prediction,
                input_tokens,
                output_tokens,
                provider_cost_nanodollars,
            )

            metrics["records"] += 1
            if prediction["refused"]:
                metrics["refusals"] += 1
            if parse_error is not None:
                metrics["parse_refusals"] += 1
            if snapped:
                metrics["snapped_to_ladder"] += 1

            target = record.get("target")
            if target is not None:
                if target.get("refused", False):
                    metrics["labeled_refusals"] += 1
                else:
                    metrics["labeled_observations"] += 1
                    if not prediction["refused"]:
                        if prediction["higher_ranked"] == target.get("higher_ranked"):
                            metrics["winner_exact"] += 1
                        if abs(prediction["ratio"] - float(target.get("ratio"))) < 1e-9:
                            metrics["ratio_exact"] += 1
                        if (
                            prediction["higher_ranked"] == target.get("higher_ranked")
                            and abs(prediction["ratio"] - float(target.get("ratio"))) < 1e-9
                        ):
                            metrics["exact_match"] += 1

            signature = canonical_signature(
                prediction,
                swapped=bool(record["presentation"]["swapped"]),
            )
            pair_signatures.setdefault(record["pair_id"], []).append(signature)

            if predictions_handle is not None:
                json.dump(
                    {
                        "prompt_index": record.get("prompt_index"),
                        "pair_id": record["pair_id"],
                        "raw_output": raw_output,
                        "prediction": prediction,
                        "parse_error": parse_error,
                    },
                    predictions_handle,
                )
                predictions_handle.write("\n")

            if idx % 10 == 0 or idx == len(records):
                print(f"[seed-cache] processed {idx}/{len(records)}")
    finally:
        conn.commit()
        conn.close()
        if predictions_handle is not None:
            predictions_handle.close()

    comparable_pairs = 0
    swap_consistent_pairs = 0
    for signatures in pair_signatures.values():
        observations = [signature for signature in signatures if signature[0] == "observation"]
        if len(observations) >= 2:
            comparable_pairs += 1
            if len(set(observations)) == 1:
                swap_consistent_pairs += 1

    print(
        "[seed-cache] "
        f"wrote {metrics['records']} cache rows to {args.cache_db} "
        f"(refusals={metrics['refusals']}, parse_refusals={metrics['parse_refusals']}, "
        f"snapped={metrics['snapped_to_ladder']})"
    )
    if metrics["labeled_observations"] > 0:
        print(
            "[seed-cache] labeled accuracy "
            f"winner={metrics['winner_exact'] / metrics['labeled_observations']:.3f} "
            f"ratio={metrics['ratio_exact'] / metrics['labeled_observations']:.3f} "
            f"exact={metrics['exact_match'] / metrics['labeled_observations']:.3f}"
        )
    if comparable_pairs > 0:
        print(
            "[seed-cache] swap_consistency="
            f"{swap_consistent_pairs / comparable_pairs:.3f} "
            f"({swap_consistent_pairs}/{comparable_pairs})"
        )


if __name__ == "__main__":
    main()
