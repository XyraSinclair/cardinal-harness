#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal LoRA SFT trainer for exported pairwise ratio records."
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
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
    refusals = sum(1 for record in records if record["target"]["refused"])
    swapped = sum(1 for record in records if record["presentation"]["swapped"])
    models = sorted({record["metadata"]["model"] for record in records})
    print(f"records={len(records)} refusals={refusals} swapped={swapped}")
    print(f"models={', '.join(models)}")
    if records:
        sample = records[0]
        print("\n--- sample assistant target ---")
        print(sample["target"]["assistant_json"])
        print("\n--- sample rendered conversation ---")
        print(render_chat_text(sample["messages"]))


def split_eval(
    records: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_ratio <= 0.0 or len(records) < 2:
        return records, []
    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_records = shuffled[:eval_size]
    train_records = shuffled[eval_size:]
    return train_records, eval_records


def run_training(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install from "
            "training/gx10_pairwise/pyproject.toml first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_supervised_dataset(input_records: list[dict[str, Any]]) -> Dataset:
        dataset = Dataset.from_list(input_records)

        def tokenize_record(record: dict[str, Any]) -> dict[str, Any]:
            prompt_messages = record["messages"][:-1]
            prompt_text = render_chat_text(
                prompt_messages,
                tokenizer,
                add_generation_prompt=True,
            )
            full_text = render_chat_text(record["messages"], tokenizer)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=args.max_seq_length,
            )
            labels = list(tokens["input_ids"])
            supervised_prefix = min(len(prompt_ids), len(labels))
            for idx in range(supervised_prefix):
                labels[idx] = -100
            tokens["labels"] = labels
            return tokens

        return dataset.map(tokenize_record, remove_columns=dataset.column_names)

    if args.eval_data:
        train_records = records
        eval_records = load_jsonl(args.eval_data)
    else:
        train_records, eval_records = split_eval(records, args.eval_ratio, args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = to_supervised_dataset(train_records)
    eval_dataset = to_supervised_dataset(eval_records) if eval_records else None

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.train_data)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise SystemExit("No training records found.")

    preview(records)
    if args.dry_run:
        return

    run_training(args, records)


if __name__ == "__main__":
    main()
