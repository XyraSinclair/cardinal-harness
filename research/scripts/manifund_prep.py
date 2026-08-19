#!/usr/bin/env python3
"""Prepare Manifund judgment inputs from the raw corpus.

Reads data/manifund/projects.jsonl (see manifund_fetch.py) and emits, under
data/manifund/items/:

  <cohort>.json        [{id, text}] item files sized for pairwise prompts
  ground_truth.csv     id, slug, cohort flags, stage, funded, raised, goal,
                       min_funding, n_donors, created_at, causes

Cohorts:
  live        stage == proposal (the open slate)
  acx2024     ACX Grants 2024 cause (real human funding decisions as truth)
  eacc        EA Community Choice cause (crowd-matched funding as truth)
  funded      every project with >0 raised
  full        everything

Also regenerates data/manifund/requests/live-slate-4attr.json (multi-rerank
request over the live cohort; validate with `cardinal validate`).

Item text = title + blurb + funding ask + description (markdown stripped to
plain-ish text), truncated to TEXT_CAP chars with an explicit [truncated]
marker so judges know the denominator.
"""

import csv
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "manifund"
ITEMS = ROOT / "items"
TEXT_CAP = 8000

COHORT_CAUSES = {
    "acx2024": "ACX Grants 2024",
    "acx2025": "ACX Grants 2025",
    "eacc": "EA Community Choice",
}


def raised(p) -> float:
    return sum(
        t.get("amount", 0)
        for t in p.get("txns", [])
        if t.get("token") == "USD"
    )


def clean_markdown(md: str) -> str:
    text = md or ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> label
    text = re.sub(r"[#*_`>]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def item_text(p) -> str:
    head = [
        f"TITLE: {p['title']}",
        f"BLURB: {(p.get('blurb') or '').strip()}",
        f"ASK: min ${p.get('min_funding') or 0:,.0f} / goal ${p.get('funding_goal') or 0:,.0f}",
        f"CAUSES: {', '.join(c['title'] for c in p.get('causes', [])) or 'none listed'}",
        "",
    ]
    body = clean_markdown(p.get("description") or "")
    text = "\n".join(head) + body
    if len(text) > TEXT_CAP:
        text = text[:TEXT_CAP] + "\n[truncated]"
    return text


def main() -> None:
    projects = [json.loads(l) for l in (ROOT / "projects.jsonl").open()]
    projects = [p for p in projects if p.get("type") != "dummy"]
    ITEMS.mkdir(parents=True, exist_ok=True)

    cohorts: dict[str, list] = {"full": projects}
    cohorts["live"] = [p for p in projects if p["stage"] == "proposal"]
    cohorts["funded"] = [p for p in projects if raised(p) > 0]
    for key, cause in COHORT_CAUSES.items():
        cohorts[key] = [
            p for p in projects if any(c["title"] == cause for c in p["causes"])
        ]

    for name, ps in cohorts.items():
        items = [{"id": p["slug"], "text": item_text(p)} for p in ps]
        (ITEMS / f"{name}.json").write_text(json.dumps(items, indent=1))
        print(f"{name}: {len(items)} items")

    with (ROOT / "ground_truth.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "slug", "id", "stage", "type", "raised", "funding_goal",
                "min_funding", "n_txns", "n_bids", "created_at", "creator",
                "cohorts", "causes",
            ]
        )
        for p in projects:
            member = [k for k, ps in cohorts.items() if k != "full" and p in ps]
            w.writerow(
                [
                    p["slug"], p["id"], p["stage"], p["type"],
                    f"{raised(p):.2f}", p.get("funding_goal") or 0,
                    p.get("min_funding") or 0, len(p.get("txns", [])),
                    len(p.get("bids", [])), p["created_at"],
                    (p.get("profiles") or {}).get("username", ""),
                    "|".join(member),
                    "|".join(c["title"] for c in p.get("causes", [])),
                ]
            )
    print(f"ground truth -> {ROOT / 'ground_truth.csv'}")

    requests_dir = ROOT / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    live_request = {
        "entities": [
            {"id": p["slug"], "text": item_text(p)} for p in cohorts["live"]
        ],
        "attributes": [
            {
                "id": "theory_of_change",
                "prompt": "plausibility of the causal path from the proposed activities to the claimed impact",
                "weight": 0.3,
            },
            {
                "id": "impact_per_dollar",
                "prompt": "expected impact per marginal dollar at the stated minimum funding ask",
                "weight": 0.3,
            },
            {
                "id": "team_evidence",
                "prompt": "strength of verifiable track-record evidence that this team can execute this plan",
                "weight": 0.25,
            },
            {
                "id": "epistemic_integrity",
                "prompt": "epistemic integrity of the write-up: honest failure modes, quantified claims, falsifiable milestones",
                "weight": 0.15,
            },
        ],
        "topk": {"k": 20, "tolerated_error": 0.15},
        "gates": [],
        "comparison_budget": 1600,
        "rater_id": "manifund-live-slate-2026-07",
        "comparison_concurrency": 8,
        "max_pair_repeats": 1,
        "randomize_presentation_order": True,
    }
    req_path = requests_dir / "live-slate-4attr.json"
    req_path.write_text(json.dumps(live_request, indent=1))
    print(f"request -> {req_path}")


if __name__ == "__main__":
    main()
