"""OpenRouter logprob-unlock census, 2026-07-19.

For each (model, reasoning-condition) cell: send the standard pairwise ratio prompt with
logprobs=true, top_logprobs=5, and record whether token logprobs come back, from which
provider, at what cost. Repeat cells n=10 for the reliability table in findings.md.

Usage: OPENROUTER_API_KEY=... python3 probe_openrouter_unlock.py
"""
import collections
import json
import os
import urllib.error
import urllib.request

KEY = os.environ["OPENROUTER_API_KEY"]
SYS = 'Output only JSON {"higher_ranked": "A"|"B", "ratio": number}.'
USR = (
    "Compare by mass.\n<entity_A>a chicken egg</entity_A>\n"
    "<entity_B>a bowling ball</entity_B>\nJSON:"
)

MODELS = ["openai/gpt-5.4-mini", "openai/gpt-5.4", "openai/gpt-5.5", "openai/gpt-5.6-sol"]
CONDITIONS = [
    ("control", {}),
    ("effort_none", {"reasoning": {"effort": "none"}}),
    ("enabled_false", {"reasoning": {"enabled": False}}),
]
REPEATS = 10


def post(body):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"error": {"message": e.read().decode()[:160]}}


total = 0.0
for model in MODELS:
    for tag, extra in CONDITIONS:
        tally = collections.Counter()
        provs = collections.Counter()
        for _ in range(REPEATS):
            code, r = post(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYS},
                        {"role": "user", "content": USR},
                    ],
                    "max_tokens": 2000,
                    "logprobs": True,
                    "top_logprobs": 5,
                    "usage": {"include": True},
                    **extra,
                }
            )
            if code != 200 or r.get("error"):
                tally[f"fail_{code}"] += 1
                continue
            lp = r["choices"][0].get("logprobs")
            total += (r.get("usage") or {}).get("cost") or 0
            provs[r.get("provider")] += 1
            tally["logprobs" if (lp and lp.get("content")) else "no_logprobs"] += 1
        print(f"{model:22s} {tag:13s} n={REPEATS} -> {dict(tally)} providers={dict(provs)}")
print(f"TOTAL ${total:.5f}")
