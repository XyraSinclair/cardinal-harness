"""Baseline methods M1-M3 for the weak-judge amplification benchmark.

M1: pointwise direct estimate (one call per item, numeric answer)
M2: pointwise 0-100 populousness score (one call per item)
M3: listwise one-shot full ordering (one call, one retry on malformed)

Writes baselines-<judge-short>.json with raw responses + parsed values.
Run: vrun python3 baselines.py openai/gpt-5.4-mini
     vrun python3 baselines.py openai/gpt-5.4-nano
"""
import json, os, re, sys, urllib.request

JUDGE = sys.argv[1]
SHORT = JUDGE.rsplit("/", 1)[1]
KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
ITEMS = [it["id"] for it in json.load(open("corpus.json"))]

def call(prompt, max_tokens=2000):
    body = {"model": JUDGE, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "usage": {"include": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    return (resp["choices"][0]["message"]["content"],
            (resp.get("usage") or {}).get("cost", 0.0))

def last_number(text):
    m = re.findall(r"-?[\d][\d,]*\.?\d*", text.replace(",", ""))
    return float(m[-1]) if m else None

out = {"judge": JUDGE, "cost": 0.0, "m1": {}, "m2": {}, "m3": None, "failures": []}

for item in ITEMS:  # M1
    text, cost = call(
        f"Estimate the current national population of {item}. Think briefly if "
        f"needed, then end your answer with the estimate as a plain integer.")
    out["cost"] += cost
    val = last_number(text)
    if val is None or val <= 0:
        out["failures"].append(("m1", item, text[:200]))
    else:
        out["m1"][item] = {"estimate": val, "raw": text[-200:]}

for item in ITEMS:  # M2
    text, cost = call(
        f"On a scale of 0 to 100, score how populous {item} is "
        f"(0 = least populous country imaginable, 100 = most populous country "
        f"in the world). End your answer with the score as a plain number.")
    out["cost"] += cost
    val = last_number(text)
    if val is None:
        out["failures"].append(("m2", item, text[:200]))
    else:
        out["m2"][item] = {"score": val, "raw": text[-200:]}

listing = "\n".join(f"- {it}" for it in ITEMS)  # M3
m3_prompt = (
    "Sort the following countries from MOST populous to LEAST populous "
    "(current national population). Output ONLY the sorted list, one country "
    f"per line, exact names as given, all {len(ITEMS)} of them, no numbering, "
    f"no commentary.\n\n{listing}")
for attempt in range(2):
    text, cost = call(m3_prompt)
    out["cost"] += cost
    lines = [l.strip("-• \t") for l in text.strip().splitlines() if l.strip()]
    matched = [l for l in lines if l in ITEMS]
    if len(matched) == len(ITEMS) and len(set(matched)) == len(ITEMS):
        out["m3"] = {"order": matched, "attempt": attempt + 1}
        break
    out["failures"].append(("m3", f"attempt{attempt+1}", text[:300]))

path = f"baselines-{SHORT}.json"
json.dump(out, open(path, "w"), indent=1)
print(f"{SHORT}: m1 {len(out['m1'])}/23  m2 {len(out['m2'])}/23  "
      f"m3 {'ok' if out['m3'] else 'FAILED'}  cost=${out['cost']:.4f}  "
      f"failures={len(out['failures'])}")
