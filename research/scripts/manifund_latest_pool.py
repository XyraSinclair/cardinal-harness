#!/usr/bin/env python3
"""Build data/manifund_latest.txt: every non-dummy Manifund proposal created
since CUTOFF, newest-first, as `title — blurb` corpus lines. Lines that are
already in data/manifund.txt (the 2026-08-16 40-pool) are reused verbatim so
the gemma/gemini judgments on that pool keep joining. Also writes
data/manifund/projects_recent.jsonl for manifund_page_data.py card joins."""
import json, pathlib, urllib.request
REPO = pathlib.Path(__file__).resolve().parent.parent
CUTOFF = "2026-08-08"
UA = "llmsorting-research (contact: xyraward@gmail.com; approved by Austin)"
BASE = "https://manifund.org/api/v0/projects"
rows, before = [], None
while True:
    url = BASE + (f"?before={before}" if before else "")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        batch = json.load(r)
    if not batch: break
    rows += batch
    before = batch[-1]["created_at"]
    if before < CUTOFF: break
recent = [p for p in rows if p["created_at"] >= CUTOFF and p.get("type") != "dummy"]
old = [l.rstrip("\n") for l in (REPO / "data/manifund.txt").open() if l.strip()]
def line(p):
    t = " ".join((p["title"] or "").split()); b = " ".join((p.get("blurb") or "").split())
    return f"{t} — {b}" if b else t
out, seen = [], set()
for p in recent:
    l = line(p)
    match = next((o for o in old if o.startswith(p["title"].strip())), None)
    l = match or l
    if l in seen: continue
    seen.add(l); out.append(l)
for o in old:  # keep renamed/delisted 40-pool lines so their judgments still join
    if o not in seen: seen.add(o); out.append(o)
(REPO / "data/manifund_latest.txt").write_text("\n".join(out) + "\n")
(REPO / "data/manifund").mkdir(exist_ok=True)
with (REPO / "data/manifund/projects_recent.jsonl").open("w") as f:
    for p in recent: f.write(json.dumps(p) + "\n")
print(f"recent={len(recent)} lines={len(out)} reused_old={sum(1 for o in old if o in seen)}/{len(old)}")
