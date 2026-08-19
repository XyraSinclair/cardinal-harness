#!/usr/bin/env python3
"""Generate the openpriors.com/manifund page dataset from the ratiometer
ledger on colo2.

Per (model, attribute, proposal) score = mean signed log2 ratio margin over
all landed pairwise judgments (winner +log2(ratio), loser -log2(ratio)),
deduped through the ReplacingMergeTree with FINAL. Proposal card metadata
joins data/manifund.txt corpus lines back to data/manifund/projects.jsonl.

Usage: python3 scripts/manifund_page_data.py [out.json]
Default out: site/data/manifund.json — served at
https://llmsorting.com/data/manifund.json (rsync site/ to
colo2:/srv/llmsorting), which openpriors.com/manifund fetches cross-origin
(CORS header set in /etc/caddy/llmsorting.caddy). Refresh = regenerate,
commit, rsync; no exopriors-core deploy needed.
"""
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH_LOCAL = "/data/clickhouse-twitter-lab/bin/clickhouse"
RUN_TAGS = [
    "fable1000-gemma31b-2026-08-16",
    "fable1000-40pool-seed2-gemma31b",
    "manifund-relentless-2026-08-15",
    "manifund-gemini-flash-2026-08-15",
]


def ch_query(query):
    # Single-line the SQL: repr()'s \n escapes arrive at the remote shell as
    # literal backslash-n and ClickHouse silently returns nothing (rc 0).
    query = " ".join(query.split())
    if os.path.exists(CH_LOCAL):
        cmd = [CH_LOCAL, "client", "--port", "19000", "-q", query]
    else:
        cmd = ["ssh", "colo2", CH_LOCAL + " client --port 19000 -q " + repr(query)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"clickhouse query failed: {proc.stderr.decode()[:500]}")
    return proc.stdout.decode()


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "site/data/manifund.json")

    corpus_lines = [l.rstrip("\n") for l in open(os.path.join(REPO, "data/manifund.txt")) if l.strip()]
    fable_names = {l.strip() for l in open(os.path.join(REPO, "batteries/fable_subtle_1000.txt")) if l.strip()}

    # Join corpus lines back to full Manifund records. The 40-pool was drawn
    # from the newest slice of Manifund (2026-08), so the fresh scrape
    # (projects_recent.jsonl, API v0, 2026-08-16) is the primary source;
    # projects.jsonl (2026-07-13 snapshot) is the fallback. A couple of pool
    # proposals were renamed or delisted after the pool was drawn — those get
    # cards synthesized from the corpus line itself, with no external link.
    records = [json.loads(l) for l in open(os.path.join(REPO, "data/manifund/projects_recent.jsonl"))]
    records += [json.loads(l) for l in open(os.path.join(REPO, "data/manifund/projects.jsonl"))]
    proposals = []
    unmatched = 0
    for line in corpus_lines:
        hits = [r for r in records if line.startswith(r["title"].strip())]
        rec = hits[0] if hits else None
        if rec is None:
            unmatched += 1
            title, _, blurb = line.partition(" — ")
            proposals.append({
                "title": title, "blurb": blurb, "slug": "", "creator": "",
                "username": "", "goal": 0, "minFunding": 0, "raised": 0,
                "offered": 0, "stage": "", "causes": [], "created": "",
            })
            continue
        raised = sum(t.get("amount", 0) for t in rec.get("txns", []) if t.get("token") in (None, "USD"))
        offered = sum(b.get("amount", 0) for b in rec.get("bids", []) if b.get("status") == "pending")
        profiles = rec.get("profiles") or {}
        proposals.append({
            "title": rec["title"],
            "blurb": rec.get("blurb") or "",
            "slug": rec.get("slug") or "",
            "creator": profiles.get("full_name") or profiles.get("username") or "",
            "username": profiles.get("username") or "",
            "goal": rec.get("funding_goal") or 0,
            "minFunding": rec.get("min_funding") or 0,
            "raised": raised,
            "offered": offered,
            "stage": rec.get("stage") or "",
            "causes": [c.get("title", "") for c in rec.get("causes") or []],
            "created": (rec.get("created_at") or "")[:10],
        })
    if unmatched:
        print(f"note: {unmatched} pool proposals not found in Manifund records "
              f"(renamed or delisted) — cards synthesized from corpus lines", file=sys.stderr)
    entity_index = {line: i for i, line in enumerate(corpus_lines)}

    tags_sql = ", ".join("'" + t.replace("\\", "\\\\").replace("'", "\\'") + "'" for t in RUN_TAGS)
    query = f"""
SELECT model, attribute, entity,
       round(avg(m), 3) AS score, count() AS n, round(avg(dp), 3) AS dprob
FROM (
  SELECT model, attribute, dir_prob AS dp,
         arrayJoin([(entity_a, if(higher_ranked = 'A', 1., -1.)),
                    (entity_b, if(higher_ranked = 'B', 1., -1.))]) AS t,
         t.1 AS entity, t.2 * log2(greatest(ratio, 1.)) AS m
  FROM ratiometer.judgments FINAL
  WHERE run_tag IN ({tags_sql}) AND higher_ranked IN ('A', 'B') AND error = ''
)
GROUP BY model, attribute, entity
FORMAT JSONEachRow"""
    rows = [json.loads(l) for l in ch_query(query).splitlines() if l.strip()]

    # attribute -> model -> {n, dp, scores[40]}
    attrs = {}
    models = set()
    skipped = 0
    total_judgments = 0
    for r in rows:
        idx = entity_index.get(r["entity"])
        if idx is None:
            skipped += 1
            continue
        models.add(r["model"])
        cell = attrs.setdefault(r["attribute"], {}).setdefault(
            r["model"], {"n": 0, "_dp": 0.0, "s": [None] * len(corpus_lines)})
        cell["s"][idx] = r["score"]
        cell["n"] += r["n"]
        cell["_dp"] += r["dprob"] * r["n"]
    if skipped:
        print(f"warning: {skipped} score rows had entities outside the corpus", file=sys.stderr)

    attr_list = []
    for name in sorted(attrs):
        entry = {"a": name, "b": "fable" if name in fable_names else "classic", "m": {}}
        for model, cell in attrs[name].items():
            n = cell["n"]
            total_judgments += n // 2  # each judgment contributed two entity rows
            entry["m"][model] = {"n": n // 2,
                                 "dp": round(cell["_dp"] / n, 3) if n else 0.0,
                                 "s": cell["s"]}
        attr_list.append(entry)

    out = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus": "manifund.txt",
        "runTags": RUN_TAGS,
        "judgments": total_judgments,
        "models": sorted(models),
        "proposals": proposals,
        "attributes": attr_list,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(out_path)
    print(f"{out_path}: {len(proposals)} proposals, {len(attr_list)} attributes, "
          f"{len(out['models'])} models, {total_judgments} judgments, {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
