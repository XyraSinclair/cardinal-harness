#!/usr/bin/env python3
"""Compose the AHP importance campaign + polarity axis + per-proposal attribute
scores into per-goal grant rankings for openpriors.com/manifund.

Three ledger-derived ingredients, all from ratiometer.judgments on colo2:

  importance_G(a)  mean signed log2 ratio margin of attribute a under grant
                   goal G (run_tag ahp-grant-goals-*), one value per goal —
                   how much goal G weights attribute a. A ledger proxy for
                   cardinal's IRLS priority vector; monotone-related, durable,
                   reproducible. (The exact IRLS solve lives in cardinal's
                   per-goal a00N.json; this stays ledger-only for one spine.)
  polarity(a)      mean signed log2 margin under the polarity axis
                   (run_tag ahp-polarity-*), centered so mid=0: >0 good high
                   pole, <0 bad high pole.
  attr(a, p)       how much proposal p exhibits attribute a (the four manifund
                   judge run tags), signed log2 margin.

grant_score_G(p) = sum_a  w_G(a) * orient(a) * attr(a, p)
  w_G(a)   = importance_G(a) shifted to >=0 and L1-normalized over attributes
             the goal judged (a nonneg weight per goal).
  orient(a)= tanh(polarity(a) / scale) in [-1, 1] — soft sign; goal-irrelevant
             attributes get muted by small w_G anyway.

A goal is included only when its pass is essentially complete (>= MIN_FRAC of
budget landed); until then the block is absent and the page shows what exists.

Usage: python3 scripts/ahp_grant_scores.py [out.json]
Default out: site/data/manifund_ahp.json (served CORS-open from llmsorting.com,
same lane as manifund.json).
"""
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH_LOCAL = "/data/clickhouse-twitter-lab/bin/clickhouse"
IMPORTANCE_TAG = "ahp-grant-goals-gemma31b-2026-08-16"
POLARITY_TAG = "ahp-polarity-gemma31b-2026-08-16"
PROPOSAL_TAGS = [
    "fable1000-gemma31b-2026-08-16",
    "fable1000-40pool-seed2-gemma31b",
    "manifund-relentless-2026-08-15",
    "manifund-gemini-flash-2026-08-15",
]
BUDGET = 12000
WEIGHT_TEMP = 0.30  # softmax temperature concentrating per-goal importance onto
                    # each goal's distinctive attributes (see compose loop); lower
                    # = sharper goal-conditioning. 0.30 gives an interpretable,
                    # non-degenerate spread.
MIN_FRAC = 0.6  # landing is atomic per goal; usable rows are ~70-80% of budget
                # after FINAL dedup + refusal/error filtering, so 0.6*budget=7200
                # cleanly separates a landed goal (~8.3-9.7k) from an unlanded one (0)


def ch_query(query):
    query = " ".join(query.split())
    if os.path.exists(CH_LOCAL):
        cmd = [CH_LOCAL, "client", "--port", "19000", "-q", query]
    else:
        cmd = ["ssh", "colo2", CH_LOCAL + " client --port 19000 -q " + repr(query)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"clickhouse query failed: {proc.stderr.decode()[:500]}")
    return proc.stdout.decode()


def margin_rows(run_tag, group_attr=True):
    """Per (attribute, entity) mean signed log2 ratio margin for a run tag."""
    key = "attribute, entity" if group_attr else "entity"
    q = f"""
SELECT {key}, round(avg(m), 4) AS score, count() AS n
FROM (
  SELECT attribute,
         arrayJoin([(entity_a, if(higher_ranked = 'A', 1., -1.)),
                    (entity_b, if(higher_ranked = 'B', 1., -1.))]) AS t,
         t.1 AS entity, t.2 * log2(greatest(ratio, 1.)) AS m
  FROM ratiometer.judgments FINAL
  WHERE run_tag = '{run_tag}' AND higher_ranked IN ('A', 'B') AND error = ''
)
GROUP BY {key}
FORMAT JSONEachRow"""
    return [json.loads(l) for l in ch_query(q).splitlines() if l.strip()]


def landed_per_attr(run_tag):
    q = f"""SELECT attribute, count() AS n FROM ratiometer.judgments FINAL
            WHERE run_tag = '{run_tag}' AND higher_ranked IN ('A','B') AND error = ''
            GROUP BY attribute FORMAT JSONEachRow"""
    return {r["attribute"]: r["n"] for r in
            (json.loads(l) for l in ch_query(q).splitlines() if l.strip())}


def goal_slug(attribute_line):
    return attribute_line.split(":", 1)[0].strip()


def attr_name(entity_line):
    return entity_line.split(":", 1)[0].strip()


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "site/data/manifund_ahp.json")

    corpus = [l.rstrip("\n") for l in open(os.path.join(REPO, "data/ahp_attributes.txt")) if l.strip()]
    name_of = {line: attr_name(line) for line in corpus}

    # Proposals, in the manifund.txt corpus order the page uses.
    proposals = [attr_name(l) if False else l.split(" — ")[0].strip()
                 for l in open(os.path.join(REPO, "data/manifund.txt")) if l.strip()]

    # --- importance per goal (each goal is one "attribute" line) ---
    imp_counts = landed_per_attr(IMPORTANCE_TAG)
    imp_by_goal = {}
    for row in margin_rows(IMPORTANCE_TAG):
        goal_line = row["attribute"]
        if imp_counts.get(goal_line, 0) < MIN_FRAC * BUDGET:
            continue  # goal not yet complete
        imp_by_goal.setdefault(goal_line, {})[row["entity"]] = row["score"]

    # --- polarity axis ---
    pol_counts = landed_per_attr(POLARITY_TAG)
    pol = {}
    pol_ready = False
    for row in margin_rows(POLARITY_TAG):
        pol[row["entity"]] = row["score"]
    if pol and next(iter(pol_counts.values()), 0) >= MIN_FRAC * BUDGET:
        pol_ready = True
    # center polarity on its median; scale by MAD for the tanh orientation
    if pol:
        vals = sorted(pol.values())
        med = vals[len(vals) // 2]
        mad = sorted(abs(v - med) for v in pol.values())[len(vals) // 2] or 1.0
        orient = {e: math.tanh((v - med) / (1.5 * mad)) for e, v in pol.items()}
    else:
        orient = {}
    # orientation keyed by short attribute name for the O(1) compose join
    orient_by_name = {name_of.get(e, attr_name(e)): o for e, o in orient.items()}

    # --- per-proposal attribute scores (attr line -> proposal -> score) ---
    tags_or = " OR ".join(f"run_tag = '{t}'" for t in PROPOSAL_TAGS)
    q = f"""
SELECT attribute, entity, round(avg(m), 4) AS score
FROM (
  SELECT attribute,
         arrayJoin([(entity_a, if(higher_ranked='A',1.,-1.)),
                    (entity_b, if(higher_ranked='B',1.,-1.))]) AS t,
         t.1 AS entity, t.2 * log2(greatest(ratio,1.)) AS m
  FROM ratiometer.judgments FINAL
  WHERE ({tags_or}) AND higher_ranked IN ('A','B') AND error = ''
)
GROUP BY attribute, entity FORMAT JSONEachRow"""
    prop_attr = {}
    for r in (json.loads(l) for l in ch_query(q).splitlines() if l.strip()):
        # proposal entities are stored as full "title — subtitle" text (and vary
        # by corpus across the 4 judge tags); key by the short-title prefix so the
        # join with the manifund.txt proposal list is corpus-independent.
        p_key = r["entity"].split(" — ")[0].strip()
        prop_attr.setdefault(r["attribute"], {})[p_key] = r["score"]
    # index proposal attr scores by short attribute name for join with corpus
    prop_by_name = {}
    for a_line, m in prop_attr.items():
        prop_by_name[attr_name(a_line)] = m

    # --- compose per-goal grant rankings ---
    goals_out = []
    for goal_line, weights in imp_by_goal.items():
        # Softmax-concentrated weights over the goal's judged attributes.
        # L1-normalizing shifted margins over ~1010 attrs yields a near-uniform
        # vector that WASHES OUT the goal-conditioning (all goals rank proposals
        # near-identically). Softmax at WEIGHT_TEMP concentrates weight on each
        # goal's distinctive top attributes; measured 2026-08-17: recovers a real
        # ~30-position swing (careful_generalist top-10 -> nearterm_welfare last
        # for scry) that flat weighting collapsed to 34-36. Top-40 attr overlap
        # between goals is only 3-18% Jaccard, so the goals are genuinely distinct.
        items = [(name_of.get(line, attr_name(line)), s) for line, s in weights.items()]
        mx = max((s for _, s in items), default=0.0)
        ex = [(n, math.exp((s - mx) / WEIGHT_TEMP)) for n, s in items]
        z = sum(e for _, e in ex) or 1.0
        w = {n: e / z for n, e in ex}
        # grant score per proposal
        scores = {}
        for p in proposals:
            acc = 0.0
            for n, wn in w.items():
                pa = prop_by_name.get(n, {}).get(p)
                if pa is None:
                    continue
                acc += wn * orient_by_name.get(n, 0.0) * pa
            scores[p] = round(acc, 4)
        top_attrs = sorted(w.items(), key=lambda kv: -kv[1])[:15]
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        goals_out.append({
            "goal": goal_slug(goal_line),
            "goalText": goal_line.split(":", 1)[1].strip() if ":" in goal_line else goal_line,
            "landed": imp_counts.get(goal_line, 0),
            "topAttributes": [{"a": n, "w": round(wn, 4)} for n, wn in top_attrs],
            "ranking": [{"proposal": p, "score": s} for p, s in ranked],
        })

    out = {
        "importanceTag": IMPORTANCE_TAG,
        "polarityTag": POLARITY_TAG,
        "polarityReady": pol_ready,
        "goalsComplete": len(goals_out),
        "goals": goals_out,
        "polarity": sorted(
            ({"a": name_of.get(e, attr_name(e)), "score": v, "orient": round(orient.get(e, 0.0), 3)}
             for e, v in pol.items()),
            key=lambda r: -r["score"],
        ) if pol_ready else [],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{out_path}: {len(goals_out)}/7 goals complete, polarity ready={pol_ready}, "
          f"{os.path.getsize(out_path)} bytes")



if __name__ == "__main__":
    main()
