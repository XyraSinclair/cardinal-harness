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
  w_G(a)   = 2^importance_G(a), L1-normalized — margins are log2 ratios, so
             exponentiation restores the elicited ratio scale. Parameter-free.
  orient(a)= tanh(polarity(a) - median) in [-1, 1] — soft sign in units of
             doublings; goal-irrelevant attributes get muted by small w_G.

A goal is included only when its pass is essentially complete (>= MIN_FRAC of
budget landed); until then the block is absent and the page shows what exists.

Usage: python3 scripts/ahp_grant_scores.py [out.json]
Default out: site/data/manifund_ahp.json (served CORS-open from llmsorting.com,
same lane as manifund.json).
"""
import json
import math
import os
import shutil
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
# Unit discipline (operator standard 2026-08-17): everything is elicited ONLY
# through pairwise ratio comparisons on a geometric ladder; margins are log2
# ratios. Weights restore the elicited units — w = 2^margin, normalized — with
# ZERO free parameters. No temperature, no Likert, no fitted constants.
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
    # center polarity on its median for the soft-sign orientation
    if pol:
        vals = sorted(pol.values())
        med = vals[len(vals) // 2]
        # Soft sign, no fitted constants: unit = one doubling (1 log2 margin).
        # An attribute judged 2x more positive than the median gets orient 0.76.
        orient = {e: math.tanh(v - med) for e, v in pol.items()}
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
        # Unit-honest AHP weights: margins are log2 of elicited importance
        # ratios, so the ratio-scale priority is 2^margin; normalize to sum 1.
        # No sharpening. Measured consequence (2026-08-17): with honest units the
        # composed proposal ranking shifts only modestly across goals — judged
        # importance is broad (top-40 attrs hold ~9% of mass) and proposal
        # exhibition profiles are halo-correlated, so a general-quality factor
        # dominates. The goal-conditioning lives in the importance vectors
        # themselves (per-goal top attributes overlap 3-18% Jaccard), which the
        # page surfaces directly.
        items = [(name_of.get(line, attr_name(line)), s) for line, s in weights.items()]
        ex = [(n, 2.0 ** s) for n, s in items]
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
            "_w": w,  # full weight dict, consumed by the interior block below
        })

    # --- AHP interior: full interpretability, all derived with zero parameters ---
    # One aligned attribute index shared by every vector on the page.
    attr_index = sorted({n for g in goals_out for n in g["_w"]})
    pos = {n: i for i, n in enumerate(attr_index)}
    for g in goals_out:
        wv = [0.0] * len(attr_index)
        for n, wn in g["_w"].items():
            wv[pos[n]] = round(wn, 8)
        g["w"] = wv
        # Per-proposal score decomposition: every attribute's contribution is
        # w * orient * exhibition; ship the top 14 by |contribution| plus the
        # residual sum so the shown terms visibly reconstruct the total.
        contribs = {}
        for p in proposals:
            terms = []
            for n, wn in g["_w"].items():
                pa = prop_by_name.get(n, {}).get(p)
                if pa is None:
                    continue
                c = wn * orient_by_name.get(n, 0.0) * pa
                terms.append((pos[n], c, pa))
            terms.sort(key=lambda t: -abs(t[1]))
            top = terms[:14]
            rest = sum(c for _, c, _ in terms[14:])
            contribs[p] = {
                "t": [[i, round(c, 6), round(pa, 3)] for i, c, pa in top],
                "rest": round(rest, 6),
            }
        g["contribs"] = contribs
        del g["_w"]

    orient_v = [round(orient_by_name.get(n, 0.0), 4) for n in attr_index]
    pol_v = [round(next((v for e, v in pol.items()
                         if name_of.get(e, attr_name(e)) == n), 0.0), 4)
             for n in attr_index]

    def pearson(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        sy = math.sqrt(sum((b - my) ** 2 for b in ys))
        return sxy / (sx * sy) if sx and sy else 0.0

    def spearman_of_rankings(ra, rb):
        pa = {r["proposal"]: i for i, r in enumerate(ra)}
        pb = {r["proposal"]: i for i, r in enumerate(rb)}
        shared = [p for p in pa if p in pb]
        return pearson([pa[p] for p in shared], [pb[p] for p in shared])

    slugs = [g["goal"] for g in goals_out]
    n_g = len(goals_out)
    rank_corr = [[round(spearman_of_rankings(goals_out[i]["ranking"], goals_out[j]["ranking"]), 3)
                  for j in range(n_g)] for i in range(n_g)]
    # weight vectors compared in log space (they live on a ratio scale)
    logw = [[math.log2(x) if x > 0 else -30.0 for x in g["w"]] for g in goals_out]
    weight_corr = [[round(pearson(logw[i], logw[j]), 3) for j in range(n_g)] for i in range(n_g)]
    offdiag = [rank_corr[i][j] for i in range(n_g) for j in range(n_g) if i < j]
    mean_rank_corr = round(sum(offdiag) / len(offdiag), 3) if offdiag else None

    out = {
        "importanceTag": IMPORTANCE_TAG,
        "attrs": attr_index,
        "orient": orient_v,
        "polarityMargins": pol_v,
        "corr": {"goals": slugs, "rank": rank_corr, "weight": weight_corr,
                 "meanRankCorr": mean_rank_corr},
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
    # Publish to the live site when the served dir exists (colo2). Without this
    # the page's fetch 404s and the whole AHP lens silently hides (2026-08-19).
    publish = os.path.join("/srv/llmsorting/data", os.path.basename(out_path))
    if os.path.isdir(os.path.dirname(publish)) and os.access(os.path.dirname(publish), os.W_OK):
        shutil.copyfile(out_path, publish)
        print(f"published -> {publish}")



if __name__ == "__main__":
    main()
