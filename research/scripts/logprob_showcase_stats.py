#!/usr/bin/env python3
"""Compute the logprob consistency story from landed PMF judgments.

Input: TSV of ratiometer.judgments rows (run_tag, model, attribute, a_hash,
b_hash, higher_ranked, dir_prob, entropy, top_prob, ratio, swapped, cached).
Output: JSON of every statistic the showcase page renders.

The analyses:
  mirror      counterbalanced pair (same attribute+unordered pair, two
              presentation orders): canonical p = P(lower-hash entity wins).
              Perfect position-invariance = p_order1 == p_order2.
  reliability bin rows by stated confidence q = P(chosen side); y = fraction
              whose counterbalanced twin picked the same winner. References:
              y=q (self-trust line) and y=q^2+(1-q)^2 (what twin agreement
              would be if the model actually sampled from its stated PMF).
  crossjudge  same unordered pair judged by both dense models (highdim runs):
              agreement rate binned by the two judges' mean stated confidence.
  triads      per (model, attribute): majority direction per unordered pair;
              among triples of items with all 3 pairs observed, fraction cyclic.
  entropy     per-model distribution of judgment entropy (nats) and dir_prob.
"""
import collections, json, math, sys


def canonical(a_hash, b_hash, higher, dir_prob):
    lo, hi = sorted((a_hash, b_hash))
    winner = a_hash if higher == "A" else b_hash
    p_winner = dir_prob if dir_prob > 0 else 0.5
    return (lo, hi), (p_winner if winner == lo else 1.0 - p_winner), winner


def main():
    rows = []
    for line in open(sys.argv[1]):
        f = line.rstrip("\n").split("\t")
        if len(f) != 12:
            continue
        rows.append({
            "run": f[0], "model": f[1], "attr": f[2], "a": f[3], "b": f[4],
            "hr": f[5], "q": float(f[6]), "H": float(f[7]), "top": float(f[8]),
            "ratio": float(f[9]), "swapped": f[10] == "true" or f[10] == "1",
            "cached": f[11] == "true" or f[11] == "1",
        })
    models = sorted({r["model"] for r in rows})
    out = {"n_rows": len(rows), "models": models}

    # ---- group counterbalanced twins: (run, model, attr, unordered pair) ----
    groups = collections.defaultdict(list)
    for r in rows:
        key, p, winner = canonical(r["a"], r["b"], r["hr"], r["q"])
        groups[(r["run"], r["model"], r["attr"], key)].append(
            {"p": p, "q": r["q"], "winner": winner, "H": r["H"]})

    # mirror scatter + swap agreement + reliability, per model
    per_model = {}
    for m in models:
        mirror = []          # (p1, p2) canonical
        twin_agree = 0
        twin_total = 0
        dq_abs = []
        rel_bins = collections.defaultdict(lambda: [0, 0])  # bin -> [agree, n]
        for (run, model, attr, key), g in groups.items():
            if model != m or len(g) != 2:
                continue
            g1, g2 = g
            mirror.append((round(g1["p"], 4), round(g2["p"], 4)))
            agree = g1["winner"] == g2["winner"]
            twin_total += 1
            twin_agree += agree
            dq_abs.append(abs(g1["p"] - g2["p"]))
            for gg in g:
                b = min(int((gg["q"] - 0.5) / 0.05), 9)
                rel_bins[b][0] += agree
                rel_bins[b][1] += 1
        H_vals = [r["H"] for r in rows if r["model"] == m]
        q_vals = [r["q"] for r in rows if r["model"] == m]
        per_model[m] = {
            "twin_pairs": twin_total,
            "swap_agreement": twin_agree / twin_total if twin_total else None,
            "mean_abs_dp": sum(dq_abs) / len(dq_abs) if dq_abs else None,
            "median_abs_dp": sorted(dq_abs)[len(dq_abs) // 2] if dq_abs else None,
            "reliability": [
                {"q_mid": 0.525 + 0.05 * b, "agree": a / n, "n": n}
                for b, (a, n) in sorted(rel_bins.items()) if n >= 20],
            "mirror_sample": mirror[:4000],
            "entropy_hist": hist(H_vals, 0.0, 4.0, 40),
            "q_hist": hist(q_vals, 0.5, 1.0, 25),
            "mean_entropy": sum(H_vals) / len(H_vals),
            "n_judgments": len(H_vals),
        }
    out["per_model"] = per_model

    # ---- cross-judge (highdim runs, both models on same pairs) ----
    hd_runs = [r for r in rows if r["run"].startswith("manifund-highdim")]
    byjudge = collections.defaultdict(dict)  # (run,attr,key) -> model -> (meanp, winner_majority, meanq)
    pair_acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in hd_runs:
        key, p, winner = canonical(r["a"], r["b"], r["hr"], r["q"])
        pair_acc[(r["run"], r["attr"], key)][r["model"]].append((p, winner, r["q"]))
    cj_bins = collections.defaultdict(lambda: [0, 0])
    cj_agree = cj_total = 0
    for k, mm in pair_acc.items():
        if len(mm) != 2:
            continue
        verdicts = {}
        confs = []
        for m, obs in mm.items():
            meanp = sum(o[0] for o in obs) / len(obs)
            verdicts[m] = meanp >= 0.5
            confs.append(sum(o[2] for o in obs) / len(obs))
        agree = len(set(verdicts.values())) == 1
        cj_total += 1
        cj_agree += agree
        mc = sum(confs) / 2
        b = min(int((mc - 0.5) / 0.05), 9)
        cj_bins[b][0] += agree
        cj_bins[b][1] += 1
    out["crossjudge"] = {
        "pairs": cj_total,
        "agreement": cj_agree / cj_total if cj_total else None,
        "by_confidence": [
            {"q_mid": 0.525 + 0.05 * b, "agree": a / n, "n": n}
            for b, (a, n) in sorted(cj_bins.items()) if n >= 15],
    }

    # ---- transitivity: cyclic triads per model ----
    tri = {}
    for m in models:
        # per (run, attr): pair -> majority direction (winner hash by mean p)
        per_attr = collections.defaultdict(dict)
        for (run, model, attr, key), g in groups.items():
            if model != m:
                continue
            meanp = sum(x["p"] for x in g) / len(g)
            lo, hi = key
            per_attr[(run, attr)][key] = lo if meanp >= 0.5 else hi
        cyc = tot = 0
        for (run, attr), edges in per_attr.items():
            items = set()
            for (lo, hi) in edges:
                items.add(lo); items.add(hi)
            items = sorted(items)
            idx = {h: i for i, h in enumerate(items)}
            beats = {}
            for (lo, hi), w in edges.items():
                beats[(lo, hi)] = w
            keys = list(edges.keys())
            present = set(edges.keys())
            n = len(items)
            # enumerate triples that have all three edges observed
            from itertools import combinations
            for a, b, c in combinations(items, 3):
                e1, e2, e3 = tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))
                if e1 in present and e2 in present and e3 in present:
                    w1, w2, w3 = beats[e1], beats[e2], beats[e3]
                    wins = collections.Counter([w1, w2, w3])
                    # cyclic iff each item wins exactly once
                    cyclic = set(wins.values()) == {1}
                    tot += 1
                    cyc += cyclic
        tri[m] = {"triads": tot, "cyclic_frac": cyc / tot if tot else None}
    out["triads"] = tri
    json.dump(out, open(sys.argv[2], "w"))
    # human summary
    for m in models:
        pm = per_model[m]
        print(f"{m}: {pm['n_judgments']} judgments, {pm['twin_pairs']} twin pairs, "
              f"swap-agree {pm['swap_agreement']:.3f}, mean|Δp| {pm['mean_abs_dp']:.3f}, "
              f"H̄ {pm['mean_entropy']:.2f}, cyclic triads {tri[m]['cyclic_frac']:.3f} "
              f"of {tri[m]['triads']}")
    print(f"cross-judge: {out['crossjudge']['agreement']:.3f} over {out['crossjudge']['pairs']} pairs")


def hist(vals, lo, hi, nbins):
    counts = [0] * nbins
    for v in vals:
        b = min(max(int((v - lo) / (hi - lo) * nbins), 0), nbins - 1)
        counts[b] += 1
    return {"lo": lo, "hi": hi, "counts": counts}


if __name__ == "__main__":
    main()
