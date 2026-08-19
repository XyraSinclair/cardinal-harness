#!/usr/bin/env python3
"""Pilot map analysis: transmissibility, external validation, fused map."""
import json, sys, math
from collections import defaultdict

base = sys.argv[1] if len(sys.argv) > 1 else "."
runs = [json.loads(l) for l in open(f"{base}/latents.jsonl")]
ref = json.load(open(f"{base}/reference_scores.json"))
ENT = [json.loads(l) for l in open(f"{base}/entities.jsonl")]
text = {e["id"]: e["text"] for e in ENT}

def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else float("nan")

by_attr = defaultdict(dict)
for r in runs:
    lat = {i: m for i, m, _ in r["latents"]}
    by_attr[r["attribute"]][r["judge"]] = lat

ref_key = {"signal density": "signal_density", "intellectual ambition": "intellectual_ambition",
           "epistemic rigor": "epistemic_rigor"}
fused_all = {}
for attr, judges in by_attr.items():
    ids = sorted(set.intersection(*[set(l) for l in judges.values()]))
    names = sorted(judges)
    vecs = {j: [judges[j][i] for i in ids] for j in names}
    # z-score each judge, fuse by mean
    def z(v):
        m = sum(v)/len(v); s = math.sqrt(sum((x-m)**2 for x in v)/len(v)) or 1
        return [(x-m)/s for x in v]
    zs = {j: z(vecs[j]) for j in names}
    fused = [sum(zs[j][k] for j in names)/len(names) for k in range(len(ids))]
    fused_map = dict(zip(ids, fused))
    fused_all[attr] = fused_map
    trans = spearman(zs[names[0]], zs[names[1]]) if len(names) == 2 else float("nan")
    short = next((k for k in ref_key if attr.startswith(k)), None)
    val = float("nan")
    if short:
        rk = ref_key[short]
        rv = [ref[i][rk] for i in ids]
        val = spearman(fused, rv)
        val_a = spearman(zs[names[0]], rv); val_b = spearman(zs[names[1]], rv)
    print(f"\n== {attr[:60]}")
    print(f"  transmissibility (cross-judge rho): {trans:+.3f}")
    print(f"  validation vs 2026 corpus scores:   fused {val:+.3f}  ({names[0].split('/')[-1]} {val_a:+.3f}, {names[1].split('/')[-1]} {val_b:+.3f})")
    top = sorted(fused_map, key=fused_map.get, reverse=True)[:3]
    bot = sorted(fused_map, key=fused_map.get)[:2]
    for i in top: print(f"  TOP {fused_map[i]:+.2f}  {text[i][:90].replace(chr(10),' ')}")
    for i in bot: print(f"  BOT {fused_map[i]:+.2f}  {text[i][:90].replace(chr(10),' ')}")

# Cross-attribute correlation of fused maps (independence of dimensions)
attrs = sorted(fused_all)
print("\n== fused cross-attribute correlations")
for i in range(len(attrs)):
    for j in range(i+1, len(attrs)):
        ids = sorted(set(fused_all[attrs[i]]) & set(fused_all[attrs[j]]))
        rho = spearman([fused_all[attrs[i]][k] for k in ids], [fused_all[attrs[j]][k] for k in ids])
        print(f"  {attrs[i][:28]} × {attrs[j][:28]}: {rho:+.3f}")

json.dump({a: m for a, m in fused_all.items()}, open(f"{base}/fused_map.json", "w"))
print("\nfused_map.json written")
