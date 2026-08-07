"""Agreement metrics between the two rails' sort outputs and traces.

Usage: python3 analyze.py api.json api-trace.jsonl cc.json cc-trace.jsonl
All metrics carry denominators per PRINCIPLES.md §3.
"""
import itertools, json, math, sys

def load_board(path):
    d = json.load(open(path))
    items = d["items"] if isinstance(d, dict) else d
    ids = [it["id"] for it in items]
    means = {it["id"]: (it.get("mean") if it.get("mean") is not None else -i)
             for i, it in enumerate(items)}
    return ids, means

def ranks(ids):
    return {eid: i for i, eid in enumerate(ids)}

def spearman(ra, rb):
    common = sorted(set(ra) & set(rb))
    n = len(common)
    d2 = sum((ra[e] - rb[e]) ** 2 for e in common)
    return 1 - 6 * d2 / (n * (n * n - 1)), n

def kendall(ra, rb):
    common = sorted(set(ra) & set(rb))
    conc = disc = 0
    for x, y in itertools.combinations(common, 2):
        s = (ra[x] - ra[y]) * (rb[x] - rb[y])
        if s > 0: conc += 1
        elif s < 0: disc += 1
    n = conc + disc
    return (conc - disc) / n if n else float("nan"), n

def load_pair_verdicts(path):
    """directed pair (a_id, b_id) -> list of judged winners (entity ids), unswapped."""
    out = {}
    rows = 0
    for line in open(path):
        t = json.loads(line)
        rows += 1
        if t.get("refused") or t.get("error") or t.get("higher_ranked") not in ("A", "B"):
            continue
        a, b = t["entity_a_id"], t["entity_b_id"]
        # Verified against multi.rs:918-985 (2026-08-06): trace entity_a_id/
        # entity_b_id are recorded in PRESENTED order (swapped already applied),
        # so higher_ranked "A" always means entity_a_id. No unswap here.
        winner = a if t["higher_ranked"] == "A" else b
        out.setdefault(tuple(sorted((a, b))), []).append(winner)
    return out, rows

def majority(winners):
    from collections import Counter
    c = Counter(winners).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None  # internal tie
    return c[0][0]

def main(api_board, api_trace, cc_board, cc_trace):
    ids_a, means_a = load_board(api_board)
    ids_c, means_c = load_board(cc_board)
    ra, rc = ranks(ids_a), ranks(ids_c)
    rho, n = spearman(ra, rc)
    tau, npairs = kendall(ra, rc)
    print(f"board: spearman={rho:.3f} (n={n} items), kendall_tau={tau:.3f} (n={npairs} pairs)")

    va, rows_a = load_pair_verdicts(api_trace)
    vc, rows_c = load_pair_verdicts(cc_trace)
    shared = sorted(set(va) & set(vc))
    agree = disagree = tied = 0
    disagreements = []
    for pair in shared:
        ma, mc = majority(va[pair]), majority(vc[pair])
        if ma is None or mc is None:
            tied += 1
        elif ma == mc:
            agree += 1
        else:
            disagree += 1
            disagreements.append(pair)
    print(f"pairs: api rows={rows_a} cc rows={rows_c}; shared undirected pairs={len(shared)}")
    denom = agree + disagree
    if denom:
        print(f"pair-level agreement: {agree}/{denom} = {agree/denom:.3f} "
              f"(+{tied} internal-tie pairs excluded)")
    for p in disagreements:
        print(f"  disagree: {p} api={majority(va[p])} cc={majority(vc[p])}")

if __name__ == "__main__":
    main(*sys.argv[1:5])
