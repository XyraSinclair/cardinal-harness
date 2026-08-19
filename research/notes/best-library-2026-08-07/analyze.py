"""Analysis for the weak-judge amplification + cardinal-recovery benchmark.

Reads truth.json, baselines-<judge>.json, cardinal-<judge>.json.
Prints Spearman/Kendall vs truth per cell + log-calibration for M4 and M1.

Run: python3 analyze.py
"""
import json, math

TRUTH = json.load(open("truth.json"))
ITEMS = list(TRUTH)
N = len(ITEMS)
PAIRS = N * (N - 1) // 2


def rankdata(values):
    """Average ranks, ties handled."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


def kendall(xs, ys):
    c = d = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            s = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / PAIRS


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den


def slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)


truth_vals = [TRUTH[i] for i in ITEMS]
ln_truth = [math.log(v) for v in truth_vals]

for short in ("gpt-5.4-mini", "gpt-5.4-nano"):
    base = json.load(open(f"baselines-{short}.json"))
    card = json.load(open(f"cardinal-{short.replace('gpt-5.4-', '')}.json"))
    card_score = {it["id"]: it["latent_mean"] for it in card["items"]}
    assert set(card_score) == set(ITEMS), "cardinal item mismatch"

    m1 = [base["m1"][i]["estimate"] for i in ITEMS]
    m2 = [base["m2"][i]["score"] for i in ITEMS]
    m3_order = base["m3"]["order"]
    m3 = [-m3_order.index(i) for i in ITEMS]  # earlier in list = more populous
    m4 = [card_score[i] for i in ITEMS]

    print(f"\n=== judge {short}  (N={N}, {PAIRS} rank pairs) ===")
    for tag, vals in (("M1 direct estimate", m1), ("M2 score 0-100", m2),
                      ("M3 listwise", m3), ("M4 cardinal", m4)):
        print(f"  {tag:20s} spearman={spearman(vals, truth_vals):+.3f} "
              f"kendall={kendall(vals, truth_vals):+.3f}")

    ln_m1 = [math.log(v) for v in m1]
    print(f"  calib M1 ln(est) vs ln(truth):  r={pearson(ln_m1, ln_truth):+.4f} "
          f"slope={slope(ln_truth, ln_m1):+.3f}")
    print(f"  calib M4 latent vs ln(truth):   r={pearson(m4, ln_truth):+.4f} "
          f"slope={slope(ln_truth, m4):+.3f}")
    print(f"  calib M2 score vs ln(truth):    r={pearson(m2, ln_truth):+.4f} "
          f"  (0-100 scale, saturation check: "
          f"{sum(1 for v in m2 if v <= 5 or v >= 95)}/{N} at scale edges)")
