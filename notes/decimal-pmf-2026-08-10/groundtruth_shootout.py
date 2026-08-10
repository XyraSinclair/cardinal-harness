#!/usr/bin/env python
"""Oracle probe #3: exact-ground-truth estimator shootout.

Phase `enumerate`: compute the EXACT grammar-masked pushforward of
google/gemma-4-E2B-it (full logits, MPS) over the ratio instrument

    {"higher_ranked": "<A|B>", "ratio": "<[0-9]{1,3}.[0-9]>"}

with masked+renormalized conditionals at every stochastic token position
(matching the measured API semantics from RESULTS.md census finding 1).
gemma-4 tokenizes digits singly, so the trie is natively digit-level:
node contexts 1 + 2 + 20 + 200 + 2220 = 2443 forwards, 22,200 leaves.
Truth saved to groundtruth_tree.json.

Phase `shootout`: emulate constrained access tiers (top5 / top20 /
chosen-token-only / sample-only) by sampling from the stored truth, and
compare estimators of E[h(Y)], h = signed log10 ratio clamped to domain
[1.0, 999.9], at equal call budgets:

  mc          plain Monte Carlo mean of h over draws
  hv1         harvest.py v1 credal ledger, midpoint imputation (+ envelope
              coverage/width, the anytime-soundness check)
  head_same   exact-head atoms + residual conditional MC from the SAME
              draws (the tempting estimator; measures Oracle's
              discover-then-subtract selection-bias warning)
  head_split  exact-head + residual MC, cross-fit (first half fixes the
              head, second half estimates the residual conditional mean)
  atom_ht     atom Horvitz-Thompson, pi_y = 1-(1-p_y)^N over discovered
              leaves (needs only chosen-token exact logprobs)

Everything is exact (no provider jitter) — bias measured here is
structural, not noise. Run with /tmp/dpmf-venv/bin/python.
"""

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE_PATH = HERE / "groundtruth_tree.json"
RESULTS_PATH = HERE / "shootout_results.json"

MODEL = "google/gemma-4-E2B-it"
PROMPT = (
    "Compare by mass. entity_A: a chicken egg. entity_B: a bowling ball. "
    'Answer ONLY JSON {"higher_ranked": "A"|"B", "ratio": "<decimal like 12.5>"} '
    "where ratio is how many times more massive the higher one is."
)
SKELETON_PRE = '{"higher_ranked": "'
SKELETON_MID = '", "ratio": "'
DOMAIN_LO, DOMAIN_HI = 1.0, 999.9
ZMAX = math.log10(DOMAIN_HI)

DIGITS = [str(d) for d in range(10)]


def h_of(direction: str, r: float) -> float:
    """Signed log10 ratio, clamped into the declared domain (total, bounded)."""
    r = min(max(r, DOMAIN_LO), DOMAIN_HI)
    z = math.log10(r)
    return z if direction == "B" else -z


def node_legal(digits: str):
    """Legal next surface tokens at a digit-trie node. digits may contain '.'."""
    if "." in digits:
        intpart, frac = digits.split(".")
        assert frac == "", "frac node only"
        return DIGITS  # exactly one frac digit
    if len(digits) == 0:
        return DIGITS
    if len(digits) < 3:
        return DIGITS + ["."]
    return ["."]  # forced after 3 int digits (conditional == 1 under masking)


def subtree_h_range(direction: str, digits: str):
    """[min,max] of h over all grammar completions of this prefix."""
    if "." in digits:
        intpart = digits.split(".")[0]
        rmin, rmax = float(intpart + ".0"), float(intpart + ".9")
    elif digits == "":
        rmin, rmax = 0.0, 999.9
    else:
        rmin = float(digits + ".0")
        rmax = float(digits + "9" * (3 - len(digits)) + ".9")
    a, b = h_of(direction, rmin), h_of(direction, rmax)
    return (min(a, b), max(a, b))


# ---------------------------------------------------------------- enumerate

def phase_enumerate():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("mps").eval()

    def ids_of(text):
        return tok.encode(text, add_special_tokens=False)

    # single-token surface ids the instrument masks to
    tid = {}
    for s in DIGITS + [".", "A", "B"]:
        enc = ids_of(s)
        assert len(enc) == 1, f"{s!r} not a single token: {enc}"
        tid[s] = enc[0]

    enc = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    base = enc["input_ids"][0].tolist() + ids_of(SKELETON_PRE)
    mid = ids_of(SKELETON_MID)

    def ctx_tokens(key):
        """key: '' (dir node) or 'D:digits' (digit node)."""
        if key == "":
            return base
        d, digits = key.split(":", 1)
        toks = base + [tid[d]] + mid
        for ch in digits:
            toks.append(tid[ch])
        return toks

    # build all node keys, grouped by level (uniform length per level)
    levels = [[""]]
    dir_keys = [f"{d}:" for d in ("A", "B")]
    levels.append(dir_keys)
    lv2 = [f"{d}:{x}" for d in ("A", "B") for x in DIGITS]
    levels.append(lv2)
    lv3 = [f"{d}:{x}{y}" for d in ("A", "B") for x in DIGITS for y in DIGITS]
    levels.append(lv3)
    # frac nodes, split by int length so each level has uniform context length
    # (grammar is [0-9]{1,3} — leading zeros are legal digit strings)
    for n in (1, 2, 3):
        levels.append([f"{d}:{s}." for d in ("A", "B")
                       for s in ("".join(c) for c in _digit_strings(n))])

    node_dist = {}
    t_start = time.time()
    total = sum(len(lv) for lv in levels)
    done = 0
    BATCH = 24
    for lv in levels:
        for i in range(0, len(lv), BATCH):
            chunk = lv[i:i + BATCH]
            toks = [ctx_tokens(k) for k in chunk]
            L = {len(t) for t in toks}
            assert len(L) == 1, f"nonuniform level chunk lengths {L}"
            batch = torch.tensor(toks, device="mps")
            with torch.no_grad():
                logits = model(batch).logits[:, -1, :].float().cpu()
            for k, row in zip(chunk, logits):
                legal = node_legal(k.split(":", 1)[1] if k else "")
                if k == "":
                    legal_ids = [tid["A"], tid["B"]]
                    legal_names = ["A", "B"]
                else:
                    legal_ids = [tid[s] for s in legal]
                    legal_names = legal
                sel = row[legal_ids]
                logp = torch.log_softmax(sel, dim=-1)  # masked + renormalized
                node_dist[k] = {nm: float(p) for nm, p in zip(legal_names, logp.exp())}
            done += len(chunk)
        print(f"  level done, {done}/{total} nodes, {time.time()-t_start:.0f}s", flush=True)

    # leaf pmf
    leaves = {}
    for d in ("A", "B"):
        pd = node_dist[""][d]
        for n in (1, 2, 3):
            for s in ("".join(c) for c in _digit_strings(n)):
                p = pd
                prefix = ""
                for ch in s:
                    p *= node_dist[f"{d}:{prefix}"][ch]
                    prefix += ch
                if len(s) < 3:
                    p *= node_dist[f"{d}:{s}"]["."]
                # after 3 digits '.' is forced: conditional 1 under masking
                for f in DIGITS:
                    pf = p * node_dist[f"{d}:{s}."][f]
                    leaves[f"{d}:{s}.{f}"] = pf

    tot = sum(leaves.values())
    print(f"leaf mass total = {tot:.12f} (must be 1 up to fp)")
    ez = sum(p * h_of(k[0], float(k.split(":")[1])) for k, p in leaves.items())
    print(f"TRUE E[h] = {ez:.6f}")

    TREE_PATH.write_text(json.dumps({
        "meta": {"model": MODEL, "prompt": PROMPT, "domain": [DOMAIN_LO, DOMAIN_HI],
                 "semantics": "masked+renormalized per census finding 1",
                 "n_nodes": len(node_dist), "n_leaves": len(leaves),
                 "leaf_mass_total": tot, "true_E_h": ez},
        "node_dist": node_dist,
        "leaves": leaves,
    }))
    print(f"wrote {TREE_PATH} ({TREE_PATH.stat().st_size/1e6:.1f} MB)")


def _digit_strings(n):
    import itertools
    return itertools.product("0123456789", repeat=n)


# ---------------------------------------------------------------- shootout

class Truth:
    def __init__(self, tree):
        self.node_dist = tree["node_dist"]
        self.leaves = tree["leaves"]
        self.true_E = tree["meta"]["true_E_h"]
        self.keys = list(self.leaves)
        self.probs = [self.leaves[k] for k in self.keys]
        self.h = [h_of(k[0], float(k.split(":")[1])) for k in self.keys]
        # per-node sorted (token, p) for top-k emulation
        self.node_sorted = {k: sorted(d.items(), key=lambda kv: -kv[1])
                            for k, d in self.node_dist.items()}

    def path_nodes(self, leaf_key):
        """Stochastic (node_key, token, p) triples along a leaf's path."""
        d, rest = leaf_key.split(":", 1)
        digits = rest  # e.g. '23.5'
        out = [("", d, self.node_dist[""][d])]
        prefix = ""
        for ch in digits:
            nk = f"{d}:{prefix}"
            if ch == "." and len(prefix) == 3:
                prefix += ch  # forced dot: not stochastic, skip
                continue
            out.append((nk, ch, self.node_dist[nk][ch]))
            prefix += ch
        return out


def draw_calls(truth, rng, n):
    """n independent draws; each = list of (node_key, token, exact_p) + leaf."""
    calls = []
    cum = truth._cum if hasattr(truth, "_cum") else None
    if cum is None:
        c, s = [], 0.0
        for p in truth.probs:
            s += p
            c.append(s)
        truth._cum = c
        cum = c
    import bisect
    for _ in range(n):
        u = rng.random() * cum[-1]
        i = bisect.bisect_left(cum, u)
        leaf = truth.keys[i]
        calls.append((leaf, truth.h[i], truth.path_nodes(leaf)))
    return calls


def ledger(truth, calls, topk):
    """harvest-v1 style ledger. Returns (atoms {leaf: p}, cells [(m, lo, hi)]).

    Node knowledge: exact chosen-token p from every visit, plus top-k of the
    node's masked dist when topk > 0. Atoms = fully-resolved root paths.
    """
    known = {}  # node_key -> {token: p}
    for _, _, path in calls:
        for nk, tok_, p in path:
            d = known.setdefault(nk, {})
            d[tok_] = p
            if topk:
                for t2, p2 in truth.node_sorted[nk][:topk]:
                    d[t2] = p2

    atoms, cells = {}, []

    def rec(nk, direction, digits, mass):
        if nk not in known:
            lo, hi = subtree_h_range(direction, digits)
            cells.append((mass, lo, hi))
            return
        kn = known[nk]
        acc = 0.0
        for tok_, p in kn.items():
            acc += p
            if nk == "":
                rec(f"{tok_}:", tok_, "", mass * p)
            else:
                nd = digits + tok_
                if "." in nd and len(nd.split(".")[1]) == 1:
                    atoms_key = f"{direction}:{nd}"
                    atoms[atoms_key] = mass * p
                elif "." not in nd and len(nd) == 3:
                    # forced dot
                    rec(f"{direction}:{nd}.", direction, nd + ".", mass * p)
                else:
                    rec(f"{direction}:{nd}", direction, nd, mass * p)
        resid = 1.0 - acc
        if resid > 1e-12:
            lo, hi = subtree_h_range(direction, digits)
            cells.append((mass * resid, lo, hi))

    rec("", "", "", 1.0)
    return atoms, cells


def est_hv1(truth, calls, topk):
    atoms, cells = ledger(truth, calls, topk)
    e = sum(p * h_of(k[0], float(k.split(":")[1])) for k, p in atoms.items())
    lo = e + sum(m * l for m, l, _ in cells)
    hi = e + sum(m * hh for m, _, hh in cells)
    mid = e + sum(m * (l + hh) / 2 for m, l, hh in cells)
    return mid, lo, hi


def est_head_residual(truth, calls, topk, split):
    """Exact-head + residual conditional MC. split=True -> cross-fit."""
    if split:
        head_calls, est_calls = calls[: len(calls) // 2], calls[len(calls) // 2:]
    else:
        head_calls = est_calls = calls
    atoms, _ = ledger(truth, head_calls, topk)
    qc = sum(atoms.values())
    e_head = sum(p * h_of(k[0], float(k.split(":")[1])) for k, p in atoms.items())
    out = [h for leaf, h, _ in est_calls if leaf not in atoms]
    if out:
        resid_mean = sum(out) / len(out)
    else:
        # no out-of-head draw in the estimation batch: impute 0 (center of the
        # symmetric h-range); contribution is bounded by (1-qc)*ZMAX
        resid_mean = 0.0
    return e_head + max(0.0, 1.0 - qc) * resid_mean


def est_atom_ht(truth, calls):
    n = len(calls)
    seen = {}
    for leaf, h, path in calls:
        if leaf not in seen:
            p = 1.0
            for _, _, cp in path:
                p *= cp
            seen[leaf] = (p, h)
    e = 0.0
    for p, h in seen.values():
        pi = 1.0 - (1.0 - p) ** n
        e += p * h / pi
    return e


def phase_shootout():
    import random
    tree = json.loads(TREE_PATH.read_text())
    truth = Truth(tree)
    true_E = truth.true_E
    print(f"true E[h] = {true_E:.6f}, leaves = {len(truth.keys)}")

    TIERS = {"top5": 5, "top20": 20, "chosen": 0, "sample": None}
    BUDGETS = [5, 10, 25, 50, 100]
    R = 300
    rows = []
    t0 = time.time()
    for tier, topk in TIERS.items():
        for n in BUDGETS:
            errs = {k: [] for k in
                    ("mc", "hv1", "head_same", "head_split", "atom_ht")}
            widths, covered = [], 0
            for rep in range(R):
                rng = random.Random(1_000_003 * BUDGETS.index(n)
                                    + 7919 * list(TIERS).index(tier) + rep)
                calls = draw_calls(truth, rng, n)
                hs = [h for _, h, _ in calls]
                errs["mc"].append(sum(hs) / n - true_E)
                if topk is None:
                    continue  # sample-only tier: MC only
                mid, lo, hi = est_hv1(truth, calls, topk)
                errs["hv1"].append(mid - true_E)
                widths.append(hi - lo)
                covered += int(lo - 1e-9 <= true_E <= hi + 1e-9)
                errs["head_same"].append(
                    est_head_residual(truth, calls, topk, split=False) - true_E)
                errs["head_split"].append(
                    est_head_residual(truth, calls, topk, split=True) - true_E)
                errs["atom_ht"].append(est_atom_ht(truth, calls) - true_E)
            for est, es in errs.items():
                if not es:
                    continue
                bias = sum(es) / len(es)
                rmse = math.sqrt(sum(e * e for e in es) / len(es))
                row = {"tier": tier, "n": n, "est": est,
                       "bias": bias, "rmse": rmse, "reps": len(es)}
                if est == "hv1":
                    row["cover"] = covered / R
                    ws = sorted(widths)
                    row["med_width"] = ws[len(ws) // 2]
                rows.append(row)
            print(f"  {tier} n={n} done {time.time()-t0:.0f}s", flush=True)

    RESULTS_PATH.write_text(json.dumps(
        {"true_E_h": true_E, "R": R, "rows": rows}, indent=1))
    print(f"wrote {RESULTS_PATH}")

    # compact table
    print(f"\n{'tier':6} {'n':>4} {'est':10} {'bias':>8} {'rmse':>8} "
          f"{'cover':>6} {'width':>7}")
    for r in rows:
        print(f"{r['tier']:6} {r['n']:>4} {r['est']:10} {r['bias']:>8.4f} "
              f"{r['rmse']:>8.4f} "
              f"{r.get('cover', float('nan')):>6.2f} "
              f"{r.get('med_width', float('nan')):>7.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("enumerate", "shootout"):
        sys.exit("usage: groundtruth_shootout.py enumerate|shootout")
    (phase_enumerate if sys.argv[1] == "enumerate" else phase_shootout)()
