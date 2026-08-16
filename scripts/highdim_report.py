#!/usr/bin/env python3
"""Two-dense-judge report on the high-dimensional attribute battery.

Reads ratiometer.judgments for the bare and elaborated run_tags and reports,
per attribute:
  agree_bare   cross-judge direction agreement, bare phrasing
  agree_elab   cross-judge direction agreement, simple-elaborated phrasing
  d_elab       agree_elab - agree_bare  (does elaboration sharpen the attribute?)
  H_qwen/H_gem mean logprob entropy per judge (bare) — how decisive each is

Direction agreement = over pairs both judges judged decisively, share where
they pick the same winner. Logprob entropy comes straight from the landed PMF.

Usage: highdim_report.py <bare_run_tag> <elab_run_tag> <judgeA> <judgeB>
"""
import subprocess, sys, json, collections

SSH = "/usr/bin/ssh"
CH = "/data/clickhouse-twitter-lab/bin/clickhouse client --port 19000 --query"


def q(sql):
    p = subprocess.run([SSH, "colo2", f"{CH} {json.dumps(sql)}"],
                       capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr[:400])
    return [l.split("\t") for l in p.stdout.splitlines() if l]


def agreement_by_attr(run_tag, ja, jb):
    rows = q("SELECT attribute, model, entity_a_hash, entity_b_hash, higher_ranked "
             "FROM ratiometer.judgments WHERE run_tag = '" + run_tag +
             "' AND refused = 0 AND higher_ranked IN ('A','B')")
    win = collections.defaultdict(lambda: collections.defaultdict(dict))
    for attr, model, ah, bh, hr in rows:
        key = tuple(sorted((ah, bh)))
        win[attr][model][key] = ah if hr == "A" else bh
    out = {}
    for attr in win:
        A, B = win[attr].get(ja, {}), win[attr].get(jb, {})
        shared = set(A) & set(B)
        if shared:
            out[attr] = sum(1 for k in shared if A[k] == B[k]) / len(shared)
    return out


def entropy_by_attr(run_tag, model):
    rows = q("SELECT attribute, round(avg(entropy),3) FROM ratiometer.judgments "
             "WHERE run_tag = '" + run_tag + "' AND model = '" + model +
             "' AND length(posterior) > 10 GROUP BY attribute")
    return {a: float(h) for a, h in rows}


def base_attr(a):
    return a.split(":", 1)[0].strip()


def main():
    bare_tag, elab_tag, ja, jb = sys.argv[1:5]
    ab = agreement_by_attr(bare_tag, ja, jb)
    ae = {base_attr(a): v for a, v in agreement_by_attr(elab_tag, ja, jb).items()}
    hq = entropy_by_attr(bare_tag, ja)
    hg = entropy_by_attr(bare_tag, jb)
    print(f"judges: {ja} vs {jb}   corpus attribute: high-dimensional\n")
    hdr = ["attribute", "agree_bare", "agree_elab", "d_elab", f"H_{ja[:6]}", f"H_{jb[:6]}"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    rows = []
    for a in sorted(ab, key=lambda a: ab[a]):
        be, el = ab[a], ae.get(a)
        d = (el - be) if el is not None else None
        rows.append((a, be, el, d, hq.get(a), hg.get(a)))
    for a, be, el, d, h1, h2 in rows:
        print(f"| {a[:44]:44} | {be:9.3f} | "
              f"{('%.3f' % el) if el is not None else '  —  ':>9} | "
              f"{('%+.3f' % d) if d is not None else '  —  ':>7} | "
              f"{('%.2f' % h1) if h1 is not None else ' — ':>6} | "
              f"{('%.2f' % h2) if h2 is not None else ' — ':>6} |")
    bar = sum(r[1] for r in rows) / len(rows)
    print(f"\nmean bare agreement: {bar:.3f}")
    ds = [r[3] for r in rows if r[3] is not None]
    if ds:
        print(f"mean elaboration effect (d_elab): {sum(ds)/len(ds):+.3f} "
              f"over {len(ds)} attributes ({sum(1 for d in ds if d>0)} improved)")


if __name__ == "__main__":
    main()
