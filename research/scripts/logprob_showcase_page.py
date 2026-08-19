#!/usr/bin/env python3
"""Render the logprob consistency showcase as a self-contained local HTML page.

Input: showcase stats JSON (from logprob_showcase_stats.py).
Output: single HTML file, inline SVG, no external requests. Palette per the
validated reference instance (two categorical slots, sequential blue ramp);
light/dark via prefers-color-scheme tokens.
"""
import json, math, sys

# palette roles (validated 2-slot categorical + sequential blue)
LIGHT = {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b",
         "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"}
DARK = {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff",
        "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

W, H_CH, PAD_L, PAD_R, PAD_T, PAD_B = 960, 340, 56, 20, 18, 40
QW, GM = "qwen38-27b", "gemma4-31b"


def sx(q, lo=0.5, hi=1.0):
    return PAD_L + (q - lo) / (hi - lo) * (W - PAD_L - PAD_R)


def sy(v, lo=0.4, hi=1.0, h=H_CH):
    return PAD_T + (1 - (v - lo) / (hi - lo)) * (h - PAD_T - PAD_B)


def polyline(points, color, width=2, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round" opacity="{opacity}"{d}/>')


def grid_and_axes(ylo, yhi, ylab, xlab, yticks, xticks, h=H_CH, xlo=0.5, xhi=1.0, pct=True):
    s = []
    for v in yticks:
        y = sy(v, ylo, yhi, h)
        s.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" class="grid"/>')
        lab = f"{v*100:.0f}%" if pct else f"{v:g}"
        s.append(f'<text x="{PAD_L-8}" y="{y+4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    for v in xticks:
        x = sx(v, xlo, xhi)
        s.append(f'<text x="{x:.1f}" y="{h-PAD_B+18}" class="tick" text-anchor="middle">{v:g}</text>')
    s.append(f'<line x1="{PAD_L}" y1="{sy(ylo,ylo,yhi,h):.1f}" x2="{W-PAD_R}" y2="{sy(ylo,ylo,yhi,h):.1f}" class="axis"/>')
    s.append(f'<text x="{PAD_L}" y="{PAD_T-4}" class="axlab">{ylab}</text>')
    s.append(f'<text x="{W-PAD_R}" y="{h-6}" class="axlab" text-anchor="end">{xlab}</text>')
    return "".join(s)


def reliability_chart(d):
    s = [f'<svg viewBox="0 0 {W} {H_CH}" role="img" aria-label="Reliability diagram">']
    s.append(grid_and_axes(0.4, 1.0, "swapped-presentation agreement",
                           "stated confidence P(chosen side)", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    # reference: perfectly calibrated sampler y = q^2+(1-q)^2
    ref = [(sx(q), sy(q * q + (1 - q) ** 2, 0.4, 1.0)) for q in
           [0.5 + i / 100 for i in range(51)]]
    s.append(polyline(ref, "var(--muted)", 1.5, dash="6 5", opacity=0.9))
    s.append(f'<text x="{sx(0.51):.0f}" y="{sy(0.455,0.4,1.0):.0f}" class="ref">dashed: perfectly calibrated sampler, q²+(1−q)²</text>')
    for m, color in [(QW, "var(--s1)"), (GM, "var(--s2)")]:
        pts = [(sx(r["q_mid"]), sy(r["agree"], 0.4, 1.0)) for r in d["per_model"][m]["reliability"]]
        s.append(polyline(pts, color, 2.5))
        for r in d["per_model"][m]["reliability"]:
            x, y = sx(r["q_mid"]), sy(r["agree"], 0.4, 1.0)
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" class="pt">'
                     f'<title>{m} · stated {r["q_mid"]:.2f} → twin agrees {r["agree"]*100:.1f}% (n={r["n"]:,})</title></circle>')
        lx, ly = pts[-1]
        anchor_dy = -10 if m == QW else 16
        s.append(f'<text x="{lx-4:.0f}" y="{ly+anchor_dy:.0f}" class="dlabel" text-anchor="end" fill="{color}">{m.split("-")[0]}</text>')
    s.append("</svg>")
    return "".join(s)


def crossjudge_chart(d):
    s = [f'<svg viewBox="0 0 {W} {H_CH}" role="img" aria-label="Cross-judge agreement by confidence">']
    s.append(grid_and_axes(0.4, 1.0, "the other dense judge agrees",
                           "both judges' mean stated confidence", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           [0.7, 0.8, 0.9, 1.0], xlo=0.68, xhi=1.0))
    rows = d["crossjudge"]["by_confidence"]
    pts = [(sx(r["q_mid"], 0.68, 1.0), sy(r["agree"], 0.4, 1.0)) for r in rows]
    s.append(polyline(pts, "var(--s1)", 2.5))
    for r in rows:
        x, y = sx(r["q_mid"], 0.68, 1.0), sy(r["agree"], 0.4, 1.0)
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--s1)" class="pt">'
                 f'<title>joint confidence {r["q_mid"]:.2f} → agreement {r["agree"]*100:.1f}% (n={r["n"]})</title></circle>')
    lastx, lasty = pts[-1]
    s.append(f'<text x="{lastx-8:.0f}" y="{lasty-10:.0f}" class="dlabel" text-anchor="end" fill="var(--s1)">'
             f'{rows[-1]["agree"]*100:.1f}%</text>')
    s.append("</svg>")
    return "".join(s)


def hist_pair(d, field, lo, hi, xlab, fmt="{:g}", label_side="right"):
    """Two small multiples sharing an x-axis."""
    hh = 190
    out = []
    for m, color in [(QW, "var(--s1)"), (GM, "var(--s2)")]:
        hd = d["per_model"][m][field]
        counts = hd["counts"]
        peak = max(counts)
        n = len(counts)
        bw = (W - PAD_L - PAD_R) / n
        s = [f'<svg viewBox="0 0 {W} {hh}" role="img" aria-label="{m} {xlab} distribution">']
        s.append(f'<line x1="{PAD_L}" y1="{hh-PAD_B}" x2="{W-PAD_R}" y2="{hh-PAD_B}" class="axis"/>')
        for i, c in enumerate(counts):
            if c == 0:
                continue
            bh = (c / peak) * (hh - PAD_T - PAD_B)
            x = PAD_L + i * bw
            v0 = hd["lo"] + (hd["hi"] - hd["lo"]) * i / n
            v1 = hd["lo"] + (hd["hi"] - hd["lo"]) * (i + 1) / n
            s.append(f'<rect x="{x+1:.1f}" y="{hh-PAD_B-bh:.1f}" width="{bw-2:.1f}" height="{bh:.1f}" '
                     f'rx="3" fill="{color}"><title>{m} · {fmt.format(v0)}–{fmt.format(v1)}: {c:,} judgments</title></rect>')
        for v in xticks_for(lo, hi):
            x = PAD_L + (v - lo) / (hi - lo) * (W - PAD_L - PAD_R)
            s.append(f'<text x="{x:.1f}" y="{hh-PAD_B+18}" class="tick" text-anchor="middle">{fmt.format(v)}</text>')
        mean_lab = f'H̄ = {d["per_model"][m]["mean_entropy"]:.2f} nats' if field == "entropy_hist" else ""
        lx, anch = (W - PAD_R, "end") if label_side == "right" else (PAD_L + 6, "start")
        s.append(f'<text x="{lx}" y="{PAD_T+6}" class="dlabel" text-anchor="{anch}" fill="{color}">{m.split("-")[0]}'
                 f'{(" · " + mean_lab) if mean_lab else ""}</text>')
        s.append(f'<text x="{W-PAD_R}" y="{hh-6}" class="axlab" text-anchor="end">{xlab}</text>')
        s.append("</svg>")
        out.append("".join(s))
    return "<div class='stack'>" + "".join(out) + "</div>"


def xticks_for(lo, hi):
    span = hi - lo
    step = 0.1 if span <= 0.6 else (0.5 if span <= 2 else 1.0)
    t, out = lo, []
    while t <= hi + 1e-9:
        out.append(round(t, 2))
        t += step
    return out


def mirror_maps(d):
    """25x25 density heatmaps of canonical p in order 1 vs order 2."""
    size, n = 380, 25
    out = []
    for m in (QW, GM):
        gridc = [[0] * n for _ in range(n)]
        for p1, p2 in d["per_model"][m]["mirror_sample"]:
            i = min(int(p1 * n), n - 1)
            j = min(int(p2 * n), n - 1)
            gridc[j][i] += 1
        peak = max(max(r) for r in gridc) or 1
        cell = (size - 60) / n
        s = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="{m} mirror density">']
        for j in range(n):
            for i in range(n):
                c = gridc[j][i]
                if c == 0:
                    continue
                # log-scaled ramp index
                t = math.log1p(c) / math.log1p(peak)
                col = SEQ[min(int(t * len(SEQ)), len(SEQ) - 1)]
                x = 44 + i * cell
                y = size - 44 - (j + 1) * cell
                s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-0.6:.1f}" height="{cell-0.6:.1f}" '
                         f'fill="{col}"><title>p₁∈[{i/n:.2f},{(i+1)/n:.2f}) p₂∈[{j/n:.2f},{(j+1)/n:.2f}): {c}</title></rect>')
        # diagonal reference
        s.append(f'<line x1="44" y1="{size-44}" x2="{44+n*cell:.1f}" y2="{size-44-n*cell:.1f}" '
                 f'stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4" opacity="0.8"/>')
        s.append(f'<line x1="44" y1="{size-44}" x2="{44+n*cell:.1f}" y2="{size-44}" class="axis"/>')
        s.append(f'<line x1="44" y1="{size-44}" x2="44" y2="{size-44-n*cell:.1f}" class="axis"/>')
        for v in (0, 0.5, 1):
            s.append(f'<text x="{44+v*n*cell:.1f}" y="{size-28}" class="tick" text-anchor="middle">{v:g}</text>')
            s.append(f'<text x="36" y="{size-44-v*n*cell+4:.1f}" class="tick" text-anchor="end">{v:g}</text>')
        s.append(f'<text x="{44+(n*cell)/2:.0f}" y="{size-8}" class="axlab" text-anchor="middle">P(x wins), order 1</text>')
        s.append(f'<text x="12" y="{size-44-(n*cell)/2:.0f}" class="axlab" text-anchor="middle" '
                 f'transform="rotate(-90 12 {size-44-(n*cell)/2:.0f})">P(x wins), order 2</text>')
        sw = d["per_model"][m]["swap_agreement"]
        out.append(f"<figure><figcaption><span class='mchip' style='background:var(--{'s1' if m==QW else 's2'})'></span>"
                   f"{m} — swap agreement {sw*100:.1f}%</figcaption>{''.join(s)}</svg></figure>")
    return "<div class='pair'>" + "".join(out) + "</div>"


def tile(value, label, sub=""):
    return (f"<div class='tile'><div class='tv'>{value}</div><div class='tl'>{label}</div>"
            f"<div class='ts'>{sub}</div></div>")


def main():
    d = json.load(open(sys.argv[1]))
    qm, gm = d["per_model"][QW], d["per_model"][GM]
    cj = d["crossjudge"]
    tri = d["triads"]
    css_tokens_light = ";".join(f"--{k}:{v}" for k, v in
                                {"surface": LIGHT["surface"], "page": LIGHT["page"], "ink": LIGHT["ink"],
                                 "ink2": LIGHT["ink2"], "muted": LIGHT["muted"], "grid": LIGHT["grid"],
                                 "axis": LIGHT["axis"], "s1": LIGHT["s1"], "s2": LIGHT["s2"]}.items())
    css_tokens_dark = ";".join(f"--{k}:{v}" for k, v in
                               {"surface": DARK["surface"], "page": DARK["page"], "ink": DARK["ink"],
                                "ink2": DARK["ink2"], "muted": DARK["muted"], "grid": DARK["grid"],
                                "axis": DARK["axis"], "s1": DARK["s1"], "s2": DARK["s2"]}.items())
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Logprob Ledger</title>
<style>
:root {{ color-scheme: light; {css_tokens_light} }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; {css_tokens_dark} }} }}
* {{ box-sizing: border-box; margin: 0 }}
body {{ background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 40px 20px 80px }}
main {{ max-width: 1020px; margin: 0 auto }}
h1 {{ font-size: 28px; letter-spacing: -0.01em }}
h2 {{ font-size: 19px; margin: 44px 0 6px }}
p.lede {{ color: var(--ink2); max-width: 72ch; margin-top: 6px }}
p.note {{ color: var(--ink2); max-width: 72ch; margin: 6px 0 14px }}
.card {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 12px;
  padding: 18px 18px 10px; margin-top: 10px }}
svg {{ width: 100%; height: auto; display: block }}
.grid {{ stroke: var(--grid); stroke-width: 1 }}
.axis {{ stroke: var(--axis); stroke-width: 1.25 }}
.tick {{ fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums }}
.axlab {{ fill: var(--muted); font-size: 12px }}
.ref {{ fill: var(--muted); font-size: 12px; font-style: italic }}
.dlabel {{ font-size: 13px; font-weight: 600 }}
.pt:hover {{ r: 6 }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 18px }}
.tile {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 12px; padding: 14px 16px }}
.tv {{ font-size: 26px; font-weight: 650; letter-spacing: -0.01em }}
.tl {{ font-size: 13px; color: var(--ink2); margin-top: 2px }}
.ts {{ font-size: 12px; color: var(--muted) }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px }}
.pair figure {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 12px; padding: 12px }}
.pair figcaption {{ font-size: 13px; color: var(--ink2); margin-bottom: 6px }}
.mchip {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px }}
.stack svg + svg {{ margin-top: 4px }}
.legend {{ display: flex; gap: 18px; font-size: 13px; color: var(--ink2); margin: 8px 2px 0 }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 6px; vertical-align: -1px }}
.legend .q::before {{ background: var(--s1) }} .legend .g::before {{ background: var(--s2) }}
@media (max-width: 760px) {{ .pair {{ grid-template-columns: 1fr }} }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px }}
caption {{ text-align: left; color: var(--ink2); font-size: 13px; margin-bottom: 6px }}
th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums }}
th:first-child, td:first-child {{ text-align: left }}
thead th {{ color: var(--ink2); font-weight: 600 }}
</style></head><body><main>
<h1>The Logprob Ledger</h1>
<p class="lede">Every pairwise judgment the two standing dense judges make is a single structured
emission whose answer-token logprobs yield a full posterior — a direction PMF, a ratio-bucket PMF,
and an entropy. All of it lands in <code>ratiometer.judgments</code>. This page asks the health
questions of that instrument: is the stated confidence real, does the verdict survive A/B being
swapped, do two independent judges converge, and does the implied ordering stay transitive?
Corpus: Manifund grant proposals; subtle high-dimensional attributes.</p>

<div class="tiles">
{tile(f"{d['n_rows']:,}", "PMF-bearing judgments", "both judges, all landed")}
{tile(f"{qm['swap_agreement']*100:.1f}% / {gm['swap_agreement']*100:.1f}%", "swap agreement (qwen / gemma)", "verdict survives A↔B order")}
{tile(f"{tri[QW]['cyclic_frac']*100:.1f}% / {tri[GM]['cyclic_frac']*100:.1f}%", "cyclic triads (qwen / gemma)", f"{tri[QW]['triads']+tri[GM]['triads']:,} fully-observed triples")}
{tile(f"{cj['by_confidence'][-1]['agree']*100:.1f}%", "cross-judge agreement when both are confident", "stated ≥ 0.95 · independent judges")}
</div>

<h2>Calibration: is the stated probability real?</h2>
<p class="note">Bin every judgment by its stated confidence, then check how often the same pair,
re-presented in the opposite order, gets the same verdict. A judge whose logprobs are honest
sampling probabilities should track the dashed reference. <b>qwen38 rides the reference across the
entire range — its PMF is a measurement.</b> gemma stamps ≥0.95 on {gm['q_hist']['counts'][-3]+gm['q_hist']['counts'][-2]+gm['q_hist']['counts'][-1]:,} of {gm['n_judgments']:,} judgments yet its
swapped twin agrees only {gm['reliability'][-1]['agree']*100:.0f}% of the time there — its confidence is a posture.</p>
<div class="card">{reliability_chart(d)}
<div class="legend"><span class="q">qwen38-27b</span><span class="g">gemma4-31b</span></div></div>

<h2>Two temperaments, one instrument</h2>
<p class="note">The same story in distribution form. gemma concentrates nearly all probability
mass on one answer (mean entropy {gm['mean_entropy']:.2f} nats); qwen spreads honestly
({qm['mean_entropy']:.2f} nats). Neither is wrong as a ranker — but qwen's posteriors carry usable
uncertainty for the solver, while gemma's need recalibration before they can be trusted as variance.</p>
<div class="card">{hist_pair(d, "entropy_hist", 0.0, 4.0, "judgment entropy (nats)", "{:.1f}")}</div>
<div class="card" style="margin-top:14px">{hist_pair(d, "q_hist", 0.5, 1.0, "stated confidence P(chosen side)", "{:.2f}", label_side="left")}</div>

<h2>Position symmetry: the mirror test</h2>
<p class="note">Each counterbalanced pair is asked in both presentation orders. Plotting the
canonical win probability from order 1 against order 2, a position-blind judge lies on the
diagonal. Mass in the off-diagonal corners is position bias — verdicts that flipped with the
seating chart.</p>
{mirror_maps(d)}

<h2>Independent judges converge — when they're sure</h2>
<p class="note">On pairs both dense judges rated, agreement climbs monotonically with their joint
stated confidence, reaching {cj['by_confidence'][-1]['agree']*100:.1f}% when both are ≥0.95 sure.
Joint confidence is a usable certainty signal: filter to it and two independent 30B-class models
almost never disagree — the attribute, not the model, owns the residue.</p>
<div class="card">{crossjudge_chart(d)}</div>

<h2>Transitivity</h2>
<p class="note">Among item triples where all three pairs were judged, only
{tri[QW]['cyclic_frac']*100:.1f}% (qwen) and {tri[GM]['cyclic_frac']*100:.1f}% (gemma) form a cycle
(A&gt;B&gt;C&gt;A). These attributes behave like real one-dimensional latents almost everywhere —
the ~3% cyclic residue is the honest price of forcing a subtle quality onto a line.</p>

<table><caption>Instrument summary — {d['n_rows']:,} PMF-bearing judgments, Manifund corpus, subtle attribute tier</caption>
<thead><tr><th>judge</th><th>judgments</th><th>swap agree</th><th>mean |Δp| across orders</th><th>mean entropy</th><th>cyclic triads</th><th>calibration</th></tr></thead>
<tbody>
<tr><td>qwen38-27b (dense)</td><td>{qm['n_judgments']:,}</td><td>{qm['swap_agreement']*100:.1f}%</td><td>{qm['mean_abs_dp']:.3f}</td><td>{qm['mean_entropy']:.2f} nats</td><td>{tri[QW]['cyclic_frac']*100:.1f}%</td><td>tracks the sampler reference — trustworthy PMF</td></tr>
<tr><td>gemma4-31b (dense)</td><td>{gm['n_judgments']:,}</td><td>{gm['swap_agreement']*100:.1f}%</td><td>{gm['mean_abs_dp']:.3f}</td><td>{gm['mean_entropy']:.2f} nats</td><td>{tri[GM]['cyclic_frac']*100:.1f}%</td><td>overconfident — decisive but PMF needs recalibration</td></tr>
</tbody></table>

<p class="note" style="margin-top:28px">Data: <code>ratiometer.judgments</code> on scry ClickHouse ·
template <code>canonical_bucket_v1</code> · counterbalanced presentation · cross-judge pairs from the
highdim battery · generated by <code>scripts/logprob_showcase_stats.py</code> +
<code>scripts/logprob_showcase_page.py</code>.</p>
</main></body></html>"""
    open(sys.argv[2], "w").write(html)
    print(f"wrote {sys.argv[2]} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
