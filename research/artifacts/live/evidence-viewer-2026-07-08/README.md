# Interactive evidence viewer — the susceptibility slider (2026-07-08)

`index.html` is a self-contained, committed page that renders the live
susceptibility evidence from `../spin-probe-2026-07-05/` and
`../spin-sweep-2026-07-05/` as something a stranger can poke at: the
contested pair ("The obstacle is the way." vs "What gets measured gets
managed.", criterion *depth of insight about living well*), a field slider
from −3 (insistent pro-B) to +3 (insistent pro-A), and the measured
response m(f) moving under it.

Design contract:

- **Zero new judgement math.** Every number is inlined verbatim from the
  committed evidence JSON files (paths in the page footer). The slider
  selects among the 7 measured field points per model; nothing is interpolated,
  smoothed, or simulated.
- **The framing text shown is the framing text sent** — the literal
  intensity-graded templates from `src/rerank/spin.rs::spun_criterion_at`
  (post-apostrophe-escape fix; erratum restated on the page).
- **Denominators on everything**: comparisons, nanodollar costs, R², the
  n = 1 caveat, and the two-run neutral spread (0.000 probe vs −0.227
  sweep) stated rather than hidden.

Serve locally (never foreground a browser; print the URL):

```bash
cd artifacts/live && python3 -m http.server <port> --bind 127.0.0.1
# → http://127.0.0.1:<port>/evidence-viewer-2026-07-08/
```

Provenance: notes/ideation-2026-07-05/differentiation.md ranked this the
#1 ship-next ("zero new judgement math, pure rendering over data already
produced"); this pack is that ship. Chart palette validated (2 series,
worst adjacent CVD ΔE 23.7, all checks pass).
