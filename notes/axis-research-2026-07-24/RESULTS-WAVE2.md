# Wave 2 results — 8 axes, 24 runs, 576 comparisons, $1.74, 0 refusals

Scored by `analyze_wave2.py` against the thresholds frozen in
`WAVE2_SPEC.md` before any run. Evidence: `wave2/sort-*.json`.

| axis | fr↔fr | op↔mini | sol↔mini | gap | decoy fr/mini | verdict |
|---|---|---|---|---|---|---|
| eschatological_seriousness | +0.902 | +0.762 | +0.671 | +0.185 | 7 / 3 | **PASS** (T3) |
| rosetta_load | +0.951 | +0.538 | +0.524 | +0.420 | 10 / 11 | **PASS** (T2) |
| scar_tissue_density | +0.944 | +0.587 | +0.706 | +0.297 | 9 / 1 | **PASS** (T2+T3) |
| live_wire_prose | +0.951 | +0.685 | +0.748 | +0.234 | 10 / 5 | **PASS** (T2+T3) |
| antimemetic_payload | +0.965 | +0.699 | +0.615 | +0.308 | 10 / 3 | **PASS** (T2+T3) |
| hostile_paraphrase_invariance | +0.818 | +0.538 | +0.448 | +0.325 | 9 / 11 | **PASS** (T2) |
| authorial_irreducibility | +0.839 | +0.916 | +0.867 | −0.052 | 9 / 8 | WEAK |
| voltage_under_load | +0.923 | +0.797 | +0.846 | +0.101 | 12 / 11 | WEAK |

Six of eight admitted to the battery stage. Frontier coherence (T1) held
on **every** axis — the frontiers see one latent in all eight wordings,
which is the transmissibility precondition. Note the wave-1 contrast:
these definitional wordings all clear T1 at 0.82–0.97, well above the bare
keyword phrasing that produced probe 1's topic-detector.

## The clean instruments

**scar_tissue_density** is the sharpest result in the program so far. The
confabulated war story (round numbers, 3 AM, "48 hours straight") ranks
**9th of 12 for both frontiers and 1st for the mini** — an 8-rank
inversion on a single item. The frontiers instead top-rank the boring
parameter values with mechanism (the $6,140-vs-$600 tokenizer bill, the
97 MB chunk, the 573° quartz-inversion soak). This is precisely the
"only contact with the territory produces this" latent, and the cheap
model provably cannot see it.

**antimemetic_payload** behaved exactly as predicted: frontiers top-rank
the shift-handoff sepsis finding and the blame-diffuse bridge post-mortem
(11, 3, 7) and dump the quotable "unpopular opinion" hot take to 10th; the
mini puts that hot take 3rd. Edginess-as-memetic-fitness is the confounder,
and only the frontiers price it correctly.

**live_wire_prose** — same shape, and the one with a named buyer. The
full-hustle founder update sits 10th for frontiers, 5th for the mini,
while frontiers top-rank the low-key items with named difficulties
(Hartmann's failing export, draft four of the grant, the fifth PCB rev).

**rosetta_load** had the largest tier gap (+0.42) but via T2, not decoy
inversion: the mini's whole ranking decorrelates (ρ≈0.53) rather than
failing on one item. Both tiers rejected the sheaf-definition decoy —
the mini simply can't order the rest.

**hostile_paraphrase_invariance** cleared on the widest frontier-mini
decorrelation after rosetta (ρ 0.54/0.45), with the frontiers doing
exactly what the axis asks: they rank the pigeonhole impossibility
argument first and drop the sober-sounding "context-dependent" nothing
to the bottom — while giving the florid antibiotic-resistance item real
credit for its quantified selection differential under the ornament.

**eschatological_seriousness** — the probe-1 remedy WORKED, by T3. The
apocalyptic mood-piece lands 7th for the frontiers and 3rd for the mini,
and frontiers top-rank the items where ultimate stakes actually reprice
something (the single-member launch veto, the abandoned tenure track, the
perpetual-trust endowment). Bare `connection_to_the_end_of_time` was a
topic detector; **the same latent under a wording that demands
action-guidance is a real axis**. That is the wave's most important
methodological finding: the wording IS the axis, which is why
`axis_prompt_hash` belongs in the ledger's provenance.

## The two WEAK verdicts — decoy failure, not axis failure

Both missed for the same honest reason: **the mini got them right too.**

- `voltage_under_load`: every tier dumped the primary decoy to 11–12th.
  Its hand-wave was lexically explicit ("we will skip the boring math"),
  so detecting it needed no ability to locate the hard part. The real
  test — a decoy that *appears* to engage the difficulty while saying
  nothing load-bearing — was never run.
- `authorial_irreducibility`: mini↔frontier agreement (0.92) actually
  *exceeded* frontier↔frontier (0.84). The quirky-blogger decoy is
  detectably contentless at any tier; and the frontiers disagreed among
  themselves more than usual, hinting the axis has genuine internal
  structure the wording doesn't yet pin down.

Both keep their [paper] status with a rewritten decoy set, not a
rewritten axis. The generalized lesson: **a decoy whose confounder is
lexically explicit tests nothing.** The confounder must be *implicit* —
present as texture, not as a phrase — or the probe measures vocabulary
detection instead of latent depth. Wave-1's cosmic-slop decoy failed the
same way from the opposite direction (its referent was too real).

## Standing

Six axes advance to stage 4 of the loop: phrasing families (`#a/#b/#c`),
the framing-invariance battery, and `canonize` transmissibility across a
wider judge panel. Two return to decoy redesign. Cumulative program cost:
$2.32 across 33 runs.

The editorial gate (criterion 6, arrest) is the operator's and has not
been run.
