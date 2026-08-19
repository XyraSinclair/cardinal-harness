# Whitespace-jitter probe battery — smoke validation (2026-08-19)

**Claim tested:** deterministic whitespace jitter in the attribute prompt
(probe k widens 1–3 seed-chosen word gaps; blake3(text,k), pure function)
yields K distinct, individually cached, replayable draws of the same
structured judgement — unblocking `repeat_pooling` (DL heterogeneity
floor), which previously starved because the content-addressed cache can
hold only one judgement per identical prompt.

**Instrument:** `cardinal probe` (experiments/src/probes.rs).

**Run:** deepseek/deepseek-v4-flash, canonical_v2, 6 entities (bench-corpus
subset, `corpus.txt`), stride-1 ring (6 pairs), K=6 → 36 calls, $0.0028.
Committed cache: `probe-cache.sqlite` (replay:
`cargo run -p llmsort-experiments --bin cardinal -- probe corpus.txt --by
"depth of insight about living well" --model deepseek/deepseek-v4-flash
--probes 6 --cache probe-cache.sqlite` → 36/36 cached, $0).

**Measured (this pack, `report.json`):**
- duplicate rate 17% — jitter moves the answer on 5/6 of repeat draws;
  real dispersion, σ_w² 0.487.
- σ_b² 0.039 > 0 — the DL floor engages even at n=6.
- Pair 0-1 (obstacle-is-the-way vs suffer-in-imagination) split 2/4 on
  DIRECTION across probes — a pair that single-probe elicitation commits
  to with false confidence. Reproduced across runs (3/3 in the first).
- naive-vs-floored solve ρ = 1.000 at this scale (ranking unchanged;
  the floor changes variances, not order, on a clean ring).

**Cross-run honesty (denominator):** an earlier independent run of the
same 36 prompts into the shared cache measured duplicate rate 40%, σ_w²
0.190. Same-prompt re-elicitation draws fresh samples (provider does NOT
response-cache at this temperature), and a 36-call battery estimates σ_w²
only to within ~2.5×. Consequences: (1) jitter's necessity is LOCAL —
without it the llmsort cache stores exactly one draw per pair and replay
dies; (2) σ_w² numbers from single small batteries are indicative, not
citable.

**Next rungs:** K-vs-precision curve (does pooled sd shrink ~1/√K or hit
the σ_b² floor); jitter-vs-plain-resample A/B on a no-cache rail (is
jittered dispersion statistically the same as pure resampling dispersion —
if yes, jitter is a free cache-identity trick, not a perturbation cost);
wire pooled draws into sort as an opt-in repeat mode.
