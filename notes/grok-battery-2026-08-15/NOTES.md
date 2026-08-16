# Grok battery v1 — does a judge model reasonably grok an attribute? (2026-08-15)

Three judges, six attribute cells, two independent seeds per cell (cache
bypassed for measurement; every judgment trace-recorded). Runner + corpora +
raw traces: `~/Projects/scratch/grok-battery-2026-08-15/`.

## The metric doctrine

Attributes worth tagging by are generally transitive, so "the model groks the
attribute" is measured as:

- **flip%** — direction coherence: of counterbalanced pairs, how many changed
  verdict with presentation order (position bias);
- **cyclic%** — transitivity: share of solution energy in cyclic (A>B>C>A)
  inconsistency;
- **retest ρ** — stability: Spearman between two fully independent runs;
- **ρ_truth** — truth recovery, on cells with a known answer (mass, lifespan,
  top speed, historical antiquity).

Low flip% + low cyclic% + high retest ρ = grokked; ρ_truth confirms where
truth exists. Soft attributes (Manifund's "existential seriousness",
"technical depth") ride the first three alone.

## Scorecards

**gemma4-26b-a4b** (128-expert MoE, ~4B active, fp8 online quant, local :8023):

| cell | ρ_truth s1/s2 | retest ρ | flip% | cyclic% | wall (2 runs) |
|---|---|---|---|---|---|
| animals_mass | 0.985 / 0.976 | 0.985 | 0.0 | 9.9 | 21.0s |
| animals_lifespan | 0.965 / 0.979 | 0.991 | 10.9 | 3.0 | 7.7s |
| animals_speed | 0.888 / 0.888 | 0.974 | 31.7 | 4.9 | 7.1s |
| events_antiquity | 0.992 / 0.988 | 0.995 | 7.4 | 8.0 | 9.9s |
| manifund_serious | — | 0.926 | 24.2 | 10.5 | 29.2s |
| manifund_depth | — | 0.972 | 10.0 | 5.9 | 29.2s |

**qwen38-27b** (dense hybrid GDN, fp8, canonical_v2, non-thinking):

| cell | ρ_truth s1/s2 | retest ρ | flip% | cyclic% | wall |
|---|---|---|---|---|---|
| animals_mass | 0.812 / 0.812 | 1.000* | 3.1 | 6.0 | 27.2s |
| animals_lifespan | 0.929 / 0.929 | 1.000* | 6.7 | 3.3 | 23.5s |
| animals_speed | 0.753 / 0.732 | 0.985 | 21.0 | 3.2 | 24.4s |
| events_antiquity | 0.952 / 0.914 | 0.926 | 12.0 | 10.5 | 34.1s |
| manifund_serious | — | 0.954 | 21.4 | 10.9 | 101.4s |
| manifund_depth | — | 0.956 | 21.5 | 6.3 | 103.5s |

\* deterministic judge + coinciding pair plans at budget 64 make seed retest
trivially 1.0 on small corpora; an unseeded 08-14 run of the same cell hit
0.976 vs today's 0.812 — plan selection matters at small budgets. Open probe.

**claude-sonnet-5** (claude-code subscription rail, $0 marginal, ~1s/cmp):

| cell | ρ_truth s1/s2 | retest ρ | flip% | cyclic% | wall |
|---|---|---|---|---|---|
| animals_mass | 0.953 / 0.879 | 0.894 | 6.1 | 7.5 | 99.0s |
| events_antiquity | 0.988 / 0.982 | 0.982 | 13.8 | 7.2 | 116.9s |

## Verdict: gemma4-26b-a4b takes the judge slot

- Best or tied-best ρ_truth on every ground-truth cell — including beating
  claude-sonnet-5 on both shared cells.
- 3–4× faster wall than qwen38 (MoE 4B-active decode) and ~10× vs Sonnet.
- **Prefix caching works: 74.4% token hit rate during the battery**
  (481888/647798). qwen38's hybrid GDN arch auto-disables vLLM prefix caching
  (`enable_prefix_caching=False` forced, counters hard-zero across 1000+
  requests) — a structural strike for high-perturbation judging, where the
  shared system+attribute prefix should be nearly free.
- Weak spots to watch: animals_speed flip 31.7% (fuzzy truth: mixed
  locomotion modes), manifund_serious retest 0.926 with flip 24.2% (soft
  criterion; qwen was steadier there, 0.954).
- 2 refusals on events_antiquity (qwen: 0).

Lane state after the battery: **gemma4-26b-a4b is the serving judge** on
colo2 :8023 (`/data/models/launch_gemma4.sh`, same co-tenant budget as the
qwen recipe; qwen relaunch is one command if wanted). Qwen's weights stay on
disk.

## A4B vs dense 31B: the consistency study (2026-08-15, second pass)

Operator question: is the A4B genuinely more consistent than the dense model
across a range of attributes, especially subtle high-dimensional ones? Premier
test corpus: Manifund grant proposals. Four subtle attribute cells added
(epistemic rigor of the theory of change, counterfactual impact of a marginal
dollar, neglectedness, tractability as proposed by this team), run on both.

**gemma4-31b dense** (fp8 online quant, 2048 ctx — KV pool only 2,348 tokens
on the shared judge slot, see serve note below):

| cell | ρ_truth s1/s2 | retest ρ | flip% | cyclic% | wall |
|---|---|---|---|---|---|
| animals_mass | 0.932 / 0.932 | 1.000 | 0.0 | 5.2 | 21.3s |
| animals_lifespan | 0.988 / 0.988 | 1.000 | 10.0 | 1.7 | 19.4s |
| animals_speed | 0.850 / 0.859 | 0.991 | 31.2 | 4.0 | 19.8s |
| events_antiquity | 0.989 / 0.989 | 1.000 | 0.0 | 6.5 | 25.0s |
| manifund_serious | — | 0.869 | 15.7 | 7.8 | 78.1s |
| manifund_depth | — | 1.000 | 8.7 | 4.3 | 80.2s |
| manifund_epistemic | — | 0.966 | 14.2 | 8.6 | 78.7s |
| manifund_counterfact | — | 0.995 | 33.0 | 12.2 | 80.2s |
| manifund_neglect | — | 0.957 | 21.9 | 6.9 | 81.5s |
| manifund_tractable | — | 0.958 | 19.3 | 9.4 | 80.5s |

**A4B on the four subtle cells** (for the pairs already in the table above):
epistemic 0.941 / flip 15.6, counterfact 0.940 / flip 21.5, neglect 0.900 /
flip 38.0, tractable 0.916 / flip 27.6.

**Cross-model agreement** (`analyze_consistency.py`; cross_rho = mean Spearman
over the four seed-pairings; disparity = mean within-model retest − cross_rho,
i.e. how much of a stable ranking is model-idiosyncratic):

| cell | retest A4B | retest 31B | cross ρ | disparity |
|---|---|---|---|---|
| animals_lifespan | 0.991 | 1.000 | 0.975 | 0.021 |
| animals_mass | 0.985 | 1.000 | 0.916 | 0.076 |
| animals_speed | 0.974 | 0.991 | 0.768 | 0.215 |
| events_antiquity | 0.995 | 1.000 | 0.978 | 0.020 |
| manifund_depth | 0.972 | 1.000 | 0.942 | 0.044 |
| manifund_epistemic | 0.941 | 0.966 | 0.880 | 0.074 |
| manifund_serious | 0.926 | 0.869 | 0.882 | 0.016 |
| manifund_neglect | 0.900 | 0.957 | 0.821 | 0.107 |
| manifund_tractable | 0.916 | 0.958 | 0.828 | 0.109 |
| manifund_counterfact | 0.940 | 0.995 | 0.620 | **0.348** |

Findings:

- **The dense 31B is the more self-consistent judge on 9 of 10 cells** —
  higher retest ρ everywhere except existential seriousness (0.869 vs 0.926),
  and its budget-240 plans genuinely differ across seeds, so its 1.000s on
  manifund_depth (40 items, 480 fresh comparisons) are earned, not the
  qwen-style plan-coincidence artifact.
- ρ_truth splits: lifespan/speed-adjacent cells comparable, mass favors A4B
  (0.98 vs 0.93), lifespan favors dense (0.988 vs 0.97). No blowout either way.
- **Stability is not shared meaning.** On "counterfactual impact of a marginal
  dollar" both models are individually rock-stable (0.940 / 0.995) yet agree
  with each other at only ρ 0.62 — each has confidently reified a *different*
  construct. Highest flip% of the subtle cells on both models too (21.5 /
  33.0): the attribute itself is under-determined by the corpus blurbs.
  Subtle high-dimensional attributes need either sharper criterion phrasing or
  multi-model triangulation before their rankings are treated as attribute
  truth rather than model taste.
- Concrete cells cross-agree at 0.92–0.98 (speed 0.77 — the fuzzy-truth cell
  again). The subtle Manifund tier sits at 0.62–0.94.

Serve note: dense 31B runs `/data/models/launch_gemma4_31b.sh` (frac 0.375,
ctx 2048 — fp8 weights ~34GB against a ~36GB free ceiling leaves a 2,348-token
KV pool). That required capping cardinal's judge output budget:
`CARDINAL_PAIRWISE_MAX_OUTPUT_TOKENS=1024` (env override added in ratiometer
387b2f8) since vLLM rejects max_tokens > max_model_len. Even with the tiny
pool: 61.4% prefix-cache hit rate, 80s per 480-comparison manifund cell pair
(~3× A4B's 27s; both fine).

Judge-slot verdict after the second pass: **A4B keeps the slot on
throughput** (3× faster, bigger KV/ctx headroom for longer corpora), but the
dense 31B is the better *instrument* — use it as the second voice when a
subtle-attribute ranking matters, and treat low cross-model ρ as the signal
that an attribute needs rephrasing. Both stay on disk; swap is one launch
script either way.

## Dense pair + logprob harnessing (2026-08-15 evening, operator decree)

A4B retired from the judge slot (kept on disk). The standing pair is the two
DENSE models — qwen3.8:27b and gemma4:31b — on the theory that a dense model
carries one coherent latent per subtle attribute where a sparse MoE may route
different aspects to different experts.

Logprobs are now first-class: `--template canonical_bucket_v1` makes every
judgment a single structured emission whose answer-token logprobs yield a full
posterior (direction PMF, ratio-bucket PMF, signed-ln-ratio distribution,
entropy). `judge_and_land.py --template canonical_bucket_v1` lands
dir_prob/entropy/top_prob/neighborhood_prob plus the serialized PMF into
`ratiometer.judgments`. Verified: 2,750/2,806 qwen judgments carry the PMF.

**Highdim pilot** (12 fascinating attributes: self-containedness, high-status,
low-status, poshness, technical calmness, earnestness, intellectual density,
legibility-to-an-outsider, coiled potential energy, craftedness, institutional
insiderness, rewards-a-careful-re-read; bare + simple-elaborated forms; both
judges; all landed with PMFs):

- Mean bare cross-judge agreement 0.757 — the fascinating tier sits where the
  subtle Manifund tier does, well below concrete attributes.
- **Elaboration rescues the vaguest attributes**: institutional insiderness
  0.625→0.833, coiled potential energy 0.647→0.854. Mean effect +0.034
  (8/12 improved). Counterexample: earnestness 0.870→0.756 — elaborating an
  attribute both judges already shared *overwrote* the shared construct.
  Doctrine: elaborate where bare agreement is low; leave high-agreement
  attributes bare.
- **Entropy signatures differ by 4×**: qwen ~2.3 nats/judgment, gemma ~0.6.
  Gemma commits hard; qwen spreads its PMF. Same rankings either way — but
  qwen's posteriors carry more usable uncertainty for the solver.

Lockstep doctrine (`scripts/dual_judge_lockstep.py`): both judges advance
attribute-by-attribute together (per-attribute lag ≈ one cell wall, 30–60s).
True lockstep needs both endpoints co-resident; the PRO 6000 currently hosts
~36GB of production rerank/embed, so co-residence requires evicting those —
Xyra's call, flagged, not taken. Until then: alternating sweeps per model
(one full sweep apart), swap = one launch script.

## Production doctrine going forward

- Measurement runs bypass the pairwise cache (independence) but always
  `--trace`; production judging keeps the cache ON so every judgment is paid
  for once — the cache is content-addressed per (pair, attribute, model,
  template).
- The relentless-judging loop wants: cardinald + ClickHouse landing for
  durable judgment records, attribute battery grown corpus-first (attributes
  chosen for expected transitivity), and prefix-ordered prompts (shared
  system+attribute head) to ride the 74% cache hit rate.
