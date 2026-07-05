# JCB test–retest (2026-07-05) — how much of the board is noise?

The adversarial review demanded a test–retest reliability number before
any leaderboard claim; this pack is it. The full 138-call battery re-run
fresh (`--no-cache`) on all six models, compared to the v1.1 run
(`comparison.txt` has per-axis deltas; both runs' raw judgements are in
their packs). At temperature 0, deltas measure provider-side
nondeterminism plus nothing else — the corpus, prompts, and seed are
identical.

## Verdict

**Reproducible:** mean |ΔJUDGE| = 0.022, max 0.064. Rank order identical
in the top 3; mini/deepseek swap at ranks 4–5 (their gap, 0.05, was
already below the documented treat-as-tie threshold). claude-haiku-4.5
reproduces EXACTLY (every axis delta 0.000 — deterministic serving).
Spin survival identical for all six models. Anthropic
formatting-blindness replicates (haiku 0.080→0.080, sonnet 0.092→0.080
nats). gpt-5.4-mini's spin survival held at 2/3 across both fresh runs —
supporting the earlier attribution of the 1/3→2/3 change to the framing
wording fix, not sampling luck.

**NOT reproducible — erratum:** gpt-5.4-nano's polarity Spearman flipped
from **+0.38 to −0.40** between runs. The v1/v1.1 finding "nano ranks
shallowness like depth" was an over-claim from a single run. The
replicated finding is: **nano's polarity relation is unstable at ±0.4
across identical runs** — its opposite-attribute scores are noise, which
is a coherence failure of a different and arguably worse kind (every
other model's polarity moved ≤ 0.10). Errata appended to both earlier
pack READMEs; the corrected statement is the only one that should be
quoted.

Also noisy on the OpenAI smalls: mini polarity +0.29 and paraphrase
+0.21 deltas — the attribute-relation axes carry the most provider noise
at this corpus size; direction axes (flip, spin, curl) are tight for
everyone.

## Cost of the answer: $0.47. The benchmark's credibility number should
be re-earned like this on every version bump.
