# Decimal logprob enumeration: full distributions over free-form elicited numbers

Status: **DECIDED — pack closed 2026-08-13.** The instrument was built, validated
(SHOOTOUT.md), ported to production (`src/rerank/decimal_ledger.rs`), and defended in a
geometry tournament whose three synthetic challengers all failed live validity
(TOURNAMENT.md banner → BINCDF.md / GRID16.md / RADIX4.md): native decimal peel won the
program outright. New probes do not belong in this pack; instrument work continues in the
engine. Original framing below, preserved as captured 2026-08-10 from an operator voice
note (the operator flagged this as "monumentally cool if solved" but hard to articulate —
this note is that articulation).

The shipped instrument is the indexed answer key (`src/prompts.rs`, indexed ratio ladder:
bucket integer 0..16 as a single answer token, strict `json_schema` pinning the token
position, top-k logprobs on that one position = the whole PMF). That solves distribution
elicitation by *engineering the general problem away*. This note states the general
problem: recovering the model's full predictive distribution over a **free-form decimal
output** ("2.35", "0.7", "18") from logprobs.

## One-sentence formalization

Compute the pushforward of the model's autoregressive distribution over token sequences
along the map `parse ∘ decode : token-seq → string → ℝ`, under two access constraints:
logprobs are (a) conditional on the actually-generated prefix and (b) truncated to top-k
per position.

The answer key is the degenerate case: depth-1 tree, `parse ∘ decode` bijective on the
support, alignment guaranteed by construction. The general instrument buys arbitrary
resolution at the price of probe calls and explicit uncertainty slack. Both are the same
mathematical object.

## The five obstructions

1. **Token/digit misalignment.** Tokenizers do not split numbers digit-by-digit. `2`,
   `25`, `2.5`, `235` can all be *single-token* alternatives at the same position. The
   top-k list at one position straddles different digit counts and different magnitudes,
   so a per-position logprob read is not a PMF over any numeric quantity. There is no
   position-indexed "digit distribution" to read off.

2. **Conditional access only.** The API returns per-position top-k alternatives
   conditioned on the prefix that was actually sampled. The alternatives are counterfactual
   branch *roots*; what follows them is unobserved. Recovering mass below a branch requires
   a forced-prefix continuation probe. The full distribution is therefore a **tree of API
   calls** (one per expanded interior node), not a field in one response.

3. **Prefix mass lives on value-sets, not values.** Digit-prefix `2` covers
   [2,3) ∪ [20,30) ∪ [200,300) ∪ … Until a branch is expanded to termination, its mass
   cannot be assigned to a value — only to a set of values. A partially expanded tree
   therefore yields a **credal PMF**: point masses `(v, [lo, hi])` plus a ledger of
   set-assigned mass. This structure is not an implementation inconvenience; it is the
   correct output type, and it is what makes budget-bounded computation honest.

4. **Two levels of many-to-one consolidation.** Token paths → digit strings (non-canonical
   tokenizations of the same string carry real chain-rule mass), and digit strings → values
   (`2.50` = `2.5`; `+3` = `3`). The right structure is the token trie **quotiented by
   decoded string**, then pushed forward through the parser. Consolidation must merge by
   value, never by token path.

5. **Termination and truncation are both random/ lossy.** "The number ended here" is itself
   a probability (mass of non-digit continuations at each node) and needs the same probe.
   And the top-k cap (hard 5 on GPT-5.x at `effort: none` — measured in
   `notes/adaptive-logprobs-2026-07-19/findings.md`) leaves residual mass `1 − Σ top-k` at
   every node, compounding multiplicatively down paths. The ledger must carry residuals as
   explicit slack. Silent renormalization is the success-shaped-fallback anti-pattern in
   probability clothing.

## Design sketch

- **Structure:** token trie, quotiented on the fly by decoded string. Node state:
  forced token prefix, decoded digit-string prefix, exact log-mass, residual slack.
- **Expansion:** best-first by reachable-mass upper bound, budget-capped. A node probe is
  one forced-prefix request with strict `json_schema` pinning the position (measured
  2026-07-19: strict schema preserves logprobs and pins token positions; loose
  `json_object` does not).
- **Sampling as exploration, probing as measurement.** Temperature-1 samples locate heavy
  hitters cheaply; forced-prefix probes make their masses exact; the residual bound closes
  the ledger. The two compose: samples propose, probes measure, slack accounts.
- **Output type:** credal PMF `{(v, [lo, hi])}` + set-mass ledger. Every downstream
  statistic (log-ratio mean, quantiles, tail mass) is computed as a `[lower, upper]`
  envelope against this structure. Denominator discipline falls out for free.
- **Stopping:** stop when the envelope width of the *target statistic* is inside
  tolerance, or budget is exhausted. Converging a log-mean usually needs far less than the
  full distribution — most mass concentrates in a few heavy-hitter values.
- **Cost:** probes share the prompt prefix → prompt-cached; marginal node cost is small.
  Tree width is bounded by top-k (5). Depth is bounded by number length.
- **Nesting / multidimensionality:** joint elicitation (direction × ratio, or several JSON
  fields) is the same object composed — a tree of credal PMFs keyed by field path, later
  fields conditioned on earlier fields' realized prefixes. The "crazy multidimensional
  consolidated structure" is exactly this: nested credal pushforwards.

## Open instrument questions (each needs a dated probe, census-style)

1. **Forced-prefix support varies by provider.** Anthropic supports exact assistant
   prefill; OpenAI chat needs the prompt-echo trick (partial answer echoed into context —
   a slightly different conditioning, which is an instrument caveat, not a detail);
   OpenRouter passthrough per provider is unmeasured. Needs a census like the 2026-07-19
   logprob-unlock census.
2. **Off-manifold forcing.** Forcing a non-canonical token prefix conditions the model on
   a sequence it would rarely emit. The chain rule still makes the measured mass real, but
   providers may mangle it. Verify: Σ over tokenizations vs. observed sampled frequency on
   a controlled digit string.
3. **Is top-5 too lossy for depth?** Residual per node can be up to `1 − Σ top-5`.
   Hypothesis: digit-position distributions are peaked and residuals are tiny; measure the
   typical per-node residual on real elicitations before declaring the instrument viable.

## Relation to existing code

- `src/prompts.rs` indexed ladder + `notes/adaptive-logprobs-2026-07-19/` are the answer-key
  instrument and its capability census.
- `src/gateway/` already carries logprob plumbing and per-provider capability volatility
  lessons (capability is probed, dated data — hard-coded gates rot in under 24 h).
- Nothing here is scheduled work. Promotion to an issue requires a frozen fixture and a
  bounded tranche per repo law (`AGENTS.md`).
