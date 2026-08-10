# The descend/peel kernel: calibrated value distributions from constrained logprob access

Status: design, reviewed and substantially upgraded from `PROBLEM.md` (same directory,
same day). PROBLEM.md stated the obstructions; this note is the plan. Tokenizer facts
below are measured (o200k_base, 2026-08-10, this directory's session). Oracle consult
in flight; reconcile on return.

## What changed since PROBLEM.md (the review findings)

PROBLEM.md's framing was right but missed five load-bearing facts. Each one collapses a
piece of the difficulty:

1. **Samples ARE probe-paths.** A temperature sample with `logprobs: true,
   top_logprobs: 5` returns the top-5 conditional PMF *at every position along the
   sampled path*. One call = one root-to-leaf trajectory + exact top-5 conditionals at
   every node on it. The PROBLEM.md dichotomy "sampling as exploration, probing as
   measurement" is not two call types — it is one call type. The whole algorithm is
   built from a single primitive:

   `descend(prefix)` = prefill `prefix`, sample to termination, harvest per-position
   top-5. Cost: one prompt-cached call.

2. **Width-nesting via logit_bias is exactly recoverable.** Suppressing a known token
   set S (bias −100) renormalizes the surviving distribution by 1/(1−m_S) where m_S is
   the already-measured mass of S. So `peel(node)` = re-probe with top-5 suppressed →
   next-5 masses, un-renormalized exactly: p(t) = p'(t)·(1−m_S). Repeat for top-10,
   top-15… Top-K at any node in ⌈K/5⌉ calls. Depth-nesting (prefill) and width-nesting
   (peeling) are orthogonal axes; PROBLEM.md only had depth.

3. **The off-manifold problem dies by construction, not by empirical bounding.** Hold
   the model's tokenizer client-side (tiktoken/HF — known for every model we route).
   Only branch/prefill at *canonical* token boundaries: a prefill string is safe iff
   its canonical tokenization is a prefix of the canonical tokenization of its
   plausible completions. Unsafe boundaries (mid-digit-group) are never prefilled —
   the group's position is peeled instead. No non-canonical forcing ever happens, so
   no off-manifold conditioning and no exponential tokenization marginalization.
   (Non-canonical emission mass by the model itself remains a measurable footnote —
   it shows up as weird tokens in top-5 and is consolidated by decoded string.)

4. **Measured tokenizer geometry (o200k_base, 2026-08-10): the tree is shallow-wide,
   not deep.** Every 1–3 digit string is a single token (all 1110 checked, zero
   exceptions); the decimal point is always its own token; JSON context tokens never
   merge into the number. So `"ratio": 2.35` has number structure
   `[2][.][35]` — depth ≤ 4 positions for any ≤3-int-digit, ≤3-frac-digit format.
   The difficulty is WIDTH: the fraction is ONE position with up to 10^d siblings,
   and top-5 sees only the peak. Depth search (PROBLEM.md's imagined hard part) is
   trivial; peeling budget is the real resource question.

5. **The grammar is the instrument.** Strict json_schema (pattern-constrained string
   or bounded number) does three jobs at once: pins token positions, bounds the value
   support (making statistic envelopes finite — prefix "2" under pattern
   `^[0-9]{1,2}\.[0-9]$` covers [2,3)∪[20,30) only), and sets the
   resolution/width tradeoff (1 frac digit = 10-way position, 2 = 100-way). The
   answer-key ladder, the digit-grid, and free-form decimals are the SAME engine run
   with three grammars:
   - answer key: depth-1 grammar over 17 single-token strings (today's instrument);
   - digit-grid (`2 . 3 5`, space-separated): every boundary safe, every position
     ≤10-way — zero peeling needed, at some naturalness cost;
   - free-form decimal: shallow-wide, needs peeling.
   One kernel, grammar-parameterized. That is the productizable abstraction.

## The kernel

### State: string-trie with a mass ledger

Nodes are *decoded-string prefixes* (not token paths), with super-edges spanning
unsafe boundaries. Each visited node holds its exact top-k conditional PMF and
truncation residual. Global ledger, maintained exactly at all times:

    enumerated + frontier + truncation_residual = 1

- **enumerated**: complete values with exact path mass (products of measured
  conditionals down safe boundaries);
- **frontier**: measured-mass branches not yet descended — exact mass, value-SET
  support (from grammar × digit prefix);
- **truncation_residual**: Σ over visited nodes of path_mass × node residual —
  known mass, support = "anything not yet seen at that node" (grammar-bounded).

Every call moves mass monotonically from residual/frontier into enumerated. Stopping
anywhere yields a sound credal PMF: point masses plus set-assigned slack. This
conservation invariant is the product's integrity guarantee — double-entry
bookkeeping for probability mass, anytime-valid.

### Moves

- `descend(b)`: prefill frontier branch b's string (safe boundary), temperature ~0,
  logprobs on. Resolves b's modal completion path exactly and exposes top-5 at every
  new node. Converts set-mass → point mass + smaller frontier.
- `peel(n)`: logit_bias-suppress n's known tokens, re-probe. Converts truncation
  residual → frontier/point mass. Exact un-renormalization as above.
- `validate(m)`: m temperature-1 samples from root. Not part of the estimate —
  an end-to-end calibration self-check: empirical frequencies vs enumerated masses
  (multinomial test). Ships in the report. An instrument that checks itself is the
  feature labs don't have.

### Policy

Greedy index: spend the next call where expected envelope-width reduction of the
TARGET statistic per call is largest. For frontier b: contribution ≈ mass_b ×
diam_θ(support_b); for peel at n: ≈ path_mass_n × residual_n × diam_θ(support_n).
Statistic-aware (for E[log r], diameter is in log space — one resolved digit shrinks
it geometrically). Both move types priced identically (one call each), so this is a
plain knapsack-greedy; the tree is so shallow that optimality gaps are negligible.

### Stopping

Stop when envelope width of θ < tolerance, or budget out. Report:
- consolidated PMF {(v, p)} with per-value exactness flags,
- envelope [θ_lo, θ_hi] (adversarial placement of slack within supports),
- point estimate under declared imputation (residual spread ∝ empirical digit prior,
  hierarchically shrunk across nodes) + sensitivity to the imputation,
- uncertainty decomposition: truncation slack vs frontier slack vs imputation spread,
- calibration self-check result.

### Cost shape (why this is product-viable)

Ratio grammar `^[0-9]{1,2}(\.[0-9]{1,2})?$`: depth ≤ 4. Typical run: 1 root descend
+ 1–2 peels at the int position + 2–4 descends into surviving int branches + 1–2
peels at their frac positions ≈ **6–12 prompt-cached calls** for ≥95–99% enumerated
mass. At GPT-5.x-mini pricing with caching that is order $0.001 per full calibrated
distribution — roughly 10× a single sampled judgment, for arbitrary-resolution PMFs
the ladder cannot express. The IRLS solver consumes this directly as (μ, σ²) with an
honest σ² instead of unit precision.

## Regime ladder (the "kernel for labs" story)

Same engine, same credal output type; only the access tier changes tightness:

| tier | access | behavior |
|---|---|---|
| L0 | full logits + prefill (lab-internal inference stack) | k = vocab: no truncation residual, exact pushforward, envelope width → 0 |
| L1 | top-k logprobs + prefill + logit_bias (OpenAI-class) | this note's full kernel |
| L2 | top-k logprobs, no prefill | root descents + peeling only; frontier stays credal deeper |
| L3 | samples only (Anthropic-class) | same trie, Dirichlet-tree (Pólya tree) posterior from path counts; envelopes become posterior credible envelopes |

The sale: "bring your inference stack, get calibrated elicited distributions with a
conservation-law integrity guarantee; the same API degrades gracefully to public
constrained access." L0 is also the validation harness for L1–L3: run both on the
same model, measure envelope soundness empirically. That evidence pack IS the
product's credibility.

## Probe sequence (de-risk order, each a dated census entry)

1. **Grammar-mask semantics** (cheapest, decides estimator core): under strict
   json_schema, are returned logprobs renormalized to grammar-legal tokens or raw?
   Test: probe a position where an illegal token would plausibly rank top-5; check
   presence. Either answer is workable; the ledger needs to know which.
2. **logit_bias liveness on GPT-5.x at effort none** (decides whether peeling exists
   on the flagship rail): bias a known top token to −100, confirm it vanishes and
   survivors renormalize as predicted by its measured mass. GPT-4.1 as control.
3. **Prefill passthrough census via OpenRouter** (decides L1 vs L2 per provider):
   assistant-prefix / continue_final_message support per provider; verify measured
   conditional at a prefilled safe boundary matches the same node reached by root
   sampling (chain-rule consistency check — this doubles as the off-manifold
   verification).
4. **Residual magnitude survey**: distribution of per-node top-5 residual across real
   elicitations (int position vs frac position). Decides default peel budget.
5. **End-to-end**: one full kernel run vs 500 temperature-1 samples on the same
   prompt; compare enumerated PMF vs empirical frequencies. The calibration
   self-check, run once as a validation study.

## Honest unknowns

- Naturalness cost of grammar constraint on judgment quality (constrained vs free
  answers may differ as *measurements*, not just formats) — same instrument-identity
  doctrine as effort-none vs sampled (adaptive-logprobs note, §2.3).
- Temperature-0 descents assume the provider honors temperature at effort none;
  if not, descents still work (any sampled path resolves mass), just less
  call-efficient.
- Provider nondeterminism (mixture backends) can make two probes of the same node
  disagree slightly; the ledger should store per-node probe provenance and treat
  disagreement as measured noise, not corruption.
- o200k facts are per-tokenizer; the boundary-safety module needs the actual
  tokenizer per routed model (HF for non-OpenAI). Client-side, cheap, but must be
  wired into routing.
