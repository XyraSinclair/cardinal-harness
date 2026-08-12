# Oracle consult: logprob program roadmap (2026-08-12)

Slug: logprob-program-roadmap-consult(-retry) · oracle 0.17.1, browser mode,
direct-master profile (first live run of that architecture — session survived,
cleanup verified). Model: **GPT-5.6 Sol, Extra High effort — NOT Pro tier**
(ChatGPT Advanced-submenu picker regression; launcher auto-fell-back to the
profile's current model; picker evidence: "Model GPT-5.6 Sol, Effort Extra
High"; the footer's "gpt-5.5-pro" is oracle's requested-key label, not the
answering model; verified=no). Runtime 4m01s. Prompt: /tmp/oracle-prompt.md
(thesis + shipped results + constraints + 6-item queue). Full transcript:
~/.oracle/sessions/logprob-program-roadmap-consult-retry/artifacts/transcript.md

Oracle's answer is input, not authority — triangulation lives in the session
digest. Verbatim answer follows.

---

## 1. Verdict on the thesis

**The thesis is strong, but the thing that has been validated is one layer lower than the thing that matters.**

You have convincingly shown that, conditional on a constrained grammar and accessible logprobs, **one model call contains much more evidence than one sampled completion**, and that resampling can recover that evidence dramatically more efficiently than frequency Monte Carlo. The 47× result plus exact open-model ground truth is enough that I would stop questioning the extraction machinery itself.

The unresolved thesis is:

> **Does distribution-native elicitation move the end-to-end accuracy / calibration / cost frontier enough to matter after optimal experimental design, calibration, and strong non-logprob baselines?**

That is much harder. There are three layers:

**decoder fidelity → measurement validity → decision value**

You have basically nailed decoder fidelity. The program should now spend very little marginal effort there. Measurement validity is the dangerous layer: does the distribution induced by `D{1,3}.D`, a radix code, a verbal answer, or a binary threshold correspond to the same latent judgment? Then decision value asks whether the extra information actually reduces uncertainty in the globally fitted scores.

The north-star quantity should become something like

[
\frac{\Delta \mathcal I(S)}{$}
]

rather than enumerated PMF mass, where (\mathcal I(S)) is global information about the cardinal score vector. With approximate posterior precision (\Lambda), an edge (e=(i,j)), incidence vector (b_e), and calibrated scalar information (I_e), a natural D-optimal approximation is

[
\frac{
\log!\left(1 + I_e, b_e^\top\Lambda^{-1}b_e\right)
}{C_e}.
]

That is the control law I would eventually want the harness optimizing: **which pair, through which elicitation geometry, on which model, for how much money?**

Your current queue is too focused on extracting increasingly beautiful distributions before proving their marginal system value.

---

## 2. Single highest-leverage next move

### Run a **budgeted elicitation-geometry tournament**, not merely the queued radix-4 experiment.

Make queue items **#4 and #6 one experiment**, and expand the arms.

At minimum compare:

| Instrument                        | Stochastic output structure | Main advantage                                             | Main risk                             |
| --------------------------------- | --------------------------: | ---------------------------------------------------------- | ------------------------------------- |
| ordinary point ratio              |                 multi-token | strongest baseline simplicity                              | throws away uncertainty               |
| point ratio × repeated samples    |                 multi-token | provider-universal MC baseline                             | expensive                             |
| current decimal peel              |           3 nodes × redraws | rich PMF                                                   | surface-form/tokenization dependence  |
| single-token log-grid             |          1 categorical node | potentially whole PMF in one call when support ≤ top-k     | discretization/codebook effect        |
| radix hierarchy                   |                1 node/stage | arbitrary resolution                                       | multiple stages/instrument distortion |
| **adaptive offset-binary**        |           **1 binary node** | exceptionally clean likelihood; universal fallback         | needs response-curve calibration      |
| strong no-logprob reasoning model |              sampled answer | tells you opportunity cost of using logprob-capable models | no direct distribution                |

Do this **before** deeper provider engineering or k-wise amortization.

The particularly important new arm is adaptive offset-binary elicitation. Let

[
z=s(A)-s(B)=\log\frac{A}{B}.
]

Instead of asking for (z), present a threshold (h):

> Is (A/B > e^h)? Answer token A or B.

Read the A/B distribution. Vary (h), preferably around the current posterior location.

A reasonable psychometric model is

[
P(\text{yes}\mid z,h)
=====================

\sigma{\beta(z-h)+b}.
]

For an ordinary sampled binary answer, Fisher information is

[
I(z)=\beta^2p(1-p),
]

maximized at (p=1/2), giving (\beta^2/4). Hence adaptive placement of (h) near the current estimate is automatically information-seeking.

More interestingly, **with logprobs you read the whole Bernoulli distribution in one stochastic token**. Your observed 1–2 percentage-point probability jitter gives, locally around (p=.5),

[
\sigma_z \approx
\frac{\sigma_p}{\beta p(1-p)}
=============================

\frac{4\sigma_p}{\beta}
\approx
\frac{0.04\text{–}0.08}{\beta}
]

log units before other calibration error. That is sufficiently small that this geometry has a serious chance of beating eight-draw decimal peeling.

And when logprobs disappear, the **identical instrument** gracefully becomes Bernoulli sampling. That cross-provider invariance is strategically valuable.

The experiment should measure **loss-versus-dollars curves**, not merely which instrument most faithfully reconstructs its own output PMF. Include hard near-ties so Kendall (\tau=1) does not ceiling out.

---

## 3. Ordered 5-item roadmap

1. **Build the geometry tournament and make endpoint loss/$ the primary metric.** This has overwhelmingly highest information-per-dollar because it can invalidate whole branches of the roadmap. Cross pairs × models × geometry × prompt paraphrases × provider repeats. Measure latent-score error, pairwise predictive log loss, ranking regret, interval coverage, and dollars to fixed target error. Your exact masked-open-model setup stays as the decoder-control condition, but add tasks whose underlying quantities are external to the model. Crucially estimate **instrument × item interactions**: a constant geometry bias is calibratable; pair-dependent geometry shifts are much more damaging.

2. **Persist the credal certificate and enough raw evidence to replay every estimator.** Queue #1 is load-bearing, but as infrastructure rather than a research destination. Preserve `e_lo/e_hi/gap`, chosen-token logprobs, usable sidebands, grammar/version, prompt hash, provider/model fingerprint, temperature, and repeat identity. Do not allow today’s `(mean, variance)` collapse to become irreversible. This is cheap relative to every future experiment and prevents having to reacquire expensive model evidence.

3. **Make the solver distribution-native; simultaneously test adaptive offset-binary.** This is the major structural change. Today you harvest a distribution and then throw almost all its geometry away by mapping it to ((E[Z], \mathrm{Var}[Z])) before IRLS. Instead let each edge provide a factor
   [
   \ell_e(s_i-s_j)
   ]
   and solve
   [
   \arg\max_s \sum_e \ell_e(s_i-s_j)
   ]
   with an explicit robust contamination component where necessary. A discretized PMF becomes a piecewise edge likelihood; binary-threshold probes become logistic factors; credal mass gives likelihood envelopes. IRLS can remain the Gaussian fast path, but it should become an approximation to the evidence, not the evidence interface. This is where logprob harvesting can become qualitatively more useful than “better error bars on point ratios.”

4. **Calibrate only quantities shown by the tournament to matter.** Queue #2 moves here. Estimate (\beta), offsets, jitter, geometry effects, and repeat correlations hierarchically, out of sample. Do not merely adjust temperature until nominal and empirical variances visually agree; test whether calibrated precision weights improve held-out global recovery versus equal weights. If your sophisticated variances fail to beat robust equal weighting, that is important negative evidence about the thesis. Temperature is a treatment variable, not just a knob: run perhaps 2–3 values and ask whether it changes semantic judgment or merely entropy.

5. **Only after a geometry wins, optimize provider mechanics and amortization.** The prefill census (#3) is worth doing as a bounded side task because it is cheap, but it should not gate the program. Neyman allocation matters only once you know which subtrees deserve sampling. k-wise/Plackett–Luce (#5) comes last: it adds contextual interactions, permutation-order effects, within-call correlations, and model-assumption risk exactly when you are still trying to establish basic measurement invariance. Once pairwise evidence is decisively good, amortization becomes engineering leverage rather than another confound.

In your current queue terminology: **#6 is first; #1 happens immediately alongside it; #4 is an experimental arm, not a commitment; #2 follows the winner; #3 is opportunistic; #5 is currently procrastination.**

---

## 4. What you are structurally missing

The largest omission is not another clever grammar. It is a **common likelihood-level evidence abstraction**.

Right now the conceptual interface seems to be approximately

[
\text{model output distribution}
\rightarrow
(E[Z],\operatorname{Var}Z)
\rightarrow
\text{IRLS}.
]

I would change this to

[
\text{elicitation}
\rightarrow
\boxed{\text{edge likelihood / credal likelihood over }z}
\rightarrow
\text{global inference}.
]

Then decimal peeling, single-token buckets, radix trees, binary probes, sampled reasoning models, human judges, and future modalities all become alternative ways of minting the same object.

This matters because mean/variance collapse can make your strongest innovation invisible. Two edge distributions can have identical moments but radically different evidence:

[
0.5,\delta_{-2} + 0.5,\delta_{2}
\qquad\text{vs}\qquad
N(0,4).
]

IRLS sees essentially the same thing. A distribution-native solve does not.

On the elicitation side, I would specifically add the **offset-binary staircase** above. It has another attractive property: you can use global inference to choose the next threshold. If edge (A,B) currently has posterior (z\sim q(z)), choose (h) near a posterior median or more precisely maximize expected posterior entropy reduction. You now have something very close to Bayesian optimal experimental design.

There is also a surprisingly strong argument for a **≤20-way single-token logarithmic codebook**, given your `top_k<=20` constraint. If constrained decoding really exposes every legal option, a 16-way category carries up to four bits of categorical resolution in one output token and yields its complete first-level PMF in one call. A hierarchical 16-ary zoom gives approximately 12 bits of location resolution in three stages. I would include this in the tournament before privileging radix-4.

Binary probably wins on robustness and provider universality; 16-way categorical may win raw information/call on logprob-friendly APIs. The common likelihood interface allows the scheduler eventually to choose between them.

---

## 5. Challenge-to-thesis: strongest case this is a dead end

The strongest attack is **not** “logprobs are noisy.” You can engineer around 1–2% jitter.

It is this:

> **Token probabilities are distributions over decoder behavior under an artificial response channel, not necessarily distributions over the latent epistemic quantity you care about.**

The exactness may therefore be precision about the wrong thing.

Several mechanisms could make that fatal. Constrained numeric grammars may introduce token preferences and codebook effects large relative to genuine uncertainty. The model's probability of writing `2.7` may be very different from its credence that the true ratio is around 2.7. Provider-specific hidden machinery can alter logits without changing underlying competence. More importantly, the highest-quality reasoning models increasingly may expose no logprobs at all. If a reasoning model with one cheap point estimate beats a non-reasoning model plus a beautifully reconstructed PMF, decoder observability is strategically irrelevant.

There is another attack: **global aggregation may make local PMFs economically redundant.** Suppose robust active pair selection plus one point judgment already recovers the ordering at $0.007. Improving the statistical efficiency of each edge by 47× does not yield a 47× product advantage if the endpoint was already saturated. Your current (\tau=1.000) result cannot distinguish “excellent measurement innovation” from “this matrix was easy enough that uncertainty hardly mattered.”

And the most dangerous failure is confident wrongness. If model logprob is mostly *within-model decisiveness*, not truth-calibrated epistemic uncertainty, precision weighting can systematically promote the worst edges: exactly the ones where a model is confidently mistaken.

### What would settle it

I would preregister a kill test around the **budget frontier**.

Across several materially different domains, difficulty bands, and model families, compare the best tuned versions of logprob evidence against:

point estimates; repeated frequency-MC; adaptive experimental design without logprobs; no-logprob reasoning models; and preferably a modest model ensemble.

Plot endpoint regret against actual dollars.

Then ask one question:

> **After every baseline gets equally competent active edge selection and calibration, does logprob evidence Pareto-dominate?**

I would want something operationally unmistakable—roughly **≥2× lower cost to reach the same held-out error/coverage on multiple nontrivial domains**, or a substantial quality improvement at the same cost, reproduced across more than one logprob-capable model. Not 5% prettier calibration on one provider.

The strongest positive evidence would be that the advantage persists under **prompt paraphrase, instrument changes, repeated provider calls, and OOD items**, while the likelihood-native solver converts the extra evidence into lower downstream regret.

The strongest negative evidence would be that after calibration and active design, one-shot point judgments from stronger reasoning models sit on the same or a better loss/$ frontier, or that the harvested distributions shift materially when you change the answer grammar while underlying point judgments remain stable.

If that happens, I would keep the harvesting machinery as a specialized measurement backend, but stop treating it as the cardinal-harness moat.

The program is therefore at an unusually clean inflection point: **stop improving the microscope and make different microscopes compete on whether they improve the map.**


4m01s · gpt-5.5-pro[browser] · ↑908 ↓3.65k ↻0 Δ4.56k
