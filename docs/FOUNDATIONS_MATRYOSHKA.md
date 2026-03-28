# Foundations, Matryoshka-Style

This document explains the same system at increasing compression levels.
Each layer contains the previous one in expanded form. The goal is not merely
to restate the same claims with more words, but to unpack additional structure,
assumptions, justifications, and caveats while preserving a single stable core.

<matryoshka_document>
  <purpose>
    Provide nested explanations of cardinal-harness at increasing token budgets,
    using moderate XML to make the semantic structure explicit without turning
    the document into a schema exercise.
  </purpose>

  <core_thesis>
    cardinal-harness is infrastructure for extracting quantitative latent
    judgements from LLMs by asking pairwise ratio questions, solving the noisy
    comparisons into globally consistent latent scores in log-space, estimating
    uncertainty, and adaptively spending additional model calls only where the
    top-K decision remains ambiguous.
  </core_thesis>

  <summary target_tokens="100" actual_style="dense">
    cardinal-harness turns vague LLM judgements into quantitative latent scores.
    Instead of asking for unstable absolute ratings, it asks pairwise ratio
    questions: how much more of some attribute does A have than B? Those ratios
    become noisy log-space observations on a comparison graph. A robust solver
    (IRLS with Huber loss) fits globally consistent scores while downweighting
    outliers. Posterior uncertainty estimates show which items remain poorly
    resolved. The planner then asks the next most decision-relevant pair,
    focusing on uncertainty near the top-K boundary, and stops when the ranking
    is sufficiently certain, rather than after an arbitrary number of calls.
  </summary>

  <summary target_tokens="272" actual_style="dense">
    cardinal-harness is a measurement system for LLM-mediated judgement. The
    central design choice is to avoid direct scoring. Absolute prompts like
    "rate this 1-10" are anchored, poorly calibrated, and inconsistent across
    contexts. Instead, the system asks pairwise ratio questions: not merely
    which item is better, but by roughly how much on a bounded geometric ladder.
    Ratio judgements are useful because their logarithms compose: if A/B and B/C
    are known, they imply information about A/C. That turns a set of comparisons
    into a linearizable inference problem over latent scores.

    Each comparison is treated as a noisy edge observation in a graph whose
    nodes are items. Confidence determines edge weight. The solver uses robust
    iteratively reweighted least squares with Huber loss, because LLM judgements
    contain occasional sharp outliers that ordinary least squares would let
    dominate the fit. Once scores are estimated, the system also estimates
    uncertainty, which matters as much as the means because the product goal is
    not just to rank items but to know when the ranking is reliable enough to
    act on.

    Query planning is therefore adaptive. Rather than compare everything with
    everything, cardinal-harness spends budget where the expected value of
    information is highest, especially around the top-K boundary. The stopping
    rule is similarly decision-oriented: stop when the chance of a material
    inversion near the selection frontier is small enough.
  </summary>

  <summary target_tokens="739" actual_style="dense">
    cardinal-harness exists because LLMs often know more than they can stably
    express through direct scalar ratings. If you ask a model to score a set of
    essays from 1 to 10, the numbers look quantitative but behave badly. They
    drift with order effects, compress toward familiar anchors, and fail to
    preserve meaningful differences across contexts. The system responds by
    changing the primitive question. Instead of asking for an absolute score, it
    asks for a relative magnitude judgement between two concrete items on a
    specific attribute: how many times more of that attribute does A have than B?

    That primitive is valuable for three reasons. First, it is local: the model
    sees two concrete objects and compares them directly instead of reconstructing
    a hidden global scale. Second, it is cardinal rather than merely ordinal: a
    ratio says not only who wins but by what rough factor. Third, after taking
    logs, multiplicative ratios become additive differences, which means the full
    set of observations can be solved as a noisy graph-structured regression
    problem. The comparison graph is the real object. Items are nodes. Each LLM
    response is an edge carrying an estimated log-difference plus a weight.

    Once framed this way, the rest of the system follows. A consistent latent
    score vector should explain the observed edges as well as possible, but LLM
    comparisons are not clean measurements. Some are merely noisy. Some are
    genuinely bad. The model may misunderstand, hallucinate, get distracted by
    irrelevant style features, or overstate its certainty. That is why the
    solver uses IRLS with Huber loss instead of plain least squares. Small
    residuals are treated quadratically, preserving efficiency when the data is
    mostly well-behaved. Large residuals are downweighted, preventing a handful
    of pathological comparisons from warping the global ranking.

    The output of the solver is not just a ranking. It is a latent score for
    each item together with uncertainty estimates. This is essential because the
    actual operational problem is almost never "produce a total ordering of all
    N items with perfect fidelity." It is usually "identify the best few items
    with enough confidence to stop spending money." The uncertainty model lets
    the planner distinguish between well-separated items and unresolved frontier
    cases. That in turn supports adaptive querying: choose the next comparison
    that most improves the decision-relevant part of the posterior, rather than
    wasting calls on already-settled parts of the ranking.

    The planner therefore targets top-K uncertainty, not uniform uncertainty.
    It uses graph structure and boundary risk to decide where more information is
    worth buying. The stopping rule does the mirror image: stop when the
    remaining probability of a consequential inversion near the K / K+1 boundary
    is small enough. This makes cardinal-harness not just a solver, but a full
    elicitation engine: prompt, compare, robustly aggregate, quantify
    uncertainty, adaptively query, and stop on a decision criterion rather than
    a fixed budget.
  </summary>

  <summary target_tokens="2008" actual_style="dense">
    <section name="Problem framing">
      cardinal-harness starts from a simple but consequential observation: LLMs
      can often make comparative judgements more reliably than they can emit
      calibrated absolute scores. A direct prompt such as "rate this proposal
      from 1 to 10 on strategic clarity" appears efficient, but it collapses
      several hard problems into one unstable number. The model must invent or
      recall a latent global scale, decide where the current item falls on that
      scale, maintain consistency across many items seen at different times, and
      map uncertainty into a single scalar. In practice this produces familiar
      pathologies: anchoring, compression toward the center of the scale,
      sensitivity to order, and drift across runs or contexts. The resulting
      numbers look cardinal while behaving like brittle rhetoric.
    </section>

    <section name="Primitive judgement">
      The system replaces that primitive with pairwise ratio elicitation. Given
      two items and an attribute, the model answers a more local question: which
      item has more of the attribute, and by roughly what factor on a bounded
      geometric ladder? This change matters because the comparison is concrete,
      contextual, and inherently relational. It is easier for a model to say
      "A is roughly 2.5 times clearer than B" than to say "A is 7.8 and B is
      6.1" in a way that preserves stable meaning across the entire dataset.
    </section>

    <section name="Why ratios, not just preferences">
      Merely asking which item is better would recover only an ordinal relation.
      That is sometimes enough to sort items, but it discards magnitude. A near
      tie and a decisive gap would both look like A > B. Ratio judgements retain
      intensity while remaining simple enough for an LLM to produce. The bounded
      ladder prevents pathological extremes and keeps the prompt interface
      regular enough for parsing, calibration, and downstream analysis.
    </section>

    <section name="Log-space composition">
      The mathematical reason ratios are attractive is that they linearize in
      log-space. If the latent strength of item i on an attribute is s_i, then a
      comparison between items i and j tries to measure log(s_i) - log(s_j). A
      ratio A/B = r implies log A - log B = log r. Once every comparison is
      translated into a log-difference, the data becomes a graph of noisy linear
      constraints over latent node potentials. This is the key structural move
      in the system: LLM outputs stop being free-floating judgements and become
      measurements inside an identifiable inference problem.
    </section>

    <section name="Observation model">
      Each pairwise response generates at least three pieces of information:
      direction, magnitude, and confidence. Direction and magnitude determine the
      signed log-ratio observation. Confidence determines how much weight that
      edge should receive in the solver. The confidence may come from self-report
      today and richer signals later, such as logprob-derived uncertainty or
      provider-side internal coherence metrics. Either way, the architecture
      treats confidence as a way of modulating observation variance rather than
      as a rhetorical aside.
    </section>

    <section name="Why robust aggregation is necessary">
      If all LLM comparisons were Gaussian noise around a true latent score
      difference, ordinary weighted least squares would be fine. But real model
      judgements contain a heavy-tailed error component. Some comparisons are
      simply bad. Perhaps the prompt was ambiguously interpreted. Perhaps the
      model latched onto an irrelevant surface feature. Perhaps the item text was
      unusually long or oddly structured. In these cases, a single edge can be
      dramatically inconsistent with the rest of the comparison graph. A
      non-robust estimator would allow those edges to pull the latent fit far
      away from the consensus implied by the broader graph.
    </section>

    <section name="IRLS with Huber loss">
      cardinal-harness uses Huber loss inside an iteratively reweighted least
      squares loop. Huber is quadratic near zero residual, so it preserves the
      efficiency of least squares when the data is locally well-behaved. It
      becomes linear in the tails, reducing the leverage of outliers without
      fully discarding them. IRLS is then the computational mechanism that turns
      this robust objective into a sequence of weighted least-squares problems,
      each of which can be solved efficiently with standard linear algebra.
      This is a pragmatic choice: statistically robust, computationally stable,
      and easy to diagnose.
    </section>

    <section name="Gauge and identifiability">
      The latent scores are only defined up to an additive constant in log-space.
      Shifting every score by the same amount leaves every pairwise difference
      unchanged. The solver therefore pins one degree of freedom per connected
      component. This is not an implementation hack; it is the normal gauge
      choice for pairwise comparison models. The meaningful outputs are the
      relative differences and induced rankings, not an absolute origin.
    </section>

    <section name="Uncertainty as first-class output">
      A ranking without uncertainty is inadequate for decision support. The
      system therefore estimates posterior variance, not just point estimates.
      Conceptually, this answers a question like: if we repeated this elicitation
      process with more or different noisy judgements, how much could each latent
      score still move? Items with many mutually reinforcing comparisons should
      have tight variance; isolated or contradictory items should have wide
      variance. This uncertainty supports both honest reporting and adaptive
      computation.
    </section>

    <section name="Planner logic">
      Once uncertainty exists, one can ask a better operational question than
      "what pair has not been compared yet?" The better question is "what next
      comparison would most improve the decision we care about?" cardinal-harness
      is not trying to saturate the comparison graph uniformly. It is trying to
      identify the relevant frontier, typically the top-K set, as efficiently as
      possible. The planner therefore blends graph-theoretic information gain
      with rank-risk terms focused on the selection boundary.
    </section>

    <section name="Why top-K focus matters">
      In many realistic uses, the tail of the ranking barely matters. If items
      ranked 42 and 43 are swapped, nothing operational changes. But if items 5
      and 6 are swapped when K = 5, the selected set changes. cardinal-harness
      encodes this asymmetry directly. Its stopping and prioritization logic care
      especially about frontier inversions: cases where an item just below the
      cut might in fact belong above it. This makes the system decision-aware
      rather than aesthetically obsessed with a perfect total order.
    </section>

    <section name="Stopping rule">
      A fixed comparison budget is a weak policy because the difficulty of the
      problem varies by dataset. Some sets have a clearly separated top tier and
      become easy quickly. Others contain many near-ties and remain ambiguous
      even after substantial budget. The stopping rule should therefore depend on
      posterior uncertainty, especially near the top-K frontier. cardinal-harness
      stops when boundary risk is sufficiently small or when a certified
      separation bound indicates that the current top-K is robust to plausible
      re-estimation.
    </section>

    <section name="Evaluation philosophy">
      Because the system is used for selection under uncertainty, no single
      metric captures all relevant failure modes. Kendall tau-b measures pairwise
      order correctness and is the most native global metric for a system built
      from pairwise data. Spearman rho measures rank displacement and catches
      large positional errors. Top-K precision and recall measure selection
      quality. Coverage checks whether uncertainty is honest. nDCG, CURL,
      weighted reversals, and Bayesian regret capture top-heavy decision quality.
      Frontier inversion probability is the most operational metric because it
      measures the exact kind of mistake the planner is designed to suppress.
    </section>

    <section name="System identity">
      Put together, cardinal-harness is not just a ranking library and not just
      a prompting trick. It is a measurement protocol for extracting latent
      comparative structure from LLMs. Its core commitments are: ask local ratio
      questions instead of unstable global ratings; aggregate with robust
      statistics instead of naïve averaging; preserve uncertainty instead of
      hiding it; spend budget adaptively instead of uniformly; and stop when the
      decision is good enough instead of when a preset loop counter expires.
    </section>
  </summary>

  <summary target_tokens="6000" actual_style="dense">
    <section name="1. What kind of system this is">
      cardinal-harness should be understood as an elicitation-and-inference
      engine rather than merely a prompt library or ranking utility. The repo
      exists to solve a specific epistemic problem: a language model often
      carries diffuse comparative knowledge about a set of items, but the most
      obvious way of asking for that knowledge, direct scalar rating, produces
      outputs that are superficially numeric and substantively unstable. The
      system therefore changes the contract between user and model. Instead of
      requesting direct absolute values, it requests local comparative
      measurements that can be aggregated into a coherent global latent scale.
      This is closer to measurement design than to ordinary LLM app building.
    </section>

    <section name="2. Why direct scoring fails">
      Direct scoring fails for several distinct reasons that are easy to
      conflate. One is calibration failure: a model does not maintain a stable,
      portable interpretation of what "7.4 out of 10" means across items, runs,
      and prompts. Another is anchoring: the first examples or recently seen
      items distort the meaning of later scores. Another is scale compression:
      models often overuse a narrow band of the scale, creating numbers that are
      formally distinct but weakly informative. Another is context dependence:
      the same item can be rated differently depending on comparison class,
      prompt framing, order, or local salience cues. These are not minor
      nuisances. They break the semantics needed for downstream quantitative use.
      If a score is not comparably meaningful across items, it cannot support
      robust selection, aggregation, or optimization.
    </section>

    <section name="3. Why pairwise comparison is the right primitive">
      Pairwise comparison is preferable because it localizes judgement. The model
      need not recover and maintain a global scale explicitly. It only needs to
      compare two concrete objects under a named attribute. This changes the
      cognitive demand of the task. The model can rely on direct discrimination
      rather than on an unstable act of absolute placement. Humans often exhibit
      the same asymmetry: they are usually better at saying which of two objects
      is heavier than at stating each weight in kilograms. The system exploits an
      analogous asymmetry in LLM behaviour.

      But pairwise comparison alone would still leave value on the table if it
      were only ordinal. A win/loss answer tells us who is above whom, but not
      how much above. That forces the inference system to reconstruct all
      magnitudes from many weak signs. Ratio elicitation preserves more of the
      model's internal comparative sense in a form that remains simple enough to
      request, parse, and aggregate.
    </section>

    <section name="4. Why ratios are mathematically convenient">
      Ratios are not only rhetorically natural; they also produce a particularly
      good algebraic object. Suppose each item i has an unknown positive latent
      attribute level x_i. A comparison saying that item i exceeds item j by a
      factor r is the statement x_i / x_j ≈ r. Taking logs converts this to
      log x_i - log x_j ≈ log r. Now every pairwise judgement is an approximate
      linear constraint on latent node potentials. The comparison dataset becomes
      a graph with signed weighted edges. This transformation is central because
      it allows the system to use mature tools from regression, graph inference,
      and uncertainty propagation instead of treating every judgement as a free
      text artefact.
    </section>

    <section name="5. The bounded geometric ladder">
      The ratio ladder is bounded and approximately geometric. This is a product
      and statistics choice at once. If the ladder were too coarse, the model
      could not express meaningful near-ties or moderate gaps. If it were too
      fine, the prompt would imply a false degree of precision and encourage
      unstable micro-distinctions. A geometric spacing is sensible because equal
      multiplicative changes map to equal additive changes in log-space, which
      matches the structure of the solver. The upper cap prevents the system from
      pretending to resolve distinctions whose exact multiplicative magnitude is
      not useful once one item is already clearly superior.
    </section>

    <section name="6. The meaning of a comparison observation">
      A single model response is not directly a truth claim. It is an
      observation with provenance and uncertainty. In the most basic version, the
      observation contains a winner, a ladder ratio, and a confidence value.
      These are converted into a signed log-ratio plus an observation weight.
      More sophisticated versions can derive confidence from token logprobs or
      future provider-side coherence signals. The important principle is that the
      system does not confuse "the model said it" with "the system should trust
      it maximally." Instead, every observation enters a weighted aggregation
      process where confidence modulates variance and residual structure matters.
    </section>

    <section name="7. Why the graph view matters">
      Seeing the dataset as a graph clarifies both identifiability and strategy.
      If comparisons connect all items richly, latent scores become better
      determined. If the graph is sparse or fractured, uncertainty remains high.
      If one region of the graph is well connected and another is poorly linked,
      the planner should spend budget asymmetrically. The graph perspective also
      explains why transitivity matters. A good system should let evidence from
      A/B and B/C inform A/C even if A and C were never directly compared. The
      point of global inference is precisely to integrate overlapping local
      judgements into a coherent latent field.
    </section>

    <section name="8. Why naïve least squares is insufficient">
      Once the problem is cast as fitting latent scores to noisy edge
      differences, ordinary least squares looks like the obvious baseline. It is
      also insufficient. LLM judgement noise is not purely light-tailed. Some
      observations are dramatically inconsistent with the broader pattern. These
      can arise from prompt misreadings, entity-text pathologies, local biases,
      accidental overconfidence, or parser-valid but substantively poor answers.
      In a non-robust estimator, a few such edges can exert disproportionate
      leverage, especially when they touch items with few other comparisons.
      Robustness is therefore not polish. It is necessary for the system to have
      sane failure behaviour under realistic LLM noise.
    </section>

    <section name="9. Why Huber specifically">
      Huber loss is a good compromise between efficiency and robustness. Near the
      origin it behaves like squared error, which is statistically efficient when
      residuals are modest and approximately Gaussian. In the tails it behaves
      linearly, reducing the incentive for the fit to contort itself around large
      outliers. More aggressive redescending losses can sometimes reject bad
      points more strongly, but they also introduce optimization quirks and can
      discard genuinely informative but surprising comparisons. Huber expresses a
      conservative philosophy: distrust large residuals more, but do not pretend
      they carry zero information.
    </section>

    <section name="10. Why IRLS is the computational mechanism">
      Iteratively reweighted least squares is the practical engine that makes the
      Huber objective tractable. Each iteration computes residual-sensitive
      weights, solves a weighted least-squares problem, and repeats until the
      solution stabilizes. This provides a transparent computational story. The
      optimization is not mysterious; it is a sequence of familiar linear solves.
      That matters for engineering because diagnostics, convergence checks,
      numerical conditioning, and uncertainty approximations can all be built on
      top of this structure. The repo's emphasis on explicit diagnostics and
      planner semantics fits naturally with IRLS.
    </section>

    <section name="11. Confidence mapping and observation variance">
      Confidence is not directly used as "trust percentage." Instead it is mapped
      into an effective variance schedule. This is conceptually important. The
      solver wants to know how noisy each edge measurement should be treated as,
      not whether the model sounded assertive. By routing confidence through a
      mapping function, the system can calibrate how strongly differences in
      confidence should affect fit weight. This also creates a seam for future
      replacement. Self-reported confidence can be swapped or blended with
      logprob-derived or provider-side measures without changing the rest of the
      inference architecture.
    </section>

    <section name="12. Gauge pinning and the meaning of scores">
      Because only differences are observed, the latent score vector is defined
      only up to an additive constant in log-space. This is standard and benign.
      Pinning one node per connected component does not distort the inference; it
      simply chooses a coordinate system. Users sometimes misread latent scores as
      absolute truths. They are not. Their meaning lies in relative gaps,
      ordering, and uncertainty. The system is honest about this by treating the
      gauge as a structural fact rather than hiding it behind arbitrary scaling.
    </section>

    <section name="13. Uncertainty is not optional metadata">
      In many ranking systems, uncertainty is absent or bolted on. Here it is
      central. The product promise is not "we always know the right answer," but
      "we can spend budget until the answer is reliable enough for the decision."
      That promise requires a posterior uncertainty model. Without it, there is
      no principled way to compare the value of another model call against its
      cost, no principled stopping rule, and no honest report of where the result
      is fragile. Uncertainty is therefore part of the object being produced, not
      a decorative confidence badge.
    </section>

    <section name="14. Small-problem exactness and large-problem pragmatism">
      The repo distinguishes between regimes. For smaller problems, exact
      posterior variance computations are feasible and desirable. For larger
      problems, approximate methods such as Hutchinson-style diagonal estimation
      become more appropriate. This is not a conceptual compromise so much as an
      engineering acknowledgement that the product has to scale. The same high-
      level contract is maintained: provide a reasonable posterior uncertainty
      estimate that is good enough to drive planning and stopping. The exact
      numerical route depends on problem size.
    </section>

    <section name="15. Why planning must be adaptive">
      A fully dense comparison design is usually wasteful. The cost of querying
      all pairs grows quadratically, while the operational benefit often
      saturates much earlier. If the task is to find the best few items, many
      comparisons in the tail have negligible decision value. An adaptive planner
      makes the system economically serious. It asks not "what comparisons remain
      unasked?" but "what comparison would reduce the uncertainty that matters
      most?" This is the point where the repo stops being a passive aggregator
      and becomes an active elicitation policy.
    </section>

    <section name="16. Effective resistance as information-gain intuition">
      Effective resistance provides a graph-theoretic proxy for how much a new
      comparison would improve the connectedness and certainty of the latent fit.
      Intuitively, comparisons between poorly linked regions of the graph are
      more informative than comparisons inside a region already saturated with
      redundant evidence. This makes effective resistance a natural ingredient in
      the planner objective. It encourages queries that reduce structural
      uncertainty, not merely queries that look superficially novel.
    </section>

    <section name="17. Why top-K risk, not just global fit">
      Information gain alone would encourage broad exploration, which is useful
      but not sufficient. The product is usually used for selection under budget,
      so the planner must care more about uncertainty at the decision boundary
      than about uncertainty deep in the tail. This is why rank-risk terms are
      added, especially frontier inversion probability. The system asks a sharp
      operational question: how likely is it that something just below the cut
      should actually be above it? That question aligns the planner with the real
      economic stakes of ranking.
    </section>

    <section name="18. Stopping as a decision rule">
      A fixed number of comparisons is easy to implement and philosophically
      weak. It spends too much on easy problems and too little on hard ones. A
      posterior stopping rule is better because it ties compute expenditure to
      the uncertainty that remains. If the frontier is already well separated,
      further calls are mostly waste. If the frontier is entangled in near-ties,
      stopping early is reckless. cardinal-harness therefore stops when boundary
      risk is low enough or when a certified separation criterion says the top-K
      is robust. This is exactly the kind of logic a serious measurement system
      should have.
    </section>

    <section name="19. Multi-attribute composition">
      Real decisions rarely depend on a single attribute. The repo therefore
      allows multiple independently elicited attribute spaces whose latent scores
      are normalized, weighted, and gated into a combined utility. This is an
      important extension because it preserves the local clarity of pairwise
      elicitation while supporting richer downstream decision semantics. Each
      attribute gets its own measurement problem; the final ranking is a
      composition layer on top. This separation is healthy. It avoids collapsing
      all reasoning into a single overloaded prompt and keeps the uncertainty and
      score structure legible.
    </section>

    <section name="20. Gating and feasibility">
      Not every ranking problem is "maximize one utility scalar over all items."
      Sometimes certain minimum standards matter. Gates allow the system to
      encode feasibility constraints such as safety or competence thresholds.
      This changes the meaning of evaluation as well: some metrics should be
      reported on all items, and some on the feasible set. The distinction is
      not cosmetic. A system can look good globally while making poor mistakes on
      the set actually eligible for selection, or vice versa.
    </section>

    <section name="21. Why evaluation needs many metrics">
      Ranking quality is not a one-dimensional concept. If one insists on a
      single number, one hides tradeoffs rather than resolving them. The repo's
      metric suite reflects this. Kendall tau-b is the most native global metric
      because it measures pairwise order agreement, which is closest to the data
      primitive the system elicits. Spearman rho measures rank displacement and
      becomes informative when a few items move a long distance. Top-K precision
      and recall evaluate set recovery. Coverage measures uncertainty honesty.
      nDCG, CURL, weighted rank reversals, and Bayesian regret emphasize top-
      heavy consequences. Frontier inversion probability is the most directly
      operational because it tracks the very event the planner tries to avoid.
    </section>

    <section name="22. Why Kendall is usually the default headline">
      If forced to choose one global metric, Kendall tau-b is usually the right
      headline for this repo because cardinal-harness is built from pairwise
      judgements. Kendall asks whether pairwise relations are correct. That maps
      naturally onto the system's epistemic primitive. Spearman is still useful,
      but it is a complementary perspective rather than the core one. This is a
      good example of the broader design philosophy: choose evaluation objects
      that match the structure of the inference problem rather than importing
      familiar metrics uncritically.
    </section>

    <section name="23. Prompting is important but not the whole story">
      The repo includes substantial prompt work, including ladder design and
      prompt layout experiments. But the prompting layer should be seen as the
      front door, not the whole house. The distinctive value of cardinal-harness
      comes from combining prompt design with robust aggregation, explicit
      uncertainty, adaptive planning, cost accounting, and caching. Many systems
      stop at "we found a prompt that seems to work." This repo assumes prompts
      are only one component of a full measurement pipeline.
    </section>

    <section name="24. Cost matters structurally">
      The system is not built for an imaginary world of free model calls. The
      gateway, pricing, usage tracking, and caching layers exist because the
      economics of elicitation matter. An algorithm that slightly improves
      ranking quality but doubles token cost may be inferior in real operation.
      Conversely, a planner that reaches the same decision confidence with far
      fewer comparisons is genuinely better. Cost semantics are therefore part of
      the product, not incidental logging. This is why the repo treats cache and
      gateway infrastructure as core rather than peripheral.
    </section>

    <section name="25. Caching and reproducibility">
      Pairwise judgements are expensive and often reusable. Caching is therefore
      not just an optimization but a practical expression of the system's data
      model. If the same model, prompt, attribute, and entities are queried
      again, the prior judgement should normally be reused. This stabilizes
      experiments, lowers cost, and turns the system into an accumulating
      knowledge base rather than a purely transient workflow. It also supports
      reproducibility by letting later runs replay prior judgement structure
      instead of silently drifting due to fresh stochastic model calls.
    </section>

    <section name="26. Relation to ANP and higher-level workflows">
      The pairwise-ratio engine is the base layer for higher-level workflows such
      as pipeline, flywheel, and commander. Those layers may introduce richer
      orchestration, decomposition, or strategic evaluation patterns, but they
      depend on the same core commitment: local pairwise judgements should be
      aggregated into cardinal latent structures with explicit uncertainty. The
      foundational document therefore belongs below those workflow abstractions.
      It explains the epistemic engine they all rely on.
    </section>

    <section name="27. Why this is a foundations document and not just a tutorial">
      A tutorial explains how to use a tool. A foundations document justifies why
      the tool has the shape it does. This repo needs both, but the latter is
      especially important because many of its choices are easy to mistake for
      arbitrary preferences: ratio ladders, log-space inference, Huber loss,
      adaptive stopping, top-K-centric planning, multiple evaluation metrics. In
      reality these choices cohere around a single thesis: if you want LLM
      judgements to behave like measurements rather than vibes, you must design
      the questions, solver, uncertainty model, planner, and stopping logic as a
      unified system.
    </section>

    <section name="28. The deepest compact statement">
      The whole system can be compressed to a single idea: ask the model for the
      kind of judgement it can make relatively well, transform that judgement
      into a mathematically composable observation, aggregate it with robust
      statistics, keep uncertainty explicit, and spend additional budget only
      where the live decision remains ambiguous. Everything else in the repo is
      an elaboration or operationalization of that idea.
    </section>

    <section name="29. A more explicit latent-variable picture">
      One useful way to think about the system is as a noisy measurement model
      over hidden utilities. Let theta_i denote the latent amount of some
      attribute possessed by item i after taking logs or otherwise choosing a
      convenient coordinate system. A comparison between i and j produces an
      observation y_ij that attempts to estimate theta_i - theta_j. In practice
      y_ij is not continuous but drawn from a ladder of admissible values, then
      mapped into log-space. There is also an observation weight or variance
      parameter v_ij determined by confidence and later modified by robust
      reweighting. The solver is then fitting theta so that weighted residuals
      y_ij - (theta_i - theta_j) are collectively small without allowing a few
      large residuals to dominate the result.

      This picture matters because it makes clear what kind of claim the system
      is and is not making. It is not claiming that the model has direct access
      to a true Platonic scalar for each item. It is claiming that there exists
      a useful latent comparative structure that can be more stably elicited
      through local relative judgements than through direct absolute prompts.
      That is a much more defensible claim, and it is empirically the right one
      for the use cases the repo targets.
    </section>

    <section name="30. Why prompt semantics and solver semantics must align">
      Many ranking systems fail because the prompt contract and the aggregation
      logic are conceptually mismatched. For example, a prompt may ask for an
      imprecise textual judgement, while the downstream code behaves as though it
      had received a calibrated cardinal measurement. cardinal-harness tries to
      avoid this by making the prompt outputs look as much as possible like the
      objects the solver actually wants: winner, rough magnitude, and confidence
      on a fixed ladder. The prompt semantics are intentionally narrow so that
      the solver is not forced to invent structure ex post. This is one reason
      the ratio ladder and strict JSON output matter. They are not merely for
      parser convenience; they are how the front end and the statistical back
      end are kept in conceptual register.
    </section>

    <section name="31. Why repeated pairwise measurement beats one-shot ranking">
      One could imagine asking a powerful model to read the whole set of items
      and simply output a final ranking with explanations. Sometimes that will
      look plausible. The problem is that it hides all the error structure. There
      is no clean way to know which local comparisons the model feels certain
      about, which are near-ties, where the output is brittle, or where more
      information would have helped. A one-shot ranking collapses judgement,
      aggregation, uncertainty, and stopping into a single opaque act. By
      contrast, repeated pairwise elicitation exposes the measurement process.
      It yields a graph of evidence that can be audited, robustly aggregated,
      partially reused, and queried further. This is a major reason the repo can
      support diagnostics and principled stopping in a way that one-shot ranking
      systems generally cannot.
    </section>

    <section name="32. The role of parseability and constrained outputs">
      Strict structured output is foundational here. If a pairwise response can
      drift arbitrarily in form, then every downstream metric mixes judgement
      quality with parser failure, extraction heuristics, and ad hoc repair
      logic. A constrained ratio ladder and minimal JSON schema sharply reduce
      that surface area. This does not eliminate semantic noise, but it removes a
      large class of avoidable interface noise. In a system whose whole value
      lies in converting model outputs into clean statistical objects, interface
      discipline is part of the epistemology, not just part of the software
      hygiene.
    </section>

    <section name="33. What confidence can and cannot mean">
      Confidence deserves caution. Self-reported confidence from an LLM is not a
      pure estimate of epistemic uncertainty. It can reflect style, training
      priors, reward-model habits, or local prompt cues. Yet dismissing it
      entirely would also be wrong, because it often contains genuine signal.
      The architecture therefore treats confidence as a noisy input to a mapping
      rather than as a literal truth. This is an important middle position. The
      system neither naively trusts confidence nor discards it. It creates a seam
      where confidence can be calibrated, blended, or replaced. That is exactly
      what a robust measurement pipeline should do with an informative but
      imperfect signal.
    </section>

    <section name="34. Failure modes the system is explicitly designed for">
      Several concrete failure modes are built into the design assumptions. One
      is local outlier judgement: a comparison that is wildly inconsistent with
      the surrounding graph. Another is sparse-data uncertainty: an item whose
      score is not well determined because it has too few informative edges.
      Another is frontier ambiguity: multiple near-tied items straddling the
      selection boundary. Another is cost inefficiency: spending many expensive
      calls reducing uncertainty in parts of the ranking that do not matter for
      the actual decision. Another is evaluation confusion: celebrating a global
      metric even though the top-K set is still unstable. The repo's solver,
      planner, stopping rule, and metric suite can be read as direct responses
      to these specific classes of failure.
    </section>

    <section name="35. What the system does not promise">
      The system does not promise access to objective truth in every domain. It
      does not claim that every attribute is intrinsically ratio-scaled in the
      philosophical sense. It does not claim that every model family will share a
      single coherent latent representation. It does not erase prompt sensitivity
      or model bias. What it does promise is narrower and more useful: given a
      chosen attribute and a chosen judge model or ensemble, it provides a much
      more principled, auditable, and uncertainty-aware way of extracting stable
      comparative structure than naïve direct scoring. That is enough to be very
      valuable, and it keeps the repo's claims appropriately disciplined.
    </section>

    <section name="36. Why cost tracking belongs in the foundations">
      It may seem odd to include gateway pricing and token accounting in a
      foundations document, but doing so is correct. In this repo, cost is not
      an afterthought because the planner is fundamentally choosing whether an
      additional unit of information is worth buying. A measurement system that
      ignores the cost of measurement cannot actually optimize a measurement
      policy. This is why pricing, usage, and caching appear alongside solver and
      planner logic in the broader codebase map. The system's real objective is
      not maximal information at any cost; it is sufficient decision confidence
      at acceptable cost.
    </section>

    <section name="37. Why caching is conceptually elegant, not just cheap">
      Caching pairwise judgements is often described as an engineering
      optimization, but it also expresses a deeper view of what the data is. If a
      judgement is defined by the tuple of model, prompt template, attribute, and
      entity texts, then repeating that exact query should normally reproduce the
      same semantic observation for system purposes. Treating it as cacheable is
      a way of asserting that the elicitation protocol has identity. This turns
      the repo into a system that accumulates comparative evidence over time
      rather than one that repeatedly burns tokens to rediscover what it already
      knows under the same conditions.
    </section>

    <section name="38. How to read the planner philosophically">
      The planner is often described operationally, but it also embodies a view
      about rational experimentation. It says that measurement should be focused
      where posterior uncertainty and decision stakes intersect. This is closely
      related to classical value-of-information ideas. An unanswered comparison
      is not important merely because it is unanswered. It is important only if
      the answer would plausibly change the choice, the confidence in the choice,
      or the allocation of later measurement budget. This helps distinguish the
      repo from systems that pursue completeness for its own sake.
    </section>

    <section name="39. Why top-K selection changes almost everything">
      Once the product goal is stated as top-K identification rather than full
      ranking reconstruction, several design choices become easier to justify.
      The planner should care most about the frontier. The stopping rule should
      depend on boundary uncertainty. The evaluation suite should include metrics
      that are top-heavy or selection-sensitive. The system should be willing to
      leave some parts of the tail relatively underexplored if the decision
      object is already robust. This is one of the most important conceptual
      compressions in the repo. A reader who misses it may wrongly interpret the
      planner as a compromised ranking algorithm, when in fact it is a faithful
      decision-oriented measurement strategy.
    </section>

    <section name="40. Why the metric suite is a map of decision semantics">
      The collection of evaluation metrics is best understood as a map from
      mathematical quantities to kinds of mistakes. Kendall tau-b corresponds to
      pairwise order mistakes. Spearman corresponds to rank displacement. Top-K
      precision and recall correspond to set-selection mistakes. Coverage
      corresponds to uncertainty dishonesty. nDCG and CURL correspond to top-
      weighted ranking mistakes. Weighted reversals correspond to interpretable
      local disorder. Bayesian regret corresponds to downstream utility loss.
      Frontier inversion probability corresponds to the planner's central risk.
      Thinking of the metric suite this way prevents a common error: treating
      evaluation as a competition to maximize whichever scalar is easiest to
      report rather than as a disciplined attempt to characterize the failure
      surface of the system.
    </section>

    <section name="41. Why this repo cares about explanation depth">
      The request for nested dense summaries is itself aligned with the repo's
      philosophy. The system has a compact core idea, but that idea unfolds into
      multiple interacting layers: prompt design, measurement semantics,
      log-space inference, robustness, uncertainty, adaptive querying, stopping,
      cost control, and evaluation. A shallow explanation makes the system sound
      like "ask pairwise questions and average them." An overlong but unstructured
      explanation loses the causal skeleton. A matryoshka document is therefore a
      good fit: it lets readers stop at the level of depth they need while
      preserving one consistent thesis through every expansion layer.
    </section>

    <section name="42. Final expansion of the core claim">
      cardinal-harness is built on the belief that if you want usable numbers
      from LLM judgement, you must respect the difference between expression and
      measurement. Models can express many things loosely, but only some kinds of
      output can be stably treated as observations inside a coherent inferential
      system. Pairwise ratio judgements are such an output. They are local enough
      to be elicited relatively reliably, structured enough to be parsed,
      cardinal enough to preserve magnitude, and algebraically well-behaved after
      a log transform. Once turned into graph edges, they can be aggregated with
      robust statistics, equipped with uncertainty, and used to drive adaptive
      decision-focused measurement. That is why the repo looks the way it does.
      The prompts, solver, planner, cache, gateway, metrics, and stopping rule
      are not separate clever tricks. They are the coordinated machinery needed
      to make LLM comparative judgement behave like an engineered measurement
      process rather than an improvised conversation.
    </section>
  </summary>
</matryoshka_document>
