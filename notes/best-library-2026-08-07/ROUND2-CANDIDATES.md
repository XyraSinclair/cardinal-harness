# Round 2 candidates: the discriminative regime (sketch, 2026-08-07)

Round 1 lesson: benchmarks on memorized facts cannot discriminate sorting
methods. Round 2 must sit where pointwise judgment is unreliable but truth
is still checkable. Candidate domains, with the selection criterion each
must pass BEFORE registration: measured baseline-M1 imperfection (run the
pointwise cell first; if Spearman > 0.95, the domain is ceiling — discard).

1. **Human-scored essays (ASAP or similar public set).** Truth: human
   rubric scores. Pointwise LLM scoring is documented-noisy; comparative
   judgment literature says pairwise is the strong instrument here. Risk:
   truth is itself a noisy panel; report attenuation-corrected bounds.
2. **Composite computable attributes over synthetic documents.** E.g.,
   short invoices/ledgers, attribute = net total (requires multi-step
   extraction+arithmetic per item). Truth: exact by construction. Pointwise
   must compute; pairwise can compare structure. Fully controllable
   difficulty dial.
3. **Non-memorized physical estimates.** E.g., mass/volume of described
   composite objects ("a wheelbarrow full of wet sand"). Truth: computed
   from densities. Models cannot recall these; they must reason. Tests
   whether ratio judgments aggregate estimation error better than direct
   estimates (the Vul/Galton wisdom-of-crowds-within-a-model question).
4. **Lists larger than context (scaling regime).** 500+ items where
   listwise M3 is structurally impossible and pointwise scores collapse
   into ties. This is cardinal's native scaling claim; truth from any of
   the above domains.

Also carried from round 1, independent of domain choice:

- Span compression fix candidate: log-anchored ratio prompts or ladder
  extension; re-measure slope on the round-1 corpus (frozen fixture,
  $0.05/run) before and after. Candidate for a public issue once a fix
  tranche is scoped.
- Position-bias spike on short famous entities (1.32 nats/pair): check
  whether entity-length or fame drives it; cheap ablation on the frozen
  corpus.
