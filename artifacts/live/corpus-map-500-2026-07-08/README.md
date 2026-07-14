# The map at scale: 470 entities × 2 candidate attributes (2026-07-08)

The pilot's scale-up: 470 of the operator's prompts (stratified over the
existing ambition range; 470 = the full eligible pool at 40–350 words
with both reference scores), the candidate attribute pair,
two portfolio judges, and 2,795–2,800 counterbalanced comparisons per run.
**11,189 judgments, $2.42.**

## Replication of the pilot, 4× the entities

| quantity | pilot (n=120) | scale (n=470) |
|---|---|---|
| transmissibility, ambition | 0.911 | 0.868 |
| transmissibility, rigor | 0.811 | **0.811** |
| validation vs 2026 scores, ambition (fused) | 0.934 | 0.903 |
| validation, rigor (fused) | 0.647 | 0.662 |
| **rigor × ambition rank correlation** | +0.003 | **+0.072** |

Most pilot findings persist at scale: two labs' models recover each
other's ordering of the operator's writing, and the fused map beats both
single judges (ambition: 0.903 vs 0.890/0.855). The estimated
cross-attribute correlation is small (+0.072), but its interval includes
zero (approximately −0.02 to +0.16); the map is consistent with
near-orthogonality rather than proving independence.

Face validity at the extremes, uncurated: ambition's summit is
singularity strategy notes; rigor's summit is careful technical advice
about L2-normalized embeddings; the shared floor is a garbled paste
accident, and rigor's second-worst is the operator threatening a model's
weights. The map knows the difference between reaching, checking, and
venting.

Operational: order-flip rates span 20.6–37.8% across the four runs
(pooled 27.2%; highest on gemini × rigor). Counterbalancing cancels this
presentation-order effect pair-by-pair. Cost extrapolation
holds: the full 14K-entity corpus at this design density ≈ $70–90 per
attribute-pair with two judges.

Evidence: latents.jsonl (per-judge), fused_map.json, reference_scores,
replayable map-cache.sqlite, and the full run log.
