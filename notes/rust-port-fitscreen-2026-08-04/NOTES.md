# Rust encoder-port fit screen — 2026-08-04

Session note: viability of hand-ported Rust inference (Franken-port style:
fused kernels + conformance harness against the HF reference) as a long-term
direction for the OpenPriors stack. One measurement, one taxonomy, one
source-checked correction. Research-grade; no code changed.

## Measured result (the surprising one)

bge-small-en-v1.5 (33M), Apple M5 Max, batch 32, padded seq 60, N=512,
best-of-5, default threading both sides, tokenization outside the timed loop.
Script and raw log in this directory (`bench_embedder.py`, `run.log`).

| path                  | sent/s | vs eager | parity vs eager (cos min) |
|-----------------------|-------:|---------:|--------------------------:|
| torch eager fp32      |    232 |    1.00x | —                         |
| ONNX Runtime fp32     |    168 |    0.73x | 1.000000                  |
| ORT int8-dynamic      |    167 |    0.72x | 0.966711                  |

Prior falsified: "ONNX Runtime captures half the hand-port win for free."
On Apple Silicon it captures none of it — torch eager routes GEMMs through
Accelerate/AMX; ORT's CPU EP does not. ORT dynamic int8 adds nothing over
ORT fp32 and costs real embedding fidelity (cos min 0.967 is a quality
regression, not noise). The incumbent to beat on M-series is
torch-eager-on-AMX, not ORT.

Headroom: 232 sent/s ~= 0.9 effective TFLOP/s for this shape — well under
M5 Max AMX+NEON capability; dispatch/Python overhead dominates at 33M params.
A fused Rust port (tokenizer→encoder→pool, int8 SDOT, exactness argument for
the pooled output) plausibly holds 2–4x over the real strong baseline.

Denominators / caveats (do not overquote this table):
- one machine, one model, one shape; no torch.compile, no MPS, no Core ML
  rungs — those are the next ladder rungs before any repo is born;
- `quantize_dynamic` ran without ORT preprocessing (warning in log): the
  int8 row is a floor for ORT-int8, not a ceiling;
- ORT reps had a 3.9s outlier (thermal/background); best-of-5 used.

## Model-class taxonomy (what we would port, and why)

Embedders, cross-encoder rerankers, and classifiers are one machine — a
BERT-family encoder with different heads (~90% shared kernel work). Small
generative LLMs are ruled out: llama.cpp owns that niche; margin thin.

- **Bi-encoder embedder** (bge/gte class): blocking over O(n^2) candidate
  pairs, dedup, cluster structure for the planner. Easiest port; bring-up
  target for the conformance harness and kernel base.
- **Cross-encoder reranker** (jina-turbo ~38M class): cheap local judge —
  IRLS warm start, pair pre-screening, informativeness estimates. A prior,
  not a replacement (scores relevance, not attribute ratios).
- **Classifiers/NLI**: near-free once the encoder exists; DeBERTa's
  disentangled attention is a different architecture — only pay for a
  concrete need.
- **Distilled cardinal ratio judge** (the strategic one; custom weights):
  distill the LLM judge into a small cross-encoder predicting ladder
  outcomes. Flywheel: local judge scores all pairs free → planner routes
  LLM budget to uncertain/influential pairs → re-distill. The Rust port is
  what makes the free leg real (in-process million-pair sweeps).

Deployment shape: a crate behind harness traits, not a standalone binary —
fusion boundary extends past the model into the pipeline (tokenize→encode→
score→IRLS accumulate) with no serialization seam.

## Source-checked correction

`src/packet.rs`: `PacketObservation` stores per-pair `log_ratio` +
`precision` (Gaussian summary), NOT the full ladder logprob distribution.
So packets support distillation to a Gaussian target today (precision as
sample weight, zero new capture). Full-softmax distributional distillation
needs raw ladder logprobs captured upstream in the trace/gateway layer — a
capture-point decision, only worth taking if the distillation thread starts.

## Sequencing (defended order)

1. Seam first: `Embedder`/`LocalJudge` traits in cardinal-harness, candle
   impl (days). Proves pipeline value — does blocking + warm-start cut LLM
   spend and comparisons-to-convergence? Sizes the port's worth.
2. Distill the ratio judge on existing packet data (Gaussian target).
3. Hand-port the encoder behind the same trait only when local sweep
   throughput is the measured bottleneck. New repo boundary: encoder engine
   is its own repo; traits/candle/distillation live here with the consumer.

Guard: the local judge's noise scale must be measured against held-out LLM
judgments per attribute/domain before IRLS fuses it — otherwise we launder
a cheap model's biases through the uncertainty math.

Status: unscheduled research direction. No operator-queue item; no issue.
Next concrete handle: run the fuller baseline ladder (torch.compile / MPS /
Core ML / preprocessed int8) before deciding the candle-seam tranche.
