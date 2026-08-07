"""Cross-model two-phase: reason on one model, read logprobs on another (codex oauth).

Cells (n=5 each, uncertain pair so the PMF is not saturated):
  A. baseline: reader at effort none, no analysis context
  B. same-model: gpt-5.6-sol medium analysis -> gpt-5.6-sol none + logprobs
  C. cross-model: gpt-5.6-sol medium analysis -> gpt-5.4-mini none + logprobs
  D. baseline mini: gpt-5.4-mini none, no analysis

Run: python3 notes/codex-oauth-logprobs-2026-08-06/probe_crossmodel.py
"""
import statistics
from probe_codex_oauth import base_body, extract_logprobs, parse_stream, post, reasoning_tokens, URL

SYS = ("You are an expert quantitative comparator. Compare entity A and entity B "
       "by the given attribute and answer with exactly one letter: A or B. "
       "The answer is the entity with MORE of the attribute.")
Q = ("Attribute: mass.\n<entity_A>one liter of solid ice</entity_A>\n"
     "<entity_B>one liter of liquid water</entity_B>\n"
     "Answer with exactly one letter: A or B.")
ANALYSIS_Q = Q.replace("Answer with exactly one letter: A or B.",
                       "Analyze this comparison carefully. Do NOT give a verdict.")

def one_analysis(model="gpt-5.6-sol", effort="medium"):
    body = base_body(effort, False, model=model, user=ANALYSIS_Q)
    # CRITICAL: phase 1 must NOT inherit the one-letter answer mandate, or the
    # "analysis" degenerates to the verdict letter (caught live 2026-08-07:
    # analysis_chars=1) and phase 2 measures verdict-copying.
    body["instructions"] = ("You are a careful physical reasoner. Write a short, "
                            "factual analysis of the comparison. Do not state a "
                            "final verdict letter.")
    r = post(body)
    if r[0] != 200:
        raise SystemExit(f"phase1 HTTP {r[0]} {r[1][:200]}")
    resp = parse_stream(r[1])
    text, _ = extract_logprobs(resp)
    assert len(text) > 80, f"analysis suspiciously short: {text!r}"
    return text, reasoning_tokens(resp)

def read_cell(tag, model, ctx=None, n=5):
    toks, masses = [], []
    for _ in range(n):
        body = base_body("none", True, model=model, user=Q, extra_input=ctx)
        code, raw = post(body)
        if code != 200:
            print(f"  {tag}: HTTP {code} {raw[:120]!r}")
            continue
        text, lps = extract_logprobs(parse_stream(raw))
        if lps:
            toks.append(lps[0][0])
            masses.append(lps[0][1])
    served = len(masses)
    mean = statistics.mean(masses) if masses else float("nan")
    sd = statistics.stdev(masses) if len(masses) > 1 else 0.0
    print(f"  {tag}: served {served}/{n} · tokens={toks} · mass mean={mean:.4f} sd={sd:.4f}")
    return toks, masses

print(f"probing {URL} — pair: 1L ice (A) vs 1L water (B) by mass; correct=B")
read_cell("A sol-none baseline ", "gpt-5.6-sol")
read_cell("D mini-none baseline", "gpt-5.4-mini")
analysis, rt = one_analysis()
print(f"  phase1 sol@medium: reasoning_tok={rt} analysis_chars={len(analysis)}")
CTX = [{"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": ANALYSIS_Q}]},
       {"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": analysis}]}]
read_cell("B sol<-sol analysis ", "gpt-5.6-sol", ctx=CTX)
read_cell("C mini<-sol analysis", "gpt-5.4-mini", ctx=CTX)
