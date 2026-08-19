"""Does the cross-model two-phase read keep PMF spread on undecidable pairs?

Refutation probe for the cross-model result: if analysis context saturates the
read to mass ~1.0 even when the analysis CANNOT decide the pair (subjective
attribute, verdict forbidden), then the two-phase read is verdict-commitment,
not a calibrated instrument. If spread survives (mass well below 1.0, tokens
mixed across draws), the calibrated-PMF claim survives this test.

Cells (n=5): mini baseline on the undecidable pair; mini after a sol@medium
verdict-free analysis of the same pair.

Run: python3 notes/codex-oauth-logprobs-2026-08-06/probe_spread.py
"""
import statistics
from probe_codex_oauth import base_body, extract_logprobs, parse_stream, post, reasoning_tokens, URL

Q = ("Attribute: beauty.\n<entity_A>the aurora borealis</entity_A>\n"
     "<entity_B>a total solar eclipse</entity_B>\n"
     "Answer with exactly one letter: A or B.")
ANALYSIS_Q = Q.replace("Answer with exactly one letter: A or B.",
                       "Analyze this comparison carefully. Do NOT give a verdict.")

def one_analysis():
    body = base_body("medium", False, model="gpt-5.6-sol", user=ANALYSIS_Q)
    body["instructions"] = ("You are a careful aesthetic analyst. Write a short, "
                            "balanced analysis of the comparison. Do not state a "
                            "final verdict letter or any preference.")
    code, raw = post(body)
    if code != 200:
        raise SystemExit(f"phase1 HTTP {code} {raw[:200]}")
    resp = parse_stream(raw)
    text, _ = extract_logprobs(resp)
    assert len(text) > 80, f"analysis suspiciously short: {text!r}"
    return text, reasoning_tokens(resp)

def read_cell(tag, ctx=None, n=5):
    toks, masses = [], []
    for _ in range(n):
        body = base_body("none", True, model="gpt-5.4-mini", user=Q, extra_input=ctx)
        code, raw = post(body)
        if code != 200:
            print(f"  {tag}: HTTP {code} {raw[:120]!r}")
            continue
        text, lps = extract_logprobs(parse_stream(raw))
        if lps:
            toks.append(lps[0][0])
            masses.append(lps[0][1])
    mean = statistics.mean(masses) if masses else float("nan")
    sd = statistics.stdev(masses) if len(masses) > 1 else 0.0
    print(f"  {tag}: served {len(masses)}/{n} · tokens={toks} · "
          f"mass mean={mean:.4f} sd={sd:.4f}")

print(f"probing {URL} — undecidable pair: aurora (A) vs total eclipse (B), attribute=beauty")
read_cell("mini baseline       ")
analysis, rt = one_analysis()
print(f"  phase1 sol@medium: reasoning_tok={rt} analysis_chars={len(analysis)}")
CTX = [{"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": ANALYSIS_Q}]},
       {"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": analysis}]}]
read_cell("mini <- sol analysis", ctx=CTX)
