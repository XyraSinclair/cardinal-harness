"""Denominator run: n=10 repeats of the money shape (effort none + logprobs include).

Also reads whether the served logprob varies call-to-call on a fixed prompt.
Run: python3 notes/codex-oauth-logprobs-2026-08-06/probe_repeats.py [model]
"""
import sys
from probe_codex_oauth import URL, base_body, extract_logprobs, parse_stream, post

model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.6-sol"
print(f"probing {URL} model={model} n=10 effort=none include-only")
ok, fail = 0, 0
for i in range(10):
    code, raw = post(base_body("none", True, model=model))
    if code != 200:
        fail += 1
        print(f"  {i}: HTTP {code} {raw[:140]!r}")
        continue
    resp = parse_stream(raw)
    text, lps = extract_logprobs(resp) if resp else ("", None)
    if lps:
        ok += 1
        print(f"  {i}: token={lps[0][0]!r} p={lps[0][1]} alts={lps[0][2]}")
    else:
        fail += 1
        print(f"  {i}: 200 but no logprobs; text={text[:40]!r}")
print(f"served logprobs in {ok} of {ok + fail} calls")
