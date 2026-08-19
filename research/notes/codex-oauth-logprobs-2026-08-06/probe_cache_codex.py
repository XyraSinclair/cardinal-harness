"""Does the codex-oauth backend serve prompt cache on a long stable prefix?

Shape mirrors docs/LOGPROBS.md 'Prompt cache and nonce perturbation': long
stable system+entity prefix, nonce at the very end, 6 calls. Reads
usage.input_tokens_details.cached_tokens from response.completed.

Run: python3 notes/codex-oauth-logprobs-2026-08-06/probe_cache_codex.py
"""
import json, secrets
from probe_codex_oauth import base_body, parse_stream, post, URL

PAD = ("Reference context (fixed): " +
       " ".join(f"item-{i:04d} has weight {i * 37 % 101}." for i in range(400)))
STABLE = (f"{PAD}\nAttribute: mass.\n<entity_A>a chicken egg</entity_A>\n"
          "<entity_B>a bowling ball</entity_B>\n"
          "Answer with exactly one letter: A or B.")

print(f"probing {URL} model=gpt-5.6-sol effort=none, 6 calls, nonce at tail")
import sys
use_key = "--key" in sys.argv
for i in range(6):
    user = f"{STABLE}\ndraw-token: {secrets.token_hex(8)}"
    body = base_body("none", True, user=user)
    if use_key:
        body["prompt_cache_key"] = "cardinal-probe-cache-2026-08-07"
    code, raw = post(body)
    if code != 200:
        print(f"  {i}: HTTP {code} {raw[:120]!r}")
        continue
    resp = parse_stream(raw)
    usage = resp.get("usage") or {}
    det = usage.get("input_tokens_details") or {}
    print(f"  {i}: input={usage.get('input_tokens')} cached={det.get('cached_tokens')} "
          f"output={usage.get('output_tokens')}")
