"""Does prompt_cache_key through OPENROUTER produce OpenAI cache hits?

The harness's live rail is OpenRouter (src/gateway/openrouter.rs serializes
prompt_cache_key into the body). The 12/12 warm-hit measurement in
docs/LOGPROBS.md was DIRECT OpenAI. This probe closes the gap: same shape
(long stable prefix, nonce at tail, prompt_cache_key), through OpenRouter,
provider pinned to openai, reading usage.prompt_tokens_details.cached_tokens.

Run: OPENROUTER_API_KEY must be in env (use vrun; the shell key is capped).
  python3 probe_cache_openrouter.py [--no-key]
"""
import json, os, secrets, sys, urllib.request

KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
PAD = ("Reference context (fixed): " +
       " ".join(f"item-{i:04d} has weight {i * 37 % 101}." for i in range(400)))
STABLE = (f"{PAD}\nAttribute: mass.\n<entity_A>a chicken egg</entity_A>\n"
          "<entity_B>a bowling ball</entity_B>\n"
          "Answer with exactly one letter: A or B.")
use_key = "--no-key" not in sys.argv

print(f"probing OpenRouter -> openai/gpt-5.4-mini, 6 calls, nonce at tail, "
      f"prompt_cache_key={'ON' if use_key else 'OFF'}")
for i in range(6):
    # NOTE: pinning provider:{"only":["openai"]} routes through the account's
    # OpenAI BYOK integration, whose stored key is stale -> provider 401
    # (measured 2026-08-07). Unpinned routing matches the harness rail anyway.
    body = {
        "model": "openai/gpt-5.4-mini",
        "messages": [{"role": "user",
                      "content": f"{STABLE}\ndraw-token: {secrets.token_hex(8)}"}],
        "max_tokens": 4,
        "usage": {"include": True},
    }
    if use_key:
        body["prompt_cache_key"] = "cardinal-probe-or-cache-2026-08-07"
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  {i}: HTTP {e.code} {e.read()[:200]!r}")
        continue
    usage = resp.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    print(f"  {i}: provider={resp.get('provider')} prompt={usage.get('prompt_tokens')} "
          f"cached={det.get('cached_tokens')} cost=${usage.get('cost')}")
