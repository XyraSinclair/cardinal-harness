"""Follow-up variants: bare logprobs include without top_logprobs; raw output dump.

Run: python3 notes/codex-oauth-logprobs-2026-08-06/probe_variants.py
"""
import json
from probe_codex_oauth import URL, base_body, parse_stream, post

print(f"probing {URL}")
print("== bare include, no top_logprobs param")
for eff in ("none", "low", None):
    b = base_body(eff, False)
    b["include"] = ["message.output_text.logprobs"]
    code, raw = post(b)
    if code != 200:
        print(f"  include-only effort={eff}: HTTP {code} {raw[:220]!r}")
        continue
    resp = parse_stream(raw)
    if resp is None:
        print(f"  include-only effort={eff}: 200, no completed event; head={raw[:200]!r}")
        continue
    msgs = [i for i in resp.get("output", []) if i.get("type") == "message"]
    lp = [c.get("logprobs") for m in msgs for c in m.get("content", [])]
    print(f"  include-only effort={eff}: status={resp.get('status')} "
          f"logprobs={[type(x).__name__ for x in lp]}")

print("== raw output structure of a control call (effort low, no logprobs)")
code, raw = post(base_body("low", False))
resp = parse_stream(raw)
if resp is None:
    print(f"  control: code={code} head={raw[:300]!r}")
else:
    slim = [{k: (v if k in ("type", "role", "status") else
                 (json.dumps(v)[:160] if k == "content" else "..."))
             for k, v in item.items()} for item in resp.get("output", [])]
    print(json.dumps({"status": resp.get("status"), "output": slim}, indent=1)[:1400])
