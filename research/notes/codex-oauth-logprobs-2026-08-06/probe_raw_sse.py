"""Dump raw SSE event types for one codex-backend call to explain empty output.

Run: python3 notes/codex-oauth-logprobs-2026-08-06/probe_raw_sse.py [effort]
"""
import collections, json, sys
from probe_codex_oauth import URL, base_body, post

effort = sys.argv[1] if len(sys.argv) > 1 else "none"
body = base_body(effort, False)
body["include"] = ["message.output_text.logprobs"]
print(f"probing {URL} effort={effort}")
code, raw = post(body)
if code != 200:
    print(f"HTTP {code} {raw!r}")
    sys.exit(1)

counts = collections.Counter()
for line in raw.splitlines():
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        continue
    try:
        ev = json.loads(payload)
    except json.JSONDecodeError:
        counts["<unparseable>"] += 1
        continue
    t = ev.get("type", "<untyped>")
    counts[t] += 1
    if t in ("response.output_item.done", "response.output_text.done",
             "response.completed", "response.failed", "response.incomplete"):
        print(f"-- {t}: {json.dumps(ev)[:600]}")
print("event counts:", dict(counts))
