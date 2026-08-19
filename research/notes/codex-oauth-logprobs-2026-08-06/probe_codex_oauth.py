"""Probe: logprobs from the ChatGPT-subscription Responses backend (codex oauth), 2026-08-06.

Question: docs/LOGPROBS.md establishes the reasoning gate (logprobs only at
reasoning_effort "none") on the official API and OpenRouter. Does the
subscription-billed codex backend (chatgpt.com/backend-api/codex/responses)
serve logprobs at all — and does it accept effort "none" / a two-phase
reason-then-read shape?

Route: through the cxp pool proxy, which injects the Authorization and
ChatGPT-Account-Id headers. This script never touches tokens. Marginal cost:
subscription quota only, ~10 small calls per run.

Run: python3 notes/codex-oauth-logprobs-2026-08-06/probe_codex_oauth.py
"""
import json, math, os, uuid, urllib.error, urllib.request

PORT = json.load(open(os.path.expanduser("~/.codexpool/port")))["port"]
URL = f"http://127.0.0.1:{PORT}/backend-api/codex/responses"

SYS = ("You are an expert quantitative comparator. Compare entity A and entity B "
       "by the given attribute and answer with exactly one letter: A or B. "
       "The answer is the entity with MORE of the attribute.")
USER = ("Attribute: mass.\n<entity_A>a chicken egg</entity_A>\n"
        "<entity_B>a bowling ball</entity_B>\nAnswer with exactly one letter: A or B.")

def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "session_id": str(uuid.uuid4()),
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return 200, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]
    except Exception as e:  # connection-level
        return -1, repr(e)[:200]

def parse_stream(raw):
    """Return the response object, with output rebuilt from output_item.done events.

    The codex backend's response.completed event carries an EMPTY output list
    (measured 2026-08-06); the real items — including logprobs — arrive only in
    the incremental response.output_item.done events.
    """
    resp_obj, items = None, []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "response.output_item.done":
            items.append(ev.get("item"))
        elif ev.get("type") == "response.completed":
            resp_obj = ev.get("response")
    if resp_obj is not None and not resp_obj.get("output"):
        resp_obj["output"] = items
    if resp_obj is not None:
        return resp_obj
    try:  # non-stream JSON body fallback
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("object") == "response":
            return obj
    except json.JSONDecodeError:
        pass
    return None

def extract_logprobs(resp_obj):
    """Return (text, [(token, prob, n_alternatives)]) for message output, else (text, None)."""
    text, lps = "", None
    for item in resp_obj.get("output", []):
        if item.get("type") != "message":
            continue
        for c in item.get("content", []):
            text += c.get("text", "")
            if c.get("logprobs"):
                lps = [(t.get("token"), round(math.exp(t.get("logprob", -99)), 4),
                        len(t.get("top_logprobs") or [])) for t in c["logprobs"]]
    return text, lps

def base_body(effort, logprobs, model="gpt-5.6-sol", stream=True, user=USER, extra_input=None):
    inp = [{"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": user}]}]
    if extra_input:
        inp = extra_input + inp
    b = {"model": model, "instructions": SYS, "input": inp,
         "store": False, "stream": stream}
    if effort is not None:
        b["reasoning"] = {"effort": effort, "summary": "auto"}
    if logprobs:
        # top_logprobs is rejected by this backend ("Unsupported parameter",
        # measured 2026-08-06 at every effort) — include-only is the working shape.
        b["include"] = ["message.output_text.logprobs"]
    return b

def reasoning_tokens(resp_obj):
    return ((resp_obj.get("usage") or {}).get("output_tokens_details") or {}).get("reasoning_tokens")

def show(tag, body):
    code, raw = post(body)
    if code != 200:
        print(f"  {tag}: HTTP {code} {raw[:220]!r}")
        return None
    resp_obj = parse_stream(raw)
    if resp_obj is None:
        print(f"  {tag}: 200 but no response.completed; head={raw[:200]!r}")
        return None
    text, lps = extract_logprobs(resp_obj)
    rt = reasoning_tokens(resp_obj)
    if lps:
        print(f"  {tag}: LOGPROBS ntok={len(lps)} reasoning_tok={rt} "
              f"text={text[:40]!r} head={lps[:6]}")
    else:
        print(f"  {tag}: 200, no logprobs; reasoning_tok={rt} text={text[:60]!r}")
    return resp_obj

def main():
    print(f"probing {URL}")
    print("== single-phase effort ladder, logprobs requested each time")
    show("E none     ", base_body("none", True))
    show("E minimal  ", base_body("minimal", True))
    show("E low      ", base_body("low", True))
    show("E unset    ", base_body(None, True))
    print("== control: effort low, no logprobs requested (backend baseline)")
    show("C low bare ", base_body("low", False))

    print("== two-phase: medium analysis, then fresh read call with analysis in context")
    analysis_user = USER.replace("Answer with exactly one letter: A or B.",
                                 "Analyze this comparison carefully. Do NOT give a verdict.")
    r1 = show("P1 medium  ", base_body("medium", False, user=analysis_user))
    if r1 is not None:
        analysis_text, _ = extract_logprobs(r1)
        ctx = [{"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": analysis_user}]},
               {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": analysis_text}]}]
        for eff in ("none", "minimal", "low"):
            show(f"P2 {eff:<8}", base_body(eff, True, extra_input=ctx))

if __name__ == "__main__":
    main()
