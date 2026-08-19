"""Probe gpt-5.5 / gpt-5.6-* for any logprobs path. 2026-07-18.

Paths tried per model:
  chat:  logprobs=true, top_logprobs=5, effort in {absent, none}
  chat:  logprobs=true only (no top_logprobs)
  chat:  top_logprobs=1 (min ask)
  resp:  Responses API include=["message.output_text.logprobs"], top_logprobs=5, effort none & absent
"""
import json, os, urllib.error, urllib.request

KEY = os.environ["OPENAI_API_KEY"]

SYS = "You compare two entities by an attribute. Output only JSON {\"higher_ranked\": \"A\"|\"B\", \"ratio\": number}."
USR = "Compare by mass.\n<entity_A>a chicken egg</entity_A>\n<entity_B>a bowling ball</entity_B>\nJSON:"

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {'error': {'message': e.read().decode()[:200]}}

def chat(model, effort, top_lp, use_lp=True):
    b = {'model': model, 'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': USR}],
         'max_completion_tokens': 2000}
    if use_lp: b['logprobs'] = True
    if top_lp is not None: b['top_logprobs'] = top_lp
    if effort is not None: b['reasoning_effort'] = effort
    return post('https://api.openai.com/v1/chat/completions', b)

def resp_api(model, effort, top_lp):
    b = {'model': model, 'input': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': USR}],
         'max_output_tokens': 2000, 'include': ['message.output_text.logprobs']}
    if top_lp is not None: b['top_logprobs'] = top_lp
    if effort is not None: b['reasoning'] = {'effort': effort}
    return post('https://api.openai.com/v1/responses', b)

def chat_verdict(code, r):
    if code != 200:
        return 'HTTP %d: %s' % (code, (r.get('error') or {}).get('message', '')[:140])
    ch = r['choices'][0]
    lp = ch.get('logprobs')
    if lp and lp.get('content'):
        toks = lp['content']
        ntop = max(len(t.get('top_logprobs') or []) for t in toks)
        return 'LOGPROBS ntok=%d max_top=%d first=%r' % (len(toks), ntop, toks[0]['token'])
    return '200 but logprobs=%r (finish=%s, content=%r)' % (lp, ch.get('finish_reason'), (ch['message'].get('content') or '')[:30])

def resp_verdict(code, r):
    if code != 200:
        return 'HTTP %d: %s' % (code, (r.get('error') or {}).get('message', '')[:140])
    out = r.get('output') or []
    for item in out:
        if item.get('type') == 'message':
            for c in item.get('content', []):
                lps = c.get('logprobs')
                if lps:
                    ntop = max(len(t.get('top_logprobs') or []) for t in lps)
                    return 'LOGPROBS ntok=%d max_top=%d first=%r' % (len(lps), ntop, lps[0].get('token'))
            return 'message present, no logprobs field (text=%r)' % (item['content'][0].get('text', '')[:30] if item.get('content') else None)
    return '200, no message item (status=%s)' % r.get('status')

MODELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']
for m in MODELS:
    print('==== %s' % m)
    for label, fn in [
        ('chat effort=absent top5 ', lambda: chat(m, None, 5)),
        ('chat effort=none  top5 ', lambda: chat(m, 'none', 5)),
        ('chat effort=none  lponly', lambda: chat(m, 'none', None)),
        ('chat effort=none  top1 ', lambda: chat(m, 'none', 1)),
        ('resp effort=none  top5 ', lambda: resp_api(m, 'none', 5)),
        ('resp effort=absent top5 ', lambda: resp_api(m, None, 5)),
    ]:
        code, r = fn()
        v = chat_verdict(code, r) if label.startswith('chat') else resp_verdict(code, r)
        print('  %s -> %s' % (label, v))
