"""Probe #1 + #2 from DESIGN.md de-risk sequence. 2026-08-10.

#1 grammar-mask semantics: under strict json_schema, are returned top-5 logprobs
   renormalized to grammar-legal tokens or raw model logprobs?
   Discriminator: the enum position ("A"|"B") — raw logprobs may show non-enum
   alternatives; masked ones cannot.
#2 logit_bias peeling: does logit_bias(-100) on the top ratio token survive at
   effort=none, remove that token, and renormalize survivors exactly by
   1/(1-p_top1)?

Cells (OpenRouter, vault openpriors key stashed 0600 at /tmp/.orkey-decimal-pmf
by the session that runs this; direct-OpenAI key was stale 2026-08-10):
  A openai/gpt-5.4-mini  effort=none schema  logprobs5           baseline
  B openai/gpt-5.4-mini  effort=none schema  logprobs5 + bias    peel under schema
  C openai/gpt-5.4-mini  effort=none plain   logprobs5 + bias    peel without schema
  D openai/gpt-4.1-mini  (no effort)  schema logprobs5 + bias    control (bias known-good)
  E openai/gpt-5.6-sol   effort=none schema  logprobs5 + bias    flagship
"""
import json, math, urllib.error, urllib.request

KEY = open('/tmp/.orkey-decimal-pmf').read().strip()

SYS = ('You are an expert subjective evaluator. Compare two entities by an attribute; '
       'estimate how many times more the higher one has. Answer JSON '
       '{"higher_ranked": "A"|"B", "ratio": "<decimal like 12.5>"}.')
USR = ('Compare by mass.\n<entity_A>a chicken egg</entity_A>\n'
       '<entity_B>a bowling ball</entity_B>\nJSON:')

SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'cmp', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False,
    'required': ['higher_ranked', 'ratio'],
    'properties': {'higher_ranked': {'type': 'string', 'enum': ['A', 'B']},
                   'ratio': {'type': 'string', 'pattern': '^[0-9]{1,3}\\.[0-9]$'}}}}}


def post(body):
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY,
                                          'Content-Type': 'application/json'})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {'error': {'message': e.read().decode()[:300]}}


def call(model, effort, schema, bias=None):
    b = {'model': model,
         'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': USR}],
         'max_completion_tokens': 500, 'logprobs': True, 'top_logprobs': 5}
    if effort is not None:
        b['reasoning'] = {'effort': effort}
    if schema:
        b['response_format'] = SCHEMA
    if bias:
        b['logit_bias'] = bias
    return post(b)


def toks(r):
    return r['choices'][0]['logprobs']['content']


def show(tag, code, r):
    if code != 200:
        print(tag, 'HTTP', code, (r.get('error') or {}).get('message', '')[:200])
        return None
    content = r['choices'][0]['message']['content']
    lp = r['choices'][0].get('logprobs')
    if not lp or not lp.get('content'):
        print(tag, 'provider=%s 200 but NO LOGPROBS content=%r' % (r.get('provider'), content[:60]))
        return None
    ts = toks(r)
    print(tag, 'provider=%s content=%r ntok=%d' % (r.get('provider'), content, len(ts)))
    # positions of interest: the enum value and the first ratio digit token
    for i, t in enumerate(ts):
        tops = [(x['token'], round(math.exp(x['logprob']), 4)) for x in t.get('top_logprobs', [])]
        print('   pos %2d %-8r top5=%s' % (i, t['token'], tops))
    return ts


def find_first_digit_pos(ts, after_substr='ratio'):
    seen = ''
    armed = False
    for i, t in enumerate(ts):
        seen += t['token']
        if after_substr in seen:
            armed = True
        if armed and t['token'][:1].isdigit():
            return i
    return None


def main():
    results = {}
    print('=== A: gpt-5.4-mini effort=none schema baseline ===')
    code, r = call('openai/gpt-5.4-mini', 'none', True)
    ts = show('A', code, r)
    results['A'] = (code, r)
    if ts is None:
        return
    pos = find_first_digit_pos(ts)
    print('first ratio-digit position:', pos)
    if pos is None:
        return
    top1 = ts[pos]['top_logprobs'][0]
    p_top1 = math.exp(top1['logprob'])
    # token id for logit_bias: encode the exact token string with o200k
    import tiktoken
    enc = tiktoken.get_encoding('o200k_base')
    ids = enc.encode(top1['token'])
    print('peel target: token %r p=%.4f ids=%s' % (top1['token'], p_top1, ids))
    if len(ids) != 1:
        print('WARNING: top token not a single o200k id; bias may miss')
    bias = {str(ids[0]): -100}

    print('=== B: same + logit_bias{%s:-100} (peel under schema) ===' % ids[0])
    code, r = call('openai/gpt-5.4-mini', 'none', True, bias)
    tsb = show('B', code, r)
    results['B'] = (code, r)
    if tsb is not None:
        posb = find_first_digit_pos(tsb)
        if posb is not None:
            got = {x['token']: math.exp(x['logprob']) for x in tsb[posb]['top_logprobs']}
            print('peel check: banned token present in top5?', top1['token'] in got)
            base = {x['token']: math.exp(x['logprob']) for x in ts[pos]['top_logprobs']}
            for tk, pb in sorted(got.items(), key=lambda kv: -kv[1]):
                if tk in base:
                    pred = base[tk] / (1 - p_top1)
                    print('   %r peeled=%.4f predicted=%.4f ratio=%.3f'
                          % (tk, pb, pred, pb / pred if pred else float('nan')))

    print('=== C: gpt-5.4-mini effort=none PLAIN + bias ===')
    code, r = call('openai/gpt-5.4-mini', 'none', False, bias)
    show('C', code, r)

    print('=== D: gpt-4.1-mini schema + bias (control) ===')
    code, r = call('openai/gpt-4.1-mini', None, True, bias)
    show('D', code, r)

    print('=== E: gpt-5.6-sol effort=none schema + bias (flagship) ===')
    code, r = call('openai/gpt-5.6-sol', 'none', True, bias)
    show('E', code, r)


if __name__ == '__main__':
    main()
