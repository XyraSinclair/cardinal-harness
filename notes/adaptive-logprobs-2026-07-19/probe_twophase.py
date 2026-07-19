"""Probes A/B/C: information-dense elicitation shapes, direct OpenAI API, 2026-07-19.

A: visible scratchpad at effort=none (analysis field in schema) vs bare schema.
B: two-phase — effort=medium analysis call, then effort=none commit call with the
   analysis as assistant context, logprobs on the commit.
C: Responses API — reasoning call, then previous_response_id continuation at
   effort=none with logprobs (does the API allow the switch?).
"""
import json, math, os, urllib.error, urllib.request

KEY = os.environ["OPENAI_API_KEY"]

PAIRS = {
    'egg_vs_bowling': ('a chicken egg', 'a bowling ball', 'mass'),
    'ice_vs_water': ('a liter of liquid water', 'a liter of ice', 'mass'),
}
SYS = ('You are an expert quantitative comparator. Compare entity A and entity B by the '
       'given attribute. higher_ranked is the entity with MORE of it; ratio is how many '
       'times more (>=1).')

def user_msg(a, b, attr):
    return f'Attribute: {attr}.\n<entity_A>{a}</entity_A>\n<entity_B>{b}</entity_B>'

BARE = {'type':'object','properties':{'higher_ranked':{'type':'string','enum':['A','B']},
        'ratio':{'type':'number'}},'required':['higher_ranked','ratio'],'additionalProperties':False}
SCRATCH = {'type':'object','properties':{'analysis':{'type':'string'},
           'higher_ranked':{'type':'string','enum':['A','B']},'ratio':{'type':'number'}},
           'required':['analysis','higher_ranked','ratio'],'additionalProperties':False}

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
    try: return 200, json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {'error':{'message':e.read().decode()[:200]}}

def chat(model, messages, schema, effort, logprobs=True):
    b = {'model':model,'messages':messages,'max_completion_tokens':4000,
         'response_format':{'type':'json_schema','json_schema':{'name':'cmp','strict':True,'schema':schema}}}
    if effort is not None: b['reasoning_effort'] = effort
    if logprobs: b['logprobs']=True; b['top_logprobs']=5
    return post('https://api.openai.com/v1/chat/completions', b)

def answer_pmfs(resp):
    """Return (direction_pmf, ratio_first_token_pmf) from logprobs content."""
    toks = resp['choices'][0]['logprobs']['content']
    text = ''.join(t['token'] for t in toks)
    out = {}
    seen = ''
    for i, t in enumerate(toks):
        tok = t['token']
        if '"higher_ranked"' in seen and 'dir' not in out and tok.strip('" ') in ('A','B'):
            out['dir'] = {a['token'].strip('" '): math.exp(a['logprob']) for a in t['top_logprobs']}
        if '"ratio"' in seen and 'ratio' not in out and tok.strip().rstrip('.').replace('.','',1).isdigit():
            out['ratio'] = {a['token']: round(math.exp(a['logprob']),4) for a in t['top_logprobs']}
        seen += tok
    return out, text

def show(tag, code, r):
    if code != 200:
        print(f'  {tag}: HTTP {code} {(r.get("error") or {}).get("message","")[:120]}')
        return None
    lp = r['choices'][0].get('logprobs')
    usage = r.get('usage', {})
    rt = usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
    if not (lp and lp.get('content')):
        print(f'  {tag}: 200, no logprobs (reasoning_tokens={rt})')
        return None
    pmfs, text = answer_pmfs(r)
    d = pmfs.get('dir'); rat = pmfs.get('ratio')
    print(f'  {tag}: ntok={len(lp["content"])} reasoning_tok={rt} dir={d} ratio_tok={rat}')
    return r

MODEL = 'gpt-5.6-sol'
for pname, (a,b,attr) in PAIRS.items():
    print(f'== {pname} ({MODEL})')
    msgs = [{'role':'system','content':SYS},{'role':'user','content':user_msg(a,b,attr)}]
    # A1 bare, effort=none
    show('A1 bare none      ', *chat(MODEL, msgs, BARE, 'none'))
    # A2 scratchpad, effort=none
    show('A2 scratch none   ', *chat(MODEL, msgs, SCRATCH, 'none'))
    # B: phase 1 medium analysis (no logprobs), phase 2 none commit with context
    code, r1 = chat(MODEL, [{'role':'system','content':SYS},
        {'role':'user','content':user_msg(a,b,attr)+'\nAnalyze this comparison carefully and thoroughly. Do NOT give a final verdict yet.'}],
        {'type':'object','properties':{'analysis':{'type':'string'}},'required':['analysis'],'additionalProperties':False},
        'medium', logprobs=False)
    if code==200:
        analysis = json.loads(r1['choices'][0]['message']['content'])['analysis']
        rt1 = r1.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0)
        print(f'  B phase1: medium reasoning_tok={rt1} analysis_chars={len(analysis)}')
        msgs2 = [{'role':'system','content':SYS},{'role':'user','content':user_msg(a,b,attr)},
                 {'role':'assistant','content':json.dumps({'analysis':analysis})},
                 {'role':'user','content':'Now give the final verdict as JSON.'}]
        show('B  commit none    ', *chat(MODEL, msgs2, BARE, 'none'))
    else:
        print(f'  B phase1: HTTP {code} {(r1.get("error") or {}).get("message","")[:100]}')

# C: Responses API effort switch with previous_response_id
print('== C: Responses API reasoning -> effort=none logprobs continuation')
a,b,attr = PAIRS['egg_vs_bowling']
code, r1 = post('https://api.openai.com/v1/responses', {
    'model': MODEL, 'input': [{'role':'system','content':SYS},
        {'role':'user','content':user_msg(a,b,attr)+'\nAnalyze carefully. No final verdict yet.'}],
    'reasoning': {'effort':'medium'}, 'max_output_tokens': 4000, 'store': True})
if code != 200:
    print(f'  phase1: HTTP {code} {(r1.get("error") or {}).get("message","")[:140]}')
else:
    rid = r1['id']
    rt = (r1.get('usage',{}).get('output_tokens_details') or {}).get('reasoning_tokens')
    print(f'  phase1 ok id={rid} reasoning_tokens={rt}')
    code, r2 = post('https://api.openai.com/v1/responses', {
        'model': MODEL, 'previous_response_id': rid,
        'input': [{'role':'user','content':'Now give the final verdict as JSON {"higher_ranked": "A"|"B", "ratio": number}.'}],
        'reasoning': {'effort':'none'}, 'max_output_tokens': 2000,
        'include': ['message.output_text.logprobs'], 'top_logprobs': 5})
    if code != 200:
        print(f'  phase2: HTTP {code} {(r2.get("error") or {}).get("message","")[:160]}')
    else:
        got = False
        for item in r2.get('output', []):
            if item.get('type')=='message':
                for c in item.get('content', []):
                    lps = c.get('logprobs')
                    if lps:
                        got = True
                        ntop = max(len(t.get('top_logprobs') or []) for t in lps)
                        print(f'  phase2: LOGPROBS ntok={len(lps)} max_top={ntop} text={c.get("text","")[:60]!r}')
        if not got: print(f'  phase2: 200 but no logprobs; status={r2.get("status")}')
