"""Probe D: prompt-cache + nonce perturbation stability. Probe C2: isolate continuation top-k sparsity.
Direct OpenAI API, 2026-07-19."""
import json, math, os, statistics, urllib.error, urllib.request

KEY = os.environ["OPENAI_API_KEY"]

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
    try: return 200, json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {'error':{'message':e.read().decode()[:200]}}

BARE = {'type':'object','properties':{'higher_ranked':{'type':'string','enum':['A','B']},
        'ratio':{'type':'number'}},'required':['higher_ranked','ratio'],'additionalProperties':False}

# ---- C2: continuation sparsity isolation (gpt-5.6-sol) ----
print('== C2: fresh effort=none Responses call, same commit question (control for continuation sparsity)')
SYS = ('You are an expert quantitative comparator. Compare entity A and entity B by the '
       'given attribute. higher_ranked is the entity with MORE of it; ratio is how many times more (>=1).')
USR = 'Attribute: mass.\n<entity_A>a chicken egg</entity_A>\n<entity_B>a bowling ball</entity_B>'
code, r = post('https://api.openai.com/v1/responses', {
    'model':'gpt-5.6-sol','input':[{'role':'system','content':SYS},{'role':'user','content':USR}],
    'reasoning':{'effort':'none'},'max_output_tokens':2000,
    'include':['message.output_text.logprobs'],'top_logprobs':5,
    'text':{'format':{'type':'json_schema','name':'cmp','strict':True,'schema':BARE}}})
if code==200:
    for item in r.get('output',[]):
        if item.get('type')=='message':
            for c in item.get('content',[]):
                if c.get('logprobs'):
                    ntop=max(len(t.get('top_logprobs') or []) for t in c['logprobs'])
                    print(f'  fresh: ntok={len(c["logprobs"])} max_top={ntop} text={c.get("text","")[:50]!r}')
else:
    print(f'  fresh: HTTP {code} {(r.get("error") or {}).get("message","")[:140]}')

# ---- D: cache + nonce stability (gpt-5.4-mini) ----
print('== D: prompt cache + nonce perturbation (gpt-5.4-mini, effort=none, temp default)')
filler = ('Reference notes on comparison methodology: consider typical instances, standard '
          'conditions, and canonical measurements. Judge by expected values over natural '
          'variation. Entities are considered in their ordinary, unmodified state. ') * 40
LONG_SYS = SYS + '\n\n' + filler  # ~ 1700+ tokens of stable prefix
def one(nonce):
    msgs=[{'role':'system','content':LONG_SYS},
          {'role':'user','content':USR + f'\n<nonce>{nonce}</nonce>'}]
    b={'model':'gpt-5.4-mini','messages':msgs,'max_completion_tokens':2000,
       'reasoning_effort':'none','logprobs':True,'top_logprobs':5,
       'prompt_cache_key':'cardinal-cache-probe-1',
       'response_format':{'type':'json_schema','json_schema':{'name':'cmp','strict':True,'schema':BARE}}}
    code,r=post('https://api.openai.com/v1/chat/completions', b)
    if code!=200: return {'err':(r.get('error') or {}).get('message','')[:80]}
    u=r.get('usage',{}); cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)
    toks=r['choices'][0]['logprobs']['content']
    seen=''; ratio=None
    for t in toks:
        if '"ratio"' in seen and ratio is None and t['token'].strip().replace('.','',1).isdigit():
            ratio={a['token']: round(math.exp(a['logprob']),4) for a in t['top_logprobs']}
        seen+=t['token']
    return {'cached':cached,'prompt_toks':u.get('prompt_tokens'),'ratio':ratio,
            'answer':r['choices'][0]['message']['content']}

results=[]
for i in range(10):
    res=one(f'probe-{i:03d}')
    results.append(res)
    print(f'  nonce {i}: cached={res.get("cached")}/{res.get("prompt_toks")} ratio_pmf={res.get("ratio")} ans={res.get("answer")}')
same=[one('probe-000') for _ in range(3)]
for j,res in enumerate(same):
    print(f'  repeat nonce0 #{j}: cached={res.get("cached")}/{res.get("prompt_toks")} ratio_pmf={res.get("ratio")}')

ok=[r for r in results if r.get('ratio')]
top1=[max(r['ratio'].values()) for r in ok]
tops=[max(r['ratio'], key=r['ratio'].get) for r in ok]
print(f'  SUMMARY across {len(ok)} nonces: top-token set={sorted(set(tops))}, '
      f'top1_p mean={statistics.mean(top1):.3f} sd={statistics.pstdev(top1):.3f}, '
      f'cache hits={sum(1 for r in results if (r.get("cached") or 0)>0)}/10')
