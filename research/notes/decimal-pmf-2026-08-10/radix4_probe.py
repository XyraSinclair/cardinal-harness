"""radix4 live round 3 — deep-grammar digit code validity cell.

The last open challenger from TOURNAMENT.md. radix4 encodes the signed
log10 ratio Z in [-3,3] as a 3-digit base-4 code (64 leaves, 0.09375
log10 wide): d1 picks a quarter of the scale, d2 a quarter of that, d3 a
quarter again. Each digit position is a 4-way enum under the grammar, so
even Azure's dynamic sidebands (3-4 entries) return the FULL conditional
at every visited node — one call reads three full conditionals along the
sampled path.

After GRID16.md the live question is sharp: are abstract positional DIGIT
codes calibrated like native numerals, or contaminated like letters?
Axes, as before:
- relabel: `asc` (0=low end) vs `desc` (digit d -> 3-d everywhere) —
  semantically identical; any Z difference is code contamination;
- mirror: (A,B) vs (B,A) should negate Z;
- depth validity: do d2/d3 conditionals carry pair-dependent information,
  or collapse to a fixed digit prior (pure artifact)?
- cross-instrument: E[Z] vs decimal peel (harvest), grid16, binary mu, truth.

Usage: python3 radix4_probe.py run       # 144 calls, ~$0.15
       python3 radix4_probe.py analyze
"""

import concurrent.futures as cf
import json
import math
import sys
import time
import urllib.error
import urllib.request

KEYFILE = '/tmp/.orkey-decimal-pmf'
OUT = 'radix4_results.json'

DIGITS = ['0', '1', '2', '3']
ZLO, ZHI = -3.0, 3.0
LEAF_W = (ZHI - ZLO) / 64.0

SYS = ('You are an expert subjective evaluator. Compare two entities by an '
       'attribute and encode the ratio as a 3-digit base-4 code. '
       'Answer JSON {"d1": "<digit>", "d2": "<digit>", "d3": "<digit>"}.')

PAIRS = {
    'egg-vs-bowling-ball': ('a chicken egg', 'a bowling ball', 'mass', -1.98),
    'cat-vs-raccoon': ('an adult house cat', 'an adult raccoon', 'mass', -0.18),
    'whale-vs-elephant': ('an adult blue whale', 'an adult African elephant', 'mass', 1.40),
    'sweden-vs-portugal': ('Sweden', 'Portugal', 'human population', 0.01),
}

MODELS = ['openai/gpt-5.4-mini', 'openai/gpt-4.1-mini', 'openai/gpt-5.6-sol']
REPS = 3

SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'radix', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False,
    'required': ['d1', 'd2', 'd3'],
    'properties': {'d1': {'type': 'string', 'enum': DIGITS},
                   'd2': {'type': 'string', 'enum': DIGITS},
                   'd3': {'type': 'string', 'enum': DIGITS}}}}}

D1_TEXT_ASC = [
    'entity_B is more than 32 times entity_A',
    'entity_B is 1 to 32 times entity_A',
    'entity_A is 1 to 32 times entity_B',
    'entity_A is more than 32 times entity_B',
]


def user_prompt(a, b, attr, variant):
    if variant == 'asc':
        d1 = D1_TEXT_ASC
        sub = '0 = lowest quarter (most toward entity_B being bigger) up to 3 = highest quarter (most toward entity_A being bigger)'
    else:
        d1 = list(reversed(D1_TEXT_ASC))
        sub = '0 = highest quarter (most toward entity_A being bigger) up to 3 = lowest quarter (most toward entity_B being bigger)'
    d1lines = '\n'.join('%d = %s' % (i, t) for i, t in enumerate(d1))
    return ('Compare by %s.\n<entity_A>%s</entity_A>\n<entity_B>%s</entity_B>\n'
            'Encode the ratio of entity_A to entity_B by %s as a 3-digit '
            'base-4 code on the signed log scale.\n'
            'd1 - quarter of the full scale:\n%s\n'
            'd2 - quarter WITHIN the d1 range on the log scale: %s.\n'
            'd3 - quarter within the d2 range, same rule.\nJSON:'
            % (attr, a, b, attr, d1lines, sub))


def call(key, model, usr):
    body = {'model': model,
            'messages': [{'role': 'system', 'content': SYS},
                         {'role': 'user', 'content': usr}],
            'max_completion_tokens': 300, 'logprobs': True,
            'top_logprobs': 20, 'temperature': 1.0,
            'response_format': SCHEMA}
    if model.startswith('openai/gpt-5'):
        body['reasoning'] = {'effort': 'none'}
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + key,
                 'Content-Type': 'application/json'})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            lp = r['choices'][0].get('logprobs')
            if lp and lp.get('content'):
                return r
            return None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def extract_nodes(resp):
    """-> {'d1': (chosen, {digit: mass}), 'd2': ..., 'd3': ...} or None."""
    ts = resp['choices'][0]['logprobs']['content']
    out = {}
    seen = ''
    want = ['d1', 'd2', 'd3']
    for t in ts:
        prev = seen
        seen += t['token']
        tok = t['token'].strip().strip('"')
        if tok in DIGITS and want:
            field = want[0]
            if field in prev[-8:] or ('"%s"' % field) in prev[-10:]:
                masses = {tok: math.exp(t['logprob'])}
                for alt in t.get('top_logprobs', []):
                    a = alt['token'].strip().strip('"')
                    if a in DIGITS and a not in masses:
                        masses[a] = math.exp(alt['logprob'])
                out[field] = (tok, masses)
                want.pop(0)
    return out if len(out) == 3 else None


def run():
    key = open(KEYFILE).read().strip()
    jobs = []
    for model in MODELS:
        for pname, (a, b, attr, _) in PAIRS.items():
            for variant in ('asc', 'desc'):
                for order in ('AB', 'BA'):
                    ea, eb = (a, b) if order == 'AB' else (b, a)
                    for rep in range(REPS):
                        jobs.append((model, pname, variant, order, ea, eb, attr, rep))
    print('%d calls queued' % len(jobs), flush=True)

    def work(j):
        model, pname, variant, order, ea, eb, attr, rep = j
        resp = call(key, model, user_prompt(ea, eb, attr, variant))
        rec = {'model': model, 'pair': pname, 'variant': variant,
               'order': order, 'rep': rep, 'nodes': None, 'cost': None}
        if resp is not None:
            ex = extract_nodes(resp)
            if ex is not None:
                rec['nodes'] = ex
            rec['cost'] = resp.get('usage', {}).get('cost')
        return rec

    results = []
    done = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for rec in ex.map(work, jobs):
            results.append(rec)
            done += 1
            if done % 24 == 0:
                print('  %d/%d' % (done, len(jobs)), flush=True)
    json.dump(results, open(OUT, 'w'), indent=1)
    ok = sum(1 for r in results if r['nodes'])
    print('done: %d/%d ok, spend $%.4f -> %s' %
          (ok, len(results), sum(r['cost'] or 0 for r in results), OUT), flush=True)


# ---------------------------------------------------------------- analysis

def canon_digit(d, variant):
    return int(d) if d == 'asc' else 0  # unused; kept simple below


def cell_tree(rows):
    """First-visit-wins conditionals from a cell's reps, digits canonicalized
    to asc semantics. -> {'': {d: p}, 'a': {d: p}, 'ab': {d: p}} keyed by
    canonical visited path prefixes; multiple observations averaged."""
    from collections import defaultdict
    obs = defaultdict(lambda: defaultdict(list))
    variant = rows[0]['variant']

    def canon(d):
        return d if variant == 'asc' else str(3 - int(d))

    for r in rows:
        if not r['nodes']:
            continue
        c1, m1 = r['nodes']['d1']
        c2, m2 = r['nodes']['d2']
        c3, m3 = r['nodes']['d3']
        for d, p in m1.items():
            obs[''][canon(d)].append(p)
        p1 = canon(c1)
        for d, p in m2.items():
            obs[p1][canon(d)].append(p)
        p2 = p1 + canon(c2)
        for d, p in m3.items():
            obs[p2][canon(d)].append(p)
    tree = {}
    for path, dd in obs.items():
        tree[path] = {d: sum(v) / len(v) for d, v in dd.items()}
    return tree


def tree_moments(tree):
    """Midpoint-imputed E[Z] plus enumerated d1 mass. Leaves with unknown
    deeper conditionals put their subtree mass at the subtree center."""
    e = 0.0
    mass_seen = 0.0
    root = tree.get('', {})
    tot = sum(root.values())
    if tot <= 0:
        return None, 0.0
    for d1, p1 in root.items():
        p1n = p1 / tot
        mass_seen += p1n
        lo1 = ZLO + int(d1) * 1.5
        sub1 = tree.get(d1)
        if not sub1:
            e += p1n * (lo1 + 0.75)
            continue
        t1 = sum(sub1.values())
        for d2, p2 in sub1.items():
            p2n = p2 / t1
            lo2 = lo1 + int(d2) * 0.375
            sub2 = tree.get(d1 + d2)
            if not sub2:
                e += p1n * p2n * (lo2 + 0.1875)
                continue
            t2 = sum(sub2.values())
            for d3, p3 in sub2.items():
                lo3 = lo2 + int(d3) * 0.09375
                e += p1n * (p2n * p3 / t2) * (lo3 + LEAF_W / 2)
    return e, mass_seen


def d1_marginal(tree):
    root = tree.get('', {})
    tot = sum(root.values())
    return {d: root.get(d, 0.0) / tot for d in '0123'} if tot > 0 else None


def entropy(dist):
    h = 0.0
    for p in dist.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def refs(model, pair):
    de = gm = bm = None
    try:
        for c in json.load(open('harvest_results.json')):
            if c['model'] == model and c['pair'] == pair:
                de = -float(c['certificate']['E_Z_head_renormalized'])
    except OSError:
        pass
    try:
        for c in json.load(open('grid16_summary.json')):
            if c['model'] == model and c['pair'] == pair:
                gm = c['E_asc']
    except OSError:
        pass
    try:
        for c in json.load(open('bincdf_summary.json')):
            if c['model'] == model and c['pair'] == pair:
                bm = c['mu']
    except OSError:
        pass
    return de, gm, bm


def analyze():
    results = json.load(open(OUT))
    from collections import defaultdict
    cells = defaultdict(list)
    for r in results:
        cells[(r['model'], r['pair'], r['variant'], r['order'])].append(r)

    print('%-20s %-20s  E_asc  E_desc  mirr_a  TVd1   H(d2)  dec_E  g16_E  bin_mu  truth'
          % ('model', 'pair'))
    summary = []
    for model in MODELS:
        for pair, (_, _, _, truth) in PAIRS.items():
            trees = {}
            for variant in ('asc', 'desc'):
                for order in ('AB', 'BA'):
                    rows = cells.get((model, pair, variant, order), [])
                    if rows:
                        trees[(variant, order)] = cell_tree(rows)
            if len(trees) < 4:
                print('%-20s %-20s  INSUFFICIENT' % (model, pair))
                continue
            e_a, _ = tree_moments(trees[('asc', 'AB')])
            e_d, _ = tree_moments(trees[('desc', 'AB')])
            e_ba, _ = tree_moments(trees[('asc', 'BA')])
            m1a = d1_marginal(trees[('asc', 'AB')])
            m1d = d1_marginal(trees[('desc', 'AB')])
            tv_d1 = 0.5 * sum(abs(m1a[d] - m1d[d]) for d in '0123')
            # depth-2 information: entropy of the d2 conditional actually
            # visited most (asc, AB); artifact prior => same H across pairs
            t = trees[('asc', 'AB')]
            subpaths = [p for p in t if len(p) == 1]
            h2 = (sum(entropy({d: v / max(sum(t[p].values()), 1e-12)
                               for d, v in t[p].items()}) for p in subpaths)
                  / len(subpaths)) if subpaths else float('nan')
            de, gm, bm = refs(model, pair)
            mirr = e_a + e_ba  # should be ~0 (E_AB = -E_BA)
            fmt = lambda x: ('%+.2f' % x) if x is not None else '  --'
            print('%-20s %-20s  %+.2f  %+.2f   %+.2f   %.3f  %.2f   %s  %s  %s  %+.2f'
                  % (model, pair, e_a, e_d, mirr, tv_d1, h2,
                     fmt(de), fmt(gm), fmt(bm), truth))
            summary.append({'model': model, 'pair': pair, 'E_asc': e_a,
                            'E_desc': e_d, 'mirror_gap': mirr, 'tv_d1': tv_d1,
                            'H_d2': h2, 'decimal_E': de, 'grid16_E': gm,
                            'bincdf_mu': bm, 'truth': truth,
                            'd1_asc': m1a, 'd1_desc': m1d})
    json.dump(summary, open('radix4_summary.json', 'w'), indent=1)
    print('-> radix4_summary.json')


if __name__ == '__main__':
    {'run': run, 'analyze': analyze}[sys.argv[1]]()
