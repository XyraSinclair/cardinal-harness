"""grid16 live round 2 — single-call 16-bin PMF read + codebook-calibration check.

TOURNAMENT.md's efficiency ceiling, demanded live WITH a calibration check
because its synthetic failure mode (codebook edge shift) is silently
overconfident, and BINCDF.md finding 3 proved live that code surface form
modulates mass. This probe measures grid16's real-provider validity on
three invariance axes plus cross-instrument agreement:

- codebook relabeling: variants `asc` (letters C..R map lowest->highest
  ratio bin) and `desc` (reversed). Semantically identical instruments;
  any Z-distribution difference is codebook contamination. Metric:
  total-variation distance after alignment.
- mirror: presenting (A,B) vs (B,A) should negate Z. Metric: TV between
  PMF_AB and bin-flipped PMF_BA.
- provider channel: Azure gpt-5.x returns DYNAMIC sidebands (3-4 entries,
  ~0.985 mass, independent of requested top_logprobs; measured 2026-08-13);
  4.1-mini honors 20 -> true single-call PMF. Enumerated mass is reported
  per cell — grid16 on 5.x is a bounded-residual read, not a full PMF.
- cross-instrument: E[Z] vs the decimal-grammar instrument (harvest pack,
  sign negated) and vs rough truth; also vs the binary staircase mu_hat
  (bincdf_summary.json) where present.

Usage: python3 grid16_probe.py run       # 144 calls, ~$0.05
       python3 grid16_probe.py analyze
"""

import concurrent.futures as cf
import json
import math
import sys
import time
import urllib.error
import urllib.request

KEYFILE = '/tmp/.orkey-decimal-pmf'
OUT = 'grid16_results.json'

LETTERS = list('CDEFGHIJKLMNOPQR')  # 16, skipping A/B (entity-tag collision)
NBINS = 16
BIN_W = 0.375                       # log10 units; domain [-3, 3]
CENTERS = [-3.0 + BIN_W * (k + 0.5) for k in range(NBINS)]
BOUNDS = ['1', '2.4', '5.6', '13', '32', '75', '178', '422']

SYS = ('You are an expert subjective evaluator. Compare two entities by an '
       'attribute; choose the single codebook letter for the ratio bucket. '
       'Answer JSON {"code": "<letter>"}.')

PAIRS = {
    'egg-vs-bowling-ball': ('a chicken egg', 'a bowling ball', 'mass', -1.98),
    'cat-vs-raccoon': ('an adult house cat', 'an adult raccoon', 'mass', -0.18),
    'whale-vs-elephant': ('an adult blue whale', 'an adult African elephant', 'mass', 1.40),
    'sweden-vs-portugal': ('Sweden', 'Portugal', 'human population', 0.01),
}

MODELS = ['openai/gpt-5.4-mini', 'openai/gpt-4.1-mini', 'openai/gpt-5.6-sol']
REPS = 3

SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'code', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False, 'required': ['code'],
    'properties': {'code': {'type': 'string', 'enum': LETTERS}}}}}


def bin_texts():
    """Bucket descriptions, Z-bin order 0 (B >> A) .. 15 (A >> B)."""
    out = ['entity_B is more than %s times entity_A' % BOUNDS[7]]
    for k in range(7, 0, -1):  # bins 1..7
        lo = BOUNDS[k - 1]
        hi = BOUNDS[k]
        out.append('entity_B is %s to %s times entity_A' % (lo, hi))
    for k in range(7):         # bins 8..14
        out.append('entity_A is %s to %s times entity_B' % (BOUNDS[k], BOUNDS[k + 1]))
    out.append('entity_A is more than %s times entity_B' % BOUNDS[7])
    return out


def letter_for_bin(k, variant):
    return LETTERS[k] if variant == 'asc' else LETTERS[NBINS - 1 - k]


def user_prompt(a, b, attr, variant):
    lines = []
    for k, text in enumerate(bin_texts()):
        lines.append('%s: %s' % (letter_for_bin(k, variant), text))
    return ('Compare by %s.\n<entity_A>%s</entity_A>\n<entity_B>%s</entity_B>\n'
            'Codebook - choose the single code whose bucket contains the true '
            'ratio by %s:\n%s\nJSON:' % (attr, a, b, attr, '\n'.join(lines)))


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


def extract_masses(resp):
    """-> (letter->mass at the code position, chosen letter) or None.
    Chosen-token mass is exact (census 2b); sidebands add exact masses for
    whatever the provider returns (dynamic on Azure gpt-5.x)."""
    ts = resp['choices'][0]['logprobs']['content']
    seen = ''
    for t in ts:
        prev = seen
        seen += t['token']
        tok = t['token'].strip().strip('"')
        if 'code' in prev and tok in LETTERS:
            masses = {tok: math.exp(t['logprob'])}
            for alt in t.get('top_logprobs', []):
                a = alt['token'].strip().strip('"')
                if a in LETTERS and a not in masses:
                    masses[a] = math.exp(alt['logprob'])
            return masses, tok
    return None


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
               'order': order, 'rep': rep, 'masses': None, 'chosen': None,
               'cost': None}
        if resp is not None:
            ex = extract_masses(resp)
            if ex is not None:
                rec['masses'], rec['chosen'] = ex
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
    ok = sum(1 for r in results if r['masses'])
    print('done: %d/%d ok, spend $%.4f -> %s' %
          (ok, len(results), sum(r['cost'] or 0 for r in results), OUT), flush=True)


# ---------------------------------------------------------------- analysis

def cell_pmf(rows):
    """rows (one model/pair/variant/order) -> (bin_probs renormalized,
    enumerated_mass, drift) with letter->bin via the variant mapping."""
    from collections import defaultdict
    obs = defaultdict(list)
    for r in rows:
        if r['masses']:
            for letter, m in r['masses'].items():
                obs[letter].append(m)
    if not obs:
        return None
    variant = rows[0]['variant']
    mass = [0.0] * NBINS
    drift = 0.0
    for letter, ms in obs.items():
        mean = sum(ms) / len(ms)
        if len(ms) > 1 and mean > 1e-3:
            drift = max(drift, (max(ms) - min(ms)) / max(ms))
        k = LETTERS.index(letter)
        b = k if variant == 'asc' else NBINS - 1 - k
        mass[b] = mean
    tot = sum(mass)
    return [m / tot for m in mass], tot, drift


def moments(p):
    e = sum(pi * c for pi, c in zip(p, CENTERS))
    v = sum(pi * (c - e) ** 2 for pi, c in zip(p, CENTERS))
    return e, math.sqrt(v)


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def decimal_e(model, pair):
    try:
        cells = json.load(open('harvest_results.json'))
    except OSError:
        return None
    for c in cells:
        if c['model'] == model and c['pair'] == pair:
            return -float(c['certificate']['E_Z_head_renormalized'])
    return None


def bincdf_mu(model, pair):
    try:
        s = json.load(open('bincdf_summary.json'))
    except OSError:
        return None
    for c in s:
        if c['model'] == model and c['pair'] == pair:
            return c['mu']
    return None


def analyze():
    results = json.load(open(OUT))
    from collections import defaultdict
    cells = defaultdict(list)
    for r in results:
        cells[(r['model'], r['pair'], r['variant'], r['order'])].append(r)

    print('%-20s %-20s  E[Z]a/d   sd_a   TVcode  TVmirr  enum   drift  '
          'dec_E   bin_mu  truth' % ('model', 'pair'))
    summary = []
    for model in MODELS:
        for pair, (_, _, _, truth) in PAIRS.items():
            got = {}
            for variant in ('asc', 'desc'):
                for order in ('AB', 'BA'):
                    c = cell_pmf(cells.get((model, pair, variant, order), []))
                    if c:
                        got[(variant, order)] = c
            if len(got) < 4:
                print('%-20s %-20s  INSUFFICIENT' % (model, pair))
                continue
            pa, ea_mass, dr_a = got[('asc', 'AB')]
            pd, ed_mass, dr_d = got[('desc', 'AB')]
            e_a, sd_a = moments(pa)
            e_d, _ = moments(pd)
            tv_code = tv(pa, pd)
            # mirror: BA flipped should match AB, per variant, averaged
            tvm = 0.0
            for variant in ('asc', 'desc'):
                pab = got[(variant, 'AB')][0]
                pba = got[(variant, 'BA')][0]
                tvm += tv(pab, list(reversed(pba)))
            tvm /= 2
            enum = min(ea_mass, ed_mass)
            drift = max(dr_a, dr_d)
            de = decimal_e(model, pair)
            bm = bincdf_mu(model, pair)
            print('%-20s %-20s  %+.2f/%+.2f  %.2f   %.3f   %.3f   %.3f  %.3f  '
                  '%s  %s  %+.2f' %
                  (model, pair, e_a, e_d, sd_a, tv_code, tvm, enum, drift,
                   ('%+.2f' % de) if de is not None else '  --  ',
                   ('%+.2f' % bm) if bm is not None else '  --  ', truth))
            summary.append({
                'model': model, 'pair': pair, 'E_asc': e_a, 'E_desc': e_d,
                'sd_asc': sd_a, 'tv_codebook': tv_code, 'tv_mirror': tvm,
                'enumerated_mass': enum, 'drift': drift, 'decimal_E': de,
                'bincdf_mu': bm, 'truth': truth,
                'pmf_asc_AB': pa, 'pmf_desc_AB': pd})
    json.dump(summary, open('grid16_summary.json', 'w'), indent=1)
    print('-> grid16_summary.json')


if __name__ == '__main__':
    {'run': run, 'analyze': analyze}[sys.argv[1]]()
