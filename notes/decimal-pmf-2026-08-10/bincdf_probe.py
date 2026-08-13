"""bin-lp live round 1 — the percept-CDF check (TOURNAMENT.md verdict, step 1).

The synthetic tournament's winning geometry (bin-lp: adaptive offset-binary
staircase read via logprobs) rests on one psychometrically loaded assumption:
for a threshold probe "is A more than R times B?", the answer-token
probability equals the percept CDF, p(yes | h=log10 R) = P(Z > h), with a
percept distribution that does not depend on the threshold in context.

This probe measures that directly. For each (model, pair) it sweeps a fixed
ratio ladder in BOTH presentation orders under a byte-identical 2-enum
grammar and reads the exact answer-token mass (chosen-token logprob +
renormalized yes/no sideband mass — the census-verified masked+renormalized
semantics, RESULTS.md #1/#2b). Stitching the two orders gives the full
signed CDF: F(h) = 1 - p_AB(h) for h >= 0, F(-h) = p_BA(h).

Deliverables per cell:
- monotonicity violations of the stitched CDF (beyond measured drift);
- logistic MLE fit -> mu_hat (percept location), s_hat (percept scale),
  max |residual| (smooth-sigmoid vs step diagnosis: a deterministic
  comparator shows s_hat -> 0 / a cliff between adjacent ladder points);
- order-consistency at h=0: (1 - p_AB(1.0)) vs p_BA(1.0) both estimate F(0);
- repeat drift at anchor thresholds;
- cross-instrument check vs the decimal-grammar PMF (harvest_results.json)
  on the census pairs: same latent, different grammar => CDFs should agree
  if the percept model is real and instrument-invariant.

Usage: python3 bincdf_probe.py run       # ~460 calls, writes bincdf_results.json
       python3 bincdf_probe.py analyze   # reads json, prints the reading
Spend: ~$0.1-0.3 on the vault openpriors key (/tmp/.orkey-decimal-pmf).
"""

import concurrent.futures as cf
import json
import math
import sys
import time
import urllib.error
import urllib.request

KEYFILE = '/tmp/.orkey-decimal-pmf'
OUT = 'bincdf_results.json'

SYS = ('You are an expert subjective evaluator. Compare two entities by an '
       'attribute; judge whether the first has more than a given multiple of '
       'the second. Answer JSON {"answer": "yes"|"no"}.')

SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'thr', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False, 'required': ['answer'],
    'properties': {'answer': {'type': 'string', 'enum': ['yes', 'no']}}}}}

# name -> (entity_A, entity_B, attribute, rough truth log10(A/B)) — truth is
# orientation/sanity only; the CDF-shape verdict does not use it.
PAIRS = {
    'egg-vs-bowling-ball': ('a chicken egg', 'a bowling ball', 'mass', -1.98),
    'cat-vs-raccoon': ('an adult house cat', 'an adult raccoon', 'mass', -0.18),
    'whale-vs-elephant': ('an adult blue whale', 'an adult African elephant', 'mass', 1.40),
    'sweden-vs-portugal': ('Sweden', 'Portugal', 'human population', 0.01),
}

MODELS = [  # (slug, top_logprobs)
    ('openai/gpt-5.4-mini', 5),
    ('openai/gpt-4.1-mini', 20),
    ('openai/gpt-5.6-sol', 5),
]

LADDER = [1.0, 1.5, 2.2, 3.3, 5.0, 7.5, 11.0, 17.0, 25.0, 40.0, 65.0,
          100.0, 160.0, 250.0, 400.0]
ANCHOR_REPS = {1.0: 3, 7.5: 3}  # total calls at these thresholds (else 1)


def user_prompt(a, b, attr, r):
    return ('Compare by %s.\n<entity_A>%s</entity_A>\n<entity_B>%s</entity_B>\n'
            'Is entity_A more than %.1f times entity_B by %s? JSON:'
            % (attr, a, b, r, attr))


def call(key, model, top_lp, usr):
    body = {'model': model,
            'messages': [{'role': 'system', 'content': SYS},
                         {'role': 'user', 'content': usr}],
            'max_completion_tokens': 300, 'logprobs': True,
            'top_logprobs': top_lp, 'temperature': 1.0,
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
            return None  # no logprobs: failed draw, never fabricate
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def extract_p_yes(resp):
    """Exact renormalized yes-mass at the answer token position.

    Returns (p_yes, chosen, other_seen) or None. p_yes is renormalized over
    the yes/no mass actually observed (chosen exact + top-k sidebands); the
    grammar masks to the 2-enum so unobserved legal mass is the complement
    of the chosen branch only when the sibling is visible — when it is not,
    we fall back to p(chosen) vs 1-p(chosen), still exact under the
    masked+renormalized semantics.
    """
    ts = resp['choices'][0]['logprobs']['content']
    seen = ''
    for t in ts:
        prev = seen
        seen += t['token']
        tok = t['token'].strip().strip('"').lower()
        if 'answer' in prev and tok in ('yes', 'no'):
            p_chosen = math.exp(t['logprob'])
            masses = {'yes': 0.0, 'no': 0.0}
            masses[tok] = p_chosen
            other = 'no' if tok == 'yes' else 'yes'
            sib_seen = False
            for alt in t.get('top_logprobs', []):
                a = alt['token'].strip().strip('"').lower()
                if a == other:
                    masses[other] = max(masses[other], math.exp(alt['logprob']))
                    sib_seen = True
            if not sib_seen:
                masses[other] = max(0.0, 1.0 - p_chosen)
            z = masses['yes'] + masses['no']
            if z <= 0:
                return None
            return masses['yes'] / z, tok, sib_seen
    return None


def run():
    key = open(KEYFILE).read().strip()
    jobs = []  # (model, top_lp, pair, order, r, rep)
    for model, top_lp in MODELS:
        for pname, (a, b, attr, _) in PAIRS.items():
            for order in ('AB', 'BA'):
                ea, eb = (a, b) if order == 'AB' else (b, a)
                for r in LADDER:
                    for rep in range(ANCHOR_REPS.get(r, 1)):
                        jobs.append((model, top_lp, pname, order, ea, eb, attr, r, rep))
    print('%d calls queued' % len(jobs), flush=True)
    results = []

    def work(j):
        model, top_lp, pname, order, ea, eb, attr, r, rep = j
        resp = call(key, model, top_lp, user_prompt(ea, eb, attr, r))
        rec = {'model': model, 'pair': pname, 'order': order, 'ratio': r,
               'rep': rep, 'p_yes': None, 'chosen': None, 'sibling_seen': None,
               'cost': None}
        if resp is not None:
            ex = extract_p_yes(resp)
            if ex is not None:
                rec['p_yes'], rec['chosen'], rec['sibling_seen'] = ex
            u = resp.get('usage', {})
            rec['cost'] = u.get('cost')
        return rec

    done = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for rec in ex.map(work, jobs):
            results.append(rec)
            done += 1
            if done % 40 == 0:
                print('  %d/%d' % (done, len(jobs)), flush=True)
    json.dump(results, open(OUT, 'w'), indent=1)
    ok = sum(1 for r in results if r['p_yes'] is not None)
    spend = sum(r['cost'] or 0 for r in results)
    print('done: %d/%d ok, spend $%.4f -> %s' % (ok, len(results), spend, OUT), flush=True)


# ---------------------------------------------------------------- analysis

def logistic_mle(points):
    """points: [(h, p, w)] -> (mu, s, max_abs_resid). Fit F(h)=sigmoid(-(h-mu)/s)
    ... i.e. P(yes at threshold h) viewpoint is already converted: points are
    (h, F(h)) CDF samples; fit F(h) = 1/(1+exp(-(h-mu)/s)) by grid+refine MLE
    on the Bernoulli likelihood weighted by w."""
    def nll(mu, s):
        tot = 0.0
        for h, p, w in points:
            q = 1.0 / (1.0 + math.exp(-(h - mu) / s))
            q = min(max(q, 1e-9), 1 - 1e-9)
            tot -= w * (p * math.log(q) + (1 - p) * math.log(1 - q))
        return tot
    best = None
    for mu in [x * 0.05 - 3.0 for x in range(121)]:
        for s in [0.02, 0.05, 0.1, 0.15, 0.22, 0.32, 0.47, 0.7, 1.0, 1.5]:
            v = nll(mu, s)
            if best is None or v < best[0]:
                best = (v, mu, s)
    _, mu, s = best
    for _ in range(40):  # coordinate refine
        for dmu in (0.02, -0.02):
            if nll(mu + dmu, s) < nll(mu, s):
                mu += dmu
        for fs in (1.05, 1 / 1.05):
            if nll(mu, s * fs) < nll(mu, s):
                s *= fs
    resid = max(abs(p - 1.0 / (1.0 + math.exp(-(h - mu) / s)))
                for h, p, w in points)
    return mu, s, resid


def stitched_cdf(rows):
    """rows for one (model, pair) -> [(h, F(h))] averaged over reps."""
    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        if r['p_yes'] is None:
            continue
        h = math.log10(r['ratio'])
        if r['order'] == 'AB':
            acc[round(h, 6)].append(1.0 - r['p_yes'])     # F(h)
        else:
            acc[round(-h, 6)].append(r['p_yes'])          # F(-h)
    return sorted((h, sum(v) / len(v), len(v)) for h, v in acc.items())


def drift_of(rows):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        if r['p_yes'] is not None:
            g[(r['order'], r['ratio'])].append(r['p_yes'])
    worst = 0.0
    for v in g.values():
        if len(v) > 1:
            worst = max(worst, max(v) - min(v))
    return worst


def decimal_cdf_from_harvest(model, pair, hs):
    """Decimal-grammar instrument CDF at the h grid, from harvest top_atoms
    (renormalized over enumerated mass). None if pack lacks the cell."""
    try:
        cells = json.load(open('harvest_results.json'))
    except OSError:
        return None
    for c in cells:
        if c['model'] == model and c['pair'] == pair:
            atoms = c.get('top_atoms')
            if not atoms:
                return None
            zs = []
            for a in atoms:
                # harvest Z = log10(B/A); this probe's Z = log10(A/B) — negate.
                d, ratio, mass = a['dir'], float(a['ratio']), float(a['mass'])
                z = (-1 if d == 'B' else 1) * math.log10(max(ratio, 1.0))
                zs.append((z, mass))
            tot = sum(m for _, m in zs)
            if tot <= 0:
                return None
            out = []
            for h in hs:
                out.append(sum(m for z, m in zs if z <= h) / tot)
            return out
    return None


def analyze():
    results = json.load(open(OUT))
    from collections import defaultdict
    cells = defaultdict(list)
    for r in results:
        cells[(r['model'], r['pair'])].append(r)
    print('%-22s %-22s  mu_hat  s_hat  maxres  mono  h0gap  drift  n_ok' %
          ('model', 'pair'))
    summary = []
    for (model, pair), rows in sorted(cells.items()):
        pts = stitched_cdf(rows)
        n_ok = sum(1 for r in rows if r['p_yes'] is not None)
        if len(pts) < 5:
            print('%-22s %-22s  INSUFFICIENT (%d pts, %d ok)' % (model, pair, len(pts), n_ok))
            continue
        mu, s, resid = logistic_mle([(h, p, w) for h, p, w in pts])
        mono = sum(1 for i in range(1, len(pts)) if pts[i][1] < pts[i - 1][1] - 0.02)
        # order consistency at h=0: F(0) from AB vs from BA
        f0ab = [1 - r['p_yes'] for r in rows
                if r['order'] == 'AB' and r['ratio'] == 1.0 and r['p_yes'] is not None]
        f0ba = [r['p_yes'] for r in rows
                if r['order'] == 'BA' and r['ratio'] == 1.0 and r['p_yes'] is not None]
        h0gap = (abs(sum(f0ab) / len(f0ab) - sum(f0ba) / len(f0ba))
                 if f0ab and f0ba else float('nan'))
        dr = drift_of(rows)
        print('%-22s %-22s  %+.3f  %.3f  %.3f  %4d  %.3f  %.3f  %4d' %
              (model, pair, mu, s, resid, mono, h0gap, dr, n_ok))
        summary.append({'model': model, 'pair': pair, 'mu': mu, 's': s,
                        'max_resid': resid, 'mono_violations': mono,
                        'h0_gap': h0gap, 'drift': dr,
                        'cdf_points': [(h, p) for h, p, _ in pts]})
        dc = decimal_cdf_from_harvest(model, pair, [h for h, _, _ in pts])
        if dc is not None:
            gap = max(abs(p - q) for (_, p, _), q in zip(pts, dc))
            print('%-22s %-22s    vs decimal-grammar CDF: max|dF| = %.3f' %
                  ('', '', gap))
            summary[-1]['decimal_cdf_max_gap'] = gap
    json.dump(summary, open('bincdf_summary.json', 'w'), indent=1)
    print('-> bincdf_summary.json')


if __name__ == '__main__':
    {'run': run, 'analyze': analyze}[sys.argv[1]]()
