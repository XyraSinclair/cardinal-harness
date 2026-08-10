"""Sophisticated logprob harvester — prototype of the descend/resample kernel.

Implements the reconciled design (DESIGN.md + RESULTS.md + ORACLE.md):
- joint stochastic trie over (direction, int-token, frac-token) under a FIXED
  byte-identical grammar (instrument identity);
- every draw is simultaneously a sample and an exact measurement: chosen-token
  logprobs give exact conditional masses even below top-k, top-k sidebands come
  free at every visited node;
- exact-atom mass ledger with conservation: enumerated + frontier + truncation
  residual = 1 (up to measured provider drift, which is tracked, not hidden);
- credal envelope on Z = log10(higher/lower) signed as log10(B/A), computed
  against a DECLARED bounded domain [1.0, 999.9] (identifiability requires the
  bound; out-of-domain atoms are reported in their own bucket);
- per-call envelope-width trajectory (anytime certificate);
- provider-drift diagnostics (per-token mass spread across observations).

Known v1 limitations (documented, deliberate):
- no prefill => allocation over subtrees is proportional-to-mass (root
  resampling), not Neyman; fine for E[Z], slower for minority-direction cells;
- point estimate uses midpoint imputation for unresolved cells (labeled
  minimax imputation, not a posterior mean).

Usage: python3 harvest.py            # runs the full cell grid, writes JSON
Spend: ~200 calls of mini/sol via OpenRouter (~$0.10-0.20 total).
"""
import concurrent.futures as cf
import json, math, time, urllib.error, urllib.request

KEY = open('/tmp/.orkey-decimal-pmf').read().strip()

SYS = ('You are an expert subjective evaluator. Compare two entities by an attribute; '
       'estimate how many times more the higher one has. Answer JSON '
       '{"higher_ranked": "A"|"B", "ratio": "<decimal like 12.5>"}.')

SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'cmp', 'strict': True, 'schema': {
    'type': 'object', 'additionalProperties': False,
    'required': ['higher_ranked', 'ratio'],
    'properties': {'higher_ranked': {'type': 'string', 'enum': ['A', 'B']},
                   'ratio': {'type': 'string', 'pattern': '^[0-9]{1,3}\\.[0-9]$'}}}}}

DOMAIN_LO, DOMAIN_HI = 1.0, 999.9      # declared instrument domain for ratio
ZMAX = math.log10(DOMAIN_HI)

PAIRS = {
    'egg-vs-bowling-ball': ('a chicken egg', 'a bowling ball', 'mass'),
    'cat-vs-raccoon': ('an adult house cat', 'an adult raccoon', 'mass'),
}
CELLS = [  # (model, top_logprobs, draws)
    ('openai/gpt-5.4-mini', 5, 40),
    ('openai/gpt-4.1-mini', 20, 40),
    ('openai/gpt-5.6-sol', 5, 25),
]


def user_prompt(a, b, attr):
    return ('Compare by %s.\n<entity_A>%s</entity_A>\n<entity_B>%s</entity_B>\nJSON:'
            % (attr, a, b))


def call(model, top_lp, usr):
    body = {'model': model,
            'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': usr}],
            'max_completion_tokens': 500, 'logprobs': True, 'top_logprobs': top_lp,
            'temperature': 1.0, 'response_format': SCHEMA}
    if model.startswith('openai/gpt-5'):
        body['reasoning'] = {'effort': 'none'}
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY,
                                          'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            lp = r['choices'][0].get('logprobs')
            if lp and lp.get('content'):
                return r
            return None  # no logprobs; treat as failed draw, do not fabricate
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def extract_path(resp):
    """-> (dir_tok, int_tok, frac_tok, obs) where obs maps node-key ->
    {'chosen': (tok, p), 'top': {tok: p}} for the three stochastic nodes."""
    ts = resp['choices'][0]['logprobs']['content']
    dir_i = int_i = dot_i = frac_i = None
    seen = ''
    for i, t in enumerate(ts):
        prev = seen
        seen += t['token']
        if dir_i is None and t['token'] in ('A', 'B') and 'higher' in prev:
            dir_i = i
        if dir_i is not None and int_i is None and 'ratio' in prev and t['token'][:1].isdigit():
            int_i = i
        if int_i is not None and dot_i is None and i > int_i and t['token'] == '.':
            dot_i = i
        if dot_i is not None and frac_i is None and i > dot_i and t['token'][:1].isdigit():
            frac_i = i
    if None in (dir_i, int_i, frac_i):
        return None
    def node(i):
        return {'chosen': (ts[i]['token'], math.exp(ts[i]['logprob'])),
                'top': {x['token']: math.exp(x['logprob']) for x in ts[i].get('top_logprobs', [])}}
    d, ii, ff = ts[dir_i]['token'], ts[int_i]['token'], ts[frac_i]['token']
    return d, ii, ff, {(): node(dir_i), (d,): node(int_i), (d, ii): node(frac_i)}


class NodeStats:
    """Accumulates mass observations per token at one trie node; averages drift."""
    def __init__(self):
        self.obs = {}     # token -> [p, p, ...]
    def add(self, tok, p):
        self.obs.setdefault(tok, []).append(p)
    def add_top(self, top):
        for tok, p in top.items():
            self.add(tok, p)
    def masses(self):
        return {t: sum(v) / len(v) for t, v in self.obs.items()}
    def drift(self):
        """Worst relative spread among tokens with non-negligible mean mass —
        near-zero junk tokens (mask padding) would otherwise dominate."""
        worst = 0.0
        for v in self.obs.values():
            m = sum(v) / len(v)
            if len(v) > 1 and m > 1e-3:
                worst = max(worst, (max(v) - min(v)) / max(v))
        return worst


def zval(direction, r):
    s = 1.0 if direction == 'B' else -1.0
    return s * math.log10(r)


def zrange_cell(direction, int_tok=None):
    """Z-range for an unresolved cell, intersected with the declared domain."""
    s = 1.0 if direction == 'B' else -1.0
    if int_tok is None:
        lo, hi = 0.0, ZMAX
    else:
        i = int(int_tok)
        r_lo, r_hi = max(float(i), DOMAIN_LO), min(i + 0.95, DOMAIN_HI)
        if r_hi < r_lo:            # cell wholly below domain (i == 0)
            r_lo, r_hi = DOMAIN_LO, DOMAIN_HI   # conservative full-domain
        lo, hi = math.log10(r_lo), math.log10(r_hi)
    return (min(s * lo, s * hi), max(s * lo, s * hi))


def ledger(nodes, mass_fn=None):
    """-> dict with exact atoms, cells (unresolved mass with Z-ranges),
    and the conservation report. mass_fn(NodeStats) -> {token: mass}; default
    drift-averaged means. Bootstrap passes a resampling mass_fn."""
    if mass_fn is None:
        mass_fn = lambda ns: ns.masses()
    atoms = {}          # (d, i, f) -> mass
    cells = []          # (mass, zlo, zhi, label)
    dir_m = mass_fn(nodes.get((), NodeStats()))
    for d, p_d in sorted(dir_m.items(), key=lambda kv: -kv[1]):
        if d not in ('A', 'B') or p_d <= 0:
            continue
        int_node = nodes.get((d,))
        if int_node is None:
            zr = zrange_cell(d)
            cells.append((p_d, zr[0], zr[1], 'dir=%s unexpanded' % d))
            continue
        int_m = mass_fn(int_node)
        acc_int = 0.0
        for it, p_i in sorted(int_m.items(), key=lambda kv: -kv[1]):
            if not it.isdigit() or p_i <= 0:
                continue
            acc_int += p_i
            frac_node = nodes.get((d, it))
            if frac_node is None:
                zr = zrange_cell(d, it)
                cells.append((p_d * p_i, zr[0], zr[1], 'd=%s int=%s frac-unresolved' % (d, it)))
                continue
            frac_m = mass_fn(frac_node)
            acc_frac = 0.0
            for ft, p_f in frac_m.items():
                if not ft.isdigit() or p_f <= 0:
                    continue
                acc_frac += p_f
                atoms[(d, it, ft)] = p_d * p_i * p_f
            resid_f = max(0.0, 1.0 - acc_frac)
            if resid_f > 1e-9:
                zr = zrange_cell(d, it)
                cells.append((p_d * p_i * resid_f, zr[0], zr[1], 'd=%s int=%s frac-residual' % (d, it)))
        resid_i = max(0.0, 1.0 - acc_int)
        if resid_i > 1e-9:
            zr = zrange_cell(d)
            cells.append((p_d * resid_i, zr[0], zr[1], 'd=%s int-residual' % d))
    resid_d = max(0.0, 1.0 - sum(p for x, p in dir_m.items() if x in ('A', 'B')))
    if resid_d > 1e-9:
        cells.append((resid_d, -ZMAX, ZMAX, 'dir-residual'))
    return atoms, cells


def certify(atoms, cells):
    head = sum(atoms.values())
    e_head = 0.0
    out_of_domain = 0.0
    for (d, it, ft), p in atoms.items():
        r = int(it) + int(ft) / 10.0
        if r < DOMAIN_LO:
            out_of_domain += p
            continue
        e_head += p * zval(d, r)
    cell_mass = sum(c[0] for c in cells)
    # Conservation gap: mass the ledger cannot attribute (drift-averaged node
    # masses need not sum exactly to 1; over-unity nodes zero their residuals).
    # Soundness demands the gap WIDEN the envelope as full-domain slack — it
    # must never silently vanish.
    gap = abs(1.0 - head - cell_mass)
    lo = (e_head + sum(p * zl for p, zl, zh, _ in cells)
          + (out_of_domain + gap) * (-ZMAX))
    hi = (e_head + sum(p * zh for p, zl, zh, _ in cells)
          + (out_of_domain + gap) * ZMAX)
    mid = e_head + sum(p * (zl + zh) / 2 for p, zl, zh, _ in cells)
    return {'enumerated_mass': head, 'cell_mass': cell_mass,
            'out_of_domain_mass': out_of_domain,
            'conservation_gap': gap,
            'E_Z_lo': lo, 'E_Z_hi': hi, 'width': hi - lo,
            'E_Z_point_midpoint_imputation': mid,
            'E_Z_head_renormalized': (e_head / max(head - out_of_domain, 1e-12))}


def jitter_stats(nodes):
    """Absolute per-token observation spread among tokens with mean mass >1%."""
    spreads = []
    for ns in nodes.values():
        for v in ns.obs.values():
            if len(v) > 1 and sum(v) / len(v) > 0.01:
                spreads.append(max(v) - min(v))
    spreads.sort()
    if not spreads:
        return {'n': 0}
    return {'n': len(spreads),
            'median_abs': spreads[len(spreads) // 2],
            'max_abs': spreads[-1]}


def bootstrap_EZ(nodes, reps=200, seed=7):
    """Provider-noise band: resample each token's mass from its observed
    values (per-node coherent draw index where possible), recompute the
    midpoint-imputation E[Z]. Returns (std, p2.5, p97.5)."""
    import random
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        def mass_fn(ns):
            return {t: rng.choice(v) for t, v in ns.obs.items()}
        atoms, cells = ledger(nodes, mass_fn)
        vals.append(certify(atoms, cells)['E_Z_point_midpoint_imputation'])
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((x - mean) ** 2 for x in vals) / (n - 1)) ** 0.5
    return {'std': std, 'p2_5': vals[int(0.025 * n)], 'p97_5': vals[int(0.975 * n)]}


def run_cell(model, top_lp, draws, pair_key, workers=6):
    a, b, attr = PAIRS[pair_key]
    usr = user_prompt(a, b, attr)
    nodes = {}
    trajectory = []
    n_ok = 0
    lock_results = []
    def one(_):
        return call(model, top_lp, usr)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for resp in ex.map(one, range(draws)):
            if resp is None:
                continue
            got = extract_path(resp)
            if got is None:
                continue
            d, it, ft, obs = got
            n_ok += 1
            for key, o in obs.items():
                ns = nodes.setdefault(key, NodeStats())
                tok, p = o['chosen']
                ns.add(tok, p)
                ns.add_top(o['top'])
            atoms, cells = ledger(nodes)
            cert = certify(atoms, cells)
            trajectory.append({'call': n_ok, 'width': cert['width'],
                               'enumerated': cert['enumerated_mass']})
    atoms, cells = ledger(nodes)
    cert = certify(atoms, cells)
    drift = max((ns.drift() for ns in nodes.values()), default=0.0)
    top_atoms = sorted(atoms.items(), key=lambda kv: -kv[1])[:8]
    return {'model': model, 'pair': pair_key, 'top_logprobs': top_lp,
            'draws_requested': draws, 'draws_ok': n_ok,
            'certificate': cert, 'worst_token_drift': drift,
            'jitter': jitter_stats(nodes),
            'provider_noise_band': bootstrap_EZ(nodes),
            'top_atoms': [{'dir': d, 'ratio': '%s.%s' % (it, ft), 'mass': round(p, 5)}
                          for (d, it, ft), p in top_atoms],
            'largest_cells': sorted([{'mass': round(p, 5), 'z_range': [round(zl, 3), round(zh, 3)],
                                      'label': lab} for p, zl, zh, lab in cells],
                                    key=lambda c: -c['mass'])[:5],
            'trajectory': trajectory}


def main():
    out = []
    for pair_key in PAIRS:
        for model, top_lp, draws in CELLS:
            t0 = time.time()
            res = run_cell(model, top_lp, draws, pair_key)
            res['wall_seconds'] = round(time.time() - t0, 1)
            out.append(res)
            c = res['certificate']
            nb = res['provider_noise_band']
            print('%-22s %-24s draws=%2d/%2d enum=%.3f gap=%.3f width=%.3f E[Z]mid=%.3f noise± %.3f jit=%.3f (%.0fs)'
                  % (pair_key, res['model'], res['draws_ok'], draws,
                     c['enumerated_mass'], c['conservation_gap'], c['width'],
                     c['E_Z_point_midpoint_imputation'], nb['std'],
                     res['jitter'].get('median_abs', 0.0),
                     res['wall_seconds']), flush=True)
    with open('harvest_results.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote harvest_results.json')


if __name__ == '__main__':
    main()
