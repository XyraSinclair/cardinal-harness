use super::*;

// ---------------------------------------------------------------------
//  Diagnostics: HCR / PCR-lite
// ---------------------------------------------------------------------

/// Weighted residual energy fraction: Σλr² / Σλμ².
///
/// This has an exact physical reading. The judgements form a signed edge
/// field (log-ratios) on the comparison graph; the least-squares solve is
/// the Hodge/Helmholtz projection of that field onto gradients of a score
/// potential. What the projection CANNOT absorb — the residual — is the
/// field's curl + harmonic component: irreducible cyclic disagreement
/// (A>B>C>A triads and their higher-order kin; "frustration" in the
/// spin-glass sense). So this ratio is the fraction of judgement energy
/// that is inconsistent BY STRUCTURE, no matter what scores anyone picks:
/// 0 = perfectly transitive judge, 1 = pure noise/cycles. Weighted by the
/// post-Huber λ, so already-downweighted outliers do not double-count.
/// Note: repeat judgements of the SAME pair fuse into one edge before this
/// is computed — intra-pair disagreement shows up in counterbalance
/// diagnostics (order flips, order-residual nats), not here; this measures
/// inter-pair (cyclic) incoherence specifically.
pub(super) fn compute_hcr(mu: &[f64], residuals: &[f64], lam_eff: &[f64], cfg: &Config) -> f64 {
    if mu.is_empty() || lam_eff.is_empty() {
        return 0.0;
    }
    let num: f64 = if residuals.is_empty() {
        0.0
    } else {
        lam_eff
            .iter()
            .zip(residuals.iter())
            .map(|(lam, r)| lam * r * r)
            .sum()
    };
    let den: f64 = lam_eff
        .iter()
        .zip(mu.iter())
        .map(|(lam, m)| lam * m * m)
        .sum::<f64>()
        + cfg.tiny;

    let hcr = num / den;
    hcr.clamp(0.0, 1.0)
}

/// Spectral identifiability diagnostics for a solved comparison graph.
///
/// - `fiedler_value`: smallest nonzero eigenvalue of the weighted graph
///   Laplacian (algebraic connectivity). It lower-bounds how well the
///   score differences are identified: posterior variance along the
///   worst-identified direction scales as 1/fiedler. Zero-adjacent values
///   mean the run's answer hinges on a near-cut edge.
/// - `foster_residual`: |Σ_e w_e·R_eff(e) − (n_touched − components)|.
///   Foster's theorem says the sum is EXACTLY n − c for any weighted
///   graph — a free correctness invariant over the same effective
///   resistances the planner optimizes; a nonzero residual means the
///   linear algebra, not the judge, is broken.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralDiagnostics {
    pub fiedler_value: f64,
    pub foster_residual: f64,
    /// Σ_e w_e·R_eff(e), reported alongside its theorem-exact target.
    pub resistance_sum: f64,
    pub expected_resistance_sum: f64,
    /// Per-edge leverage h_e = w_e·R_eff(e) ∈ `[0,1]` — how much each
    /// judgement determines its own fitted value. Trace identity:
    /// Σ h_e = n − c exactly (Foster), the model's degrees of freedom.
    pub edge_leverage: Vec<f64>,
}

/// Compute spectral diagnostics from fused edge endpoints and weights.
/// Dense eigen-decomposition: intended for graphs up to a few hundred
/// touched vertices (returns `None` above `max_dim` or on empty input).
pub fn spectral_diagnostics(
    endpoints: &[(usize, usize)],
    lam_eff: &[f64],
    n_vertices: usize,
    max_dim: usize,
) -> Option<SpectralDiagnostics> {
    use nalgebra::DMatrix;
    if endpoints.is_empty() || endpoints.len() != lam_eff.len() {
        return None;
    }
    // Remap touched vertices to a compact index.
    let mut map = vec![usize::MAX; n_vertices];
    let mut touched = 0usize;
    for &(i, j) in endpoints {
        for v in [i, j] {
            if map[v] == usize::MAX {
                map[v] = touched;
                touched += 1;
            }
        }
    }
    if touched < 2 || touched > max_dim {
        return None;
    }
    let mut lap = DMatrix::<f64>::zeros(touched, touched);
    for (&(i, j), &w) in endpoints.iter().zip(lam_eff.iter()) {
        let (a, b) = (map[i], map[j]);
        if a == b {
            continue;
        }
        lap[(a, a)] += w;
        lap[(b, b)] += w;
        lap[(a, b)] -= w;
        lap[(b, a)] -= w;
    }
    let eig = SymmetricEigen::new(lap);
    let mut order: Vec<usize> = (0..touched).collect();
    order.sort_by(|&x, &y| {
        eig.eigenvalues[x]
            .partial_cmp(&eig.eigenvalues[y])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let scale = order
        .iter()
        .map(|&k| eig.eigenvalues[k].abs())
        .fold(0.0f64, f64::max)
        .max(1e-300);
    let zero_tol = scale * 1e-9;
    let components = order
        .iter()
        .filter(|&&k| eig.eigenvalues[k].abs() <= zero_tol)
        .count()
        .max(1);
    let fiedler_value = order
        .iter()
        .map(|&k| eig.eigenvalues[k])
        .find(|&v| v > zero_tol)
        .unwrap_or(0.0);

    // Effective resistances via the eigen pseudo-inverse:
    // R(i,j) = Σ_{λ_k > 0} (v_k[i] − v_k[j])² / λ_k.
    let mut resistance_sum = 0.0;
    let mut edge_leverage = Vec::with_capacity(endpoints.len());
    for (&(i, j), &w) in endpoints.iter().zip(lam_eff.iter()) {
        let (a, b) = (map[i], map[j]);
        if a == b {
            edge_leverage.push(0.0);
            continue;
        }
        let mut r = 0.0;
        for &k in &order {
            let lambda = eig.eigenvalues[k];
            if lambda <= zero_tol {
                continue;
            }
            let d = eig.eigenvectors[(a, k)] - eig.eigenvectors[(b, k)];
            r += d * d / lambda;
        }
        resistance_sum += w * r;
        edge_leverage.push((w * r).clamp(0.0, 1.0));
    }
    let expected = (touched - components) as f64;
    Some(SpectralDiagnostics {
        fiedler_value,
        foster_residual: (resistance_sum - expected).abs(),
        resistance_sum,
        expected_resistance_sum: expected,
        edge_leverage,
    })
}

/// Leave-one-out consistency diagnostics: each judgement tested against the
/// prediction of the REST of the graph.
///
/// The sum-over-histories reading of a comparison graph: the fitted value
/// for edge e is the precision-weighted average of every path between its
/// endpoints, and the edge's own measurement should agree with the
/// ensemble of paths that EXCLUDE it. With leverage h_e = λ_e·R_eff(e),
/// the leave-one-out residual is r_e/(1−h_e) and its correctly
/// studentized form is
///
///   z_e = r_e · √λ_e / √(1 − h_e)
///
/// so |z| > 3 flags a judgement the whole rest of the graph disagrees
/// with (an outlier, a corruption, or a genuinely held minority belief —
/// the diagnostic locates the disagreement, it does not adjudicate it).
///
/// Studentization subtlety with teeth (the first implementation failed
/// its own planted test): the IRLS robustifier CRUSHES an outlier's
/// effective weight, so scaling by post-Huber λ_eff hides exactly the
/// judgements the solver quietly downweighted. The diagnostic therefore
/// scales by the judgement's CLAIMED precision (raw λ) against a robust
/// MAD estimate of the weighted residual scale — it must see what the
/// robustifier saw, not what it left behind. Bridges (h_e ≈ 1) carry no
/// cross-check — removing them disconnects the graph — and are counted,
/// not scored: a judgement only one edge supports is unaudited, which is
/// itself worth knowing.
#[derive(Debug, Clone, PartialEq)]
pub struct LooDiagnostics {
    /// Studentized leave-one-out residual per edge (None for bridges).
    pub z: Vec<Option<f64>>,
    /// Largest |z| observed (0 when none scoreable).
    pub max_abs_z: f64,
    /// Edge indices with |z| > 3: the judgements the rest of the graph
    /// votes against.
    pub flagged: Vec<usize>,
    /// Edges with leverage ≈ 1: unauditable single-path judgements.
    pub bridges: usize,
    /// Robust scale (1.4826·MAD) of the √λ-weighted residuals.
    pub sigma_hat: f64,
}

pub(super) fn compute_loo(residuals: &[f64], lam_raw: &[f64], leverage: &[f64]) -> LooDiagnostics {
    // Robust scale of weighted residuals, immune to the very outliers
    // this diagnostic exists to find.
    let mut weighted: Vec<f64> = residuals
        .iter()
        .zip(lam_raw.iter())
        .map(|(&r, &lam)| (r * lam.max(0.0).sqrt()).abs())
        .collect();
    weighted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if weighted.is_empty() {
        0.0
    } else {
        weighted[weighted.len() / 2]
    };
    let sigma_hat = (1.4826 * mad).max(1e-12);

    let mut z = Vec::with_capacity(residuals.len());
    let mut max_abs_z = 0.0f64;
    let mut flagged = Vec::new();
    let mut bridges = 0usize;
    for (k, ((&r, &lam), &h)) in residuals
        .iter()
        .zip(lam_raw.iter())
        .zip(leverage.iter())
        .enumerate()
    {
        if h > 1.0 - 1e-9 {
            bridges += 1;
            z.push(None);
            continue;
        }
        let zk = r * lam.max(0.0).sqrt() / (sigma_hat * (1.0 - h).sqrt());
        if zk.abs() > max_abs_z {
            max_abs_z = zk.abs();
        }
        if zk.abs() > 3.0 {
            flagged.push(k);
        }
        z.push(Some(zk));
    }
    LooDiagnostics {
        z,
        max_abs_z,
        flagged,
        bridges,
        sigma_hat,
    }
}

/// The full combinatorial Hodge split of the cyclic residual.
///
/// The solve projects the observed edge field μ onto gradients; the
/// residual r lives in the cycle space (dimension |E| − |V| + components).
/// That cyclic energy splits further, w-orthogonally (w = λ_eff):
///
///   cycle space = im(curl*) ⊕ H
///
/// where curl is taken over the FILLED triangles (3-cliques whose three
/// edges were all judged). `im(curl*)` is locally cyclic disagreement — a
/// per-triad audit (A>B>C>A) can catch it. `H` (harmonic) is
/// divergence-free AND curl-free on every filled triangle yet nonzero: a
/// cycle longer than a triangle whose closing chords were never elicited.
/// No triad spot-check can see it, by construction. Sparse (cheap)
/// comparison graphs are exactly the graphs whose cycles are mostly long —
/// elicitation efficiency and triad-auditability are in tension, and this
/// diagnostic measures which kind of frustration a run actually has.
///
/// Invariant (Pythagoras in the w-inner product):
/// `local_curl_frac + harmonic_frac == hcr` up to the same clamping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HodgeSplit {
    /// Share of judgement energy in triangle-supported (auditable) curl.
    pub local_curl_frac: f64,
    /// Share of judgement energy in harmonic (triad-invisible) cycles.
    pub harmonic_frac: f64,
    /// Number of filled triangles found in the fused edge set.
    pub filled_triangles: usize,
    /// Dimension of the harmonic space: cycle_dim − rank(curl on filled
    /// triangles). Zero means the graph DESIGN cannot host harmonic
    /// disagreement — report it like a denominator.
    pub harmonic_dim: usize,
}

/// Compute the Hodge split from the fused edges and solver residuals.
/// Pure function of the same inputs `compute_hcr` consumes, plus the edge
/// endpoints. See [`HodgeSplit`].
pub fn compute_hodge_split(
    endpoints: &[(usize, usize)],
    mu: &[f64],
    residuals: &[f64],
    lam_eff: &[f64],
    n_vertices: usize,
    cfg: &Config,
) -> HodgeSplit {
    let m = endpoints.len();
    let empty = HodgeSplit {
        local_curl_frac: 0.0,
        harmonic_frac: 0.0,
        filled_triangles: 0,
        harmonic_dim: 0,
    };
    if m == 0 || residuals.len() != m || lam_eff.len() != m || mu.len() != m {
        return empty;
    }

    // Cycle dimension: |E| − |V| + components, on the vertices present.
    let components = {
        let mut parent: Vec<usize> = (0..n_vertices).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut root = x;
            while parent[root] != root {
                root = parent[root];
            }
            let mut cur = x;
            while parent[cur] != root {
                let next = parent[cur];
                parent[cur] = root;
                cur = next;
            }
            root
        }
        let mut touched = vec![false; n_vertices];
        for &(i, j) in endpoints {
            touched[i] = true;
            touched[j] = true;
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri != rj {
                parent[ri] = rj;
            }
        }
        (0..n_vertices)
            .filter(|&v| touched[v] && find(&mut parent, v) == v)
            .count()
    };
    let touched_vertices = {
        let mut seen = vec![false; n_vertices];
        for &(i, j) in endpoints {
            seen[i] = true;
            seen[j] = true;
        }
        seen.iter().filter(|&&b| b).count()
    };
    let cycle_dim = (m + components).saturating_sub(touched_vertices);

    let den: f64 = lam_eff
        .iter()
        .zip(mu.iter())
        .map(|(lam, x)| lam * x * x)
        .sum::<f64>()
        + cfg.tiny;
    if cycle_dim == 0 {
        return empty;
    }

    // Enumerate filled triangles over the fused edge set.
    use std::collections::HashMap as Map;
    let mut edge_index: Map<(usize, usize), usize> = Map::new();
    for (k, &(i, j)) in endpoints.iter().enumerate() {
        let key = (i.min(j), i.max(j));
        edge_index.entry(key).or_insert(k);
    }
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for &(i, j) in edge_index.keys() {
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    for list in &mut adjacency {
        list.sort_unstable();
        list.dedup();
    }
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    for a in 0..n_vertices {
        for &b in adjacency[a].iter().filter(|&&b| b > a) {
            for &c in adjacency[b].iter().filter(|&&c| c > b) {
                if edge_index.contains_key(&(a, c)) {
                    triangles.push([a, b, c]);
                }
            }
        }
    }
    let t = triangles.len();

    // Orientation sign of edge (u, v) relative to canonical (min, max):
    // the residual r_e is the flow along i→j of the stored edge; we treat
    // the stored orientation as canonical and give the triangle boundary
    // signs relative to it.
    let signed = |u: usize, v: usize| -> (usize, f64) {
        let key = (u.min(v), u.max(v));
        let k = edge_index[&key];
        let (ei, _ej) = endpoints[k];
        let sign = if ei == u { 1.0 } else { -1.0 };
        (k, sign)
    };

    // C: t × m sparse rows of (edge, sign) triples for cycle a→b→c→a.
    let rows: Vec<[(usize, f64); 3]> = triangles
        .iter()
        .map(|&[a, b, c]| [signed(a, b), signed(b, c), signed(c, a)])
        .collect();

    if t == 0 {
        let num: f64 = lam_eff
            .iter()
            .zip(residuals.iter())
            .map(|(lam, r)| lam * r * r)
            .sum();
        return HodgeSplit {
            local_curl_frac: 0.0,
            harmonic_frac: (num / den).clamp(0.0, 1.0),
            filled_triangles: 0,
            harmonic_dim: cycle_dim,
        };
    }

    // Normal equations (C W⁻¹ Cᵀ) z = C r ; projection = W⁻¹ Cᵀ z.
    // Dense t × t assembly; t is small on real comparison graphs. The
    // system is consistent by construction; a pivoted Gauss with zero-pivot
    // skip handles rank deficiency (dependent triangles) exactly.
    let w_inv: Vec<f64> = lam_eff.iter().map(|&l| 1.0 / l.max(cfg.tiny)).collect();
    let mut a = vec![vec![0.0f64; t + 1]; t];
    for (p, row_p) in rows.iter().enumerate() {
        for (q, row_q) in rows.iter().enumerate().skip(p) {
            let mut acc = 0.0;
            for &(e1, s1) in row_p {
                for &(e2, s2) in row_q {
                    if e1 == e2 {
                        acc += s1 * s2 * w_inv[e1];
                    }
                }
            }
            a[p][q] = acc;
            if p != q {
                a[q][p] = acc;
            }
        }
        let rhs: f64 = row_p.iter().map(|&(e, sgn)| sgn * residuals[e]).sum();
        a[p][t] = rhs;
    }
    // Pivoted Gaussian elimination with zero-pivot skip.
    let scale: f64 = a
        .iter()
        .map(|row| row[..t].iter().map(|x| x.abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max)
        .max(cfg.tiny);
    let tol = scale * 1e-12;
    let mut z = vec![0.0f64; t];
    let mut pivot_row = 0usize;
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    for col in 0..t {
        let Some(best) =
            (pivot_row..t).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
        else {
            break;
        };
        if a[best][col].abs() <= tol {
            continue;
        }
        a.swap(pivot_row, best);
        let pv = a[pivot_row][col];
        for r in 0..t {
            if r != pivot_row && a[r][col].abs() > 0.0 {
                let f = a[r][col] / pv;
                let (head, tail) = a.split_at_mut(pivot_row.max(r));
                let (row_r, row_p) = if r < pivot_row {
                    (&mut head[r], &tail[0])
                } else {
                    (&mut tail[0], &head[pivot_row])
                };
                for (x, y) in row_r[col..=t].iter_mut().zip(row_p[col..=t].iter()) {
                    *x -= f * y;
                }
            }
        }
        pivots.push((pivot_row, col));
        pivot_row += 1;
    }
    let curl_rank = pivots.len();
    for &(r, c) in &pivots {
        z[c] = a[r][t] / a[r][c];
    }

    // local = W⁻¹ Cᵀ z (in canonical edge orientation).
    let mut local = vec![0.0f64; m];
    for (p, row_p) in rows.iter().enumerate() {
        for &(e, sgn) in row_p {
            local[e] += w_inv[e] * sgn * z[p];
        }
    }
    let local_energy: f64 = lam_eff
        .iter()
        .zip(local.iter())
        .map(|(lam, x)| lam * x * x)
        .sum();
    let harmonic_energy: f64 = lam_eff
        .iter()
        .zip(residuals.iter().zip(local.iter()))
        .map(|(lam, (r, l))| lam * (r - l) * (r - l))
        .sum();

    HodgeSplit {
        local_curl_frac: (local_energy / den).clamp(0.0, 1.0),
        harmonic_frac: (harmonic_energy / den).clamp(0.0, 1.0),
        filled_triangles: t,
        harmonic_dim: cycle_dim.saturating_sub(curl_rank),
    }
}

pub(super) fn probe_seed(seed: u64, edges: &[Edge], lam_eff: &[f64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    edges.len().hash(&mut hasher);
    for (k, e) in edges.iter().enumerate() {
        e.i.hash(&mut hasher);
        e.j.hash(&mut hasher);
        e.mu.to_bits().hash(&mut hasher);
        e.lam.to_bits().hash(&mut hasher);
        if k < lam_eff.len() {
            lam_eff[k].to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

pub(super) fn compute_pcr_lite(
    mu: &[f64],
    residuals: &[f64],
    lam_eff: &[f64],
    cfg: &Config,
) -> f64 {
    if mu.is_empty() || residuals.is_empty() || lam_eff.is_empty() {
        return 0.0;
    }
    let w_sum: f64 = lam_eff.iter().sum::<f64>() + cfg.tiny;
    let mse_resid: f64 = lam_eff
        .iter()
        .zip(residuals.iter())
        .map(|(lam, r)| lam * r * r)
        .sum::<f64>()
        / w_sum;
    let mean_mu = mu.iter().sum::<f64>() / (mu.len() as f64);
    let var_signal: f64 = mu
        .iter()
        .map(|v| {
            let d = v - mean_mu;
            d * d
        })
        .sum::<f64>()
        / (mu.len() as f64)
        + cfg.tiny;

    let pcr = 1.0 - mse_resid / var_signal;
    pcr.clamp(0.0, 1.0)
}
