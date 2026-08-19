//! E2 — grok gauge calibration on anchors (llmsorting PROGRAM.md §2/§3).
//!
//! The grok gauge claims a judge "groks" a transitive attribute when its
//! pairwise judgements behave like noisy readings of one latent scalar. Its
//! band thresholds (grokked / partial / not) are conventions until they are
//! checked somewhere truth is known. This example runs the gauge's readouts
//! on ANCHOR pools — countries by population, rivers by length, cities by
//! metro population — where the true latent (log of the true value) is
//! external, so gauge verdicts can be scored against truth.
//!
//! Protocol per (pool, model): three runs of the CANONICAL pairwise path
//! (`sort_documents`, canonical_v2, counterbalanced — no forked prompts):
//! the attribute wording at the default 4·n budget, a paraphrase wording at
//! 2·n, and a negated wording at 2·n. Readouts, each with denominators:
//! order agreement + mean |Δ ln r| over swapped pairs (from the trace),
//! Hodge cyclic residual fraction (re-solving the traced observations
//! through the production IRLS engine), polarity and paraphrase Spearman
//! across runs, signal (latent spread vs mean posterior std), and the truth
//! axis: Spearman of fitted latents vs true log values plus the calibration
//! slope of fitted latent differences on true log-ratio differences.
//!
//! Bands (PROGRAM.md §2 provisional): grokked = order agreement ≥ 0.90 AND
//! curl (hcr) ≤ 0.10 AND polarity ≤ −0.80 AND paraphrase ≥ 0.80. The WST
//! criterion is unmeasured here (no repeat draws in this budget) and is
//! reported as such. Operationalization of the unnamed lower bands: 4
//! criteria pass = grokked, 2–3 = partial, ≤ 1 = not.
//!
//! Dry run ($0):  cargo run --release --example anchor_gauge -- --offline
//! Live (capped): xyra-vault run repos/documents/openpriors/env/env -- \
//!     cargo run --release --example anchor_gauge

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use serde::Serialize;

use llmsort::gateway::{
    Attribution, ChatGateway, ChatRequest, ChatResponse, FinishReason, NoopUsageSink,
    ProviderError, ProviderGateway, Role,
};
use llmsort::rating_engine::{AttributeParams, EngineSpec, Observation, RaterParams, RatingEngine};
use llmsort::rerank::sort::{sort_documents, SortOptions, SortedTexts};
use llmsort::rerank::types::RerankDocument;
use llmsort::rerank::{ComparisonTrace, JsonlTraceSink, RerankExecution, RerankRunOptions};

/// canonical_v2 ratio ladder ceiling: the synthetic judge saturates exactly
/// where the live instrument's answer vocabulary does.
const SYNTHETIC_RATIO_CEILING: f64 = 26.0;
/// Synthetic pricing (per-token nanodollars) so the offline path exercises
/// the cost accounting: gpt-4.1-mini list price $0.40/M in, $1.60/M out.
const SYNTH_ND_PER_INPUT_TOKEN: i64 = 400;
const SYNTH_ND_PER_OUTPUT_TOKEN: i64 = 1600;

#[derive(Parser, Debug)]
#[command(
    name = "anchor_gauge",
    about = "Grok-gauge readouts on anchor pools with known true values (E2)"
)]
struct Args {
    /// Deterministic synthetic judge (truth + lognormal noise); no network, $0.
    #[arg(long)]
    offline: bool,
    /// Comma-separated OpenRouter model slugs (offline uses only the first).
    #[arg(long, default_value = "openai/gpt-4.1-mini,openai/gpt-5.4-nano")]
    models: String,
    /// Seed for the pairwise planner (per-run seeds are derived from it).
    #[arg(long, default_value_t = 17)]
    seed: u64,
    /// Output directory for report.json / trace.jsonl.
    #[arg(long, default_value = "artifacts/live/anchor-gauge-2026-08-15")]
    out_dir: PathBuf,
    /// Hard live-spend cap (USD) across all runs; the protocol aborts above
    /// it (checked at run granularity — the sort path owns the call loop).
    #[arg(long, default_value_t = 5.0)]
    spend_cap_usd: f64,
    /// Offline judge's lognormal percept noise (σ in ln space). The default
    /// makes the offline arm a PLUMBING check: a low-noise judge must
    /// recover planted truth decisively or the pipeline is broken. At
    /// σ = 0.35 (E1's figure) the close-packed rivers pool is genuinely
    /// noise-dominated at this budget — a power fact, not a bug.
    #[arg(long, default_value_t = 0.15)]
    synthetic_sigma: f64,
}

// ---------------------------------------------------------------------
//  Anchor pools: entities with externally known true values
// ---------------------------------------------------------------------

struct Pool {
    slug: &'static str,
    /// The gauge's primary attribute wording.
    attribute: &'static str,
    /// Paraphrase wording (target Spearman ≈ +1 vs attribute).
    paraphrase: &'static str,
    /// Negated wording (target Spearman ≈ −1 vs attribute).
    negation: &'static str,
    truth_source: &'static str,
    /// (entity text, true value). Entity text is the bare name
    /// (+ country for cities), exactly what the judge sees.
    entities: &'static [(&'static str, f64)],
}

/// Countries by population, 2024 estimates (UN World Population Prospects
/// 2024, persons).
const COUNTRIES: &[(&str, f64)] = &[
    ("China", 1_419_321_278.0),
    ("India", 1_450_935_791.0),
    ("United States", 345_426_571.0),
    ("Indonesia", 283_487_931.0),
    ("Brazil", 211_998_573.0),
    ("Nigeria", 232_679_478.0),
    ("Japan", 123_753_041.0),
    ("Mexico", 130_861_007.0),
    ("Germany", 84_552_242.0),
    ("France", 66_548_530.0),
    ("United Kingdom", 69_138_192.0),
    ("Italy", 59_342_867.0),
    ("Kenya", 56_432_944.0),
    ("Australia", 26_713_205.0),
    ("Netherlands", 18_228_742.0),
    ("New Zealand", 5_213_944.0),
];

/// Rivers by length in km (commonly cited figures, Wikipedia "List of rivers
/// by length" 2024; Amazon length is contested — 6,400 km used here;
/// Mississippi/Yellow/Ob figures are for the full river systems).
const RIVERS: &[(&str, f64)] = &[
    ("Nile", 6_650.0),
    ("Amazon", 6_400.0),
    ("Yangtze", 6_300.0),
    ("Mississippi", 6_275.0),
    ("Yenisei", 5_539.0),
    ("Yellow", 5_464.0),
    ("Ob", 5_410.0),
    ("Paraná", 4_880.0),
    ("Congo", 4_700.0),
    ("Amur", 4_444.0),
    ("Lena", 4_400.0),
    ("Mekong", 4_350.0),
    ("Mackenzie", 4_241.0),
    ("Niger", 4_200.0),
    ("Danube", 2_850.0),
    ("Rhine", 1_230.0),
];

/// Cities by metro-area population (UN World Urbanization Prospects urban
/// agglomeration, 2024 projections, persons — note the agglomeration
/// definition runs low vs "metro area" for London/Paris/Chicago/Vienna;
/// this definitional fuzz is a stated confound of the cities pool).
const CITIES: &[(&str, f64)] = &[
    ("Tokyo, Japan", 37_000_000.0),
    ("Delhi, India", 33_800_000.0),
    ("Shanghai, China", 29_900_000.0),
    ("São Paulo, Brazil", 22_800_000.0),
    ("Mexico City, Mexico", 22_500_000.0),
    ("Cairo, Egypt", 22_600_000.0),
    ("Mumbai, India", 21_700_000.0),
    ("Beijing, China", 22_200_000.0),
    ("Osaka, Japan", 19_000_000.0),
    ("New York, United States", 18_900_000.0),
    ("Karachi, Pakistan", 17_600_000.0),
    ("Istanbul, Turkey", 16_000_000.0),
    ("London, United Kingdom", 9_700_000.0),
    ("Paris, France", 11_300_000.0),
    ("Chicago, United States", 9_000_000.0),
    ("Vienna, Austria", 2_000_000.0),
];

fn anchor_pools() -> Vec<Pool> {
    vec![
        Pool {
            slug: "countries",
            attribute: "population",
            paraphrase: "number of people living there",
            negation: "smallness of population",
            truth_source: "UN World Population Prospects 2024 (persons, 2024 estimates)",
            entities: COUNTRIES,
        },
        Pool {
            slug: "rivers",
            attribute: "length in kilometres",
            paraphrase: "how long the river runs",
            negation: "shortness of the river",
            truth_source: "commonly cited lengths, Wikipedia list of rivers by length (km, 2024)",
            entities: RIVERS,
        },
        Pool {
            slug: "cities",
            attribute: "metro-area population",
            paraphrase: "how many people the metropolitan area holds",
            negation: "smallness of metro-area population",
            truth_source: "UN World Urbanization Prospects urban agglomeration 2024 (persons)",
            entities: CITIES,
        },
    ]
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Wording {
    Attribute,
    Paraphrase,
    Negation,
}

impl Wording {
    const ALL: [Wording; 3] = [Wording::Attribute, Wording::Paraphrase, Wording::Negation];

    fn slug(self) -> &'static str {
        match self {
            Wording::Attribute => "attribute",
            Wording::Paraphrase => "paraphrase",
            Wording::Negation => "negation",
        }
    }

    fn text(self, pool: &Pool) -> &'static str {
        match self {
            Wording::Attribute => pool.attribute,
            Wording::Paraphrase => pool.paraphrase,
            Wording::Negation => pool.negation,
        }
    }

    /// Comparison budget: the primary wording rides the sort path's default
    /// (4·n); paraphrase and negation get 2·n each.
    fn budget(self, n: usize) -> Option<usize> {
        match self {
            Wording::Attribute => None,
            Wording::Paraphrase | Wording::Negation => Some(2 * n),
        }
    }
}

// ---------------------------------------------------------------------
//  Deterministic synthetic judge (offline arm)
// ---------------------------------------------------------------------

/// Same escaping as `src/prompts.rs`: the judge sees rendered bytes.
fn escape_xml_chars(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Answers canonical_v2 pairwise prompts from the planted truth: latents are
/// the true log values, percept noise is lognormal (Gaussian in ln space,
/// σ = `--synthetic-sigma`), deterministic per exact prompt bytes. The
/// negated wording answers with the sign flipped — the property the offline
/// validation must recover.
struct SyntheticJudge {
    /// (escaped wording text, pool index, sign), longest wording first so
    /// that substring wordings ("population" ⊂ "smallness of population")
    /// can never shadow a longer match.
    wordings: Vec<(String, usize, f64)>,
    /// Escaped entity text -> (pool index, entity index).
    entities: HashMap<String, (usize, usize)>,
    /// Per pool: ln(true value) per entity.
    z: Vec<Vec<f64>>,
    seed: u64,
    sigma: f64,
}

impl SyntheticJudge {
    fn new(pools: &[Pool], seed: u64, sigma: f64) -> Self {
        let mut wordings = Vec::new();
        let mut entities = HashMap::new();
        let mut z = Vec::new();
        for (pool_idx, pool) in pools.iter().enumerate() {
            for (wording, sign) in [
                (Wording::Attribute, 1.0),
                (Wording::Paraphrase, 1.0),
                (Wording::Negation, -1.0),
            ] {
                wordings.push((escape_xml_chars(wording.text(pool)), pool_idx, sign));
            }
            for (entity_idx, (name, _)) in pool.entities.iter().enumerate() {
                let prior = entities.insert(escape_xml_chars(name), (pool_idx, entity_idx));
                assert!(prior.is_none(), "entity names must be globally unique");
            }
            z.push(pool.entities.iter().map(|(_, v)| v.ln()).collect());
        }
        wordings.sort_by_key(|(text, _, _)| std::cmp::Reverse(text.len()));
        Self {
            wordings,
            entities,
            z,
            seed,
            sigma,
        }
    }

    /// Deterministic standard-normal draw from the exact prompt bytes.
    fn noise(&self, parts: &[&str]) -> f64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed.to_le_bytes());
        for p in parts {
            hasher.update(&(p.len() as u64).to_le_bytes());
            hasher.update(p.as_bytes());
        }
        let bytes = hasher.finalize();
        let b = bytes.as_bytes();
        let u1 = (u64::from_le_bytes(b[0..8].try_into().expect("8 bytes")) >> 11) as f64
            / (1u64 << 53) as f64;
        let u2 = (u64::from_le_bytes(b[8..16].try_into().expect("8 bytes")) >> 11) as f64
            / (1u64 << 53) as f64;
        let u1 = u1.max(1e-12);
        self.sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn entity_between(&self, user: &str, open: &str, close: &str) -> Result<usize, ProviderError> {
        let start = user
            .find(open)
            .ok_or_else(|| ProviderError::invalid_request(format!("missing {open}")))?
            + open.len();
        let end = user[start..]
            .find(close)
            .ok_or_else(|| ProviderError::invalid_request(format!("missing {close}")))?
            + start;
        let text = user[start..end].trim();
        self.entities
            .get(text)
            .map(|&(_, entity_idx)| entity_idx)
            .ok_or_else(|| ProviderError::invalid_request("synthetic judge: unknown entity text"))
    }
}

#[async_trait::async_trait]
impl ChatGateway for SyntheticJudge {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let system = req
            .messages
            .iter()
            .find(|m| matches!(m.role, Role::System))
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let user = req
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let (pool_idx, sign) = self
            .wordings
            .iter()
            .find(|(text, _, _)| user.contains(text.as_str()))
            .map(|&(_, pool_idx, sign)| (pool_idx, sign))
            .ok_or_else(|| ProviderError::invalid_request("synthetic judge: unknown wording"))?;
        let z = &self.z[pool_idx];
        let a = self.entity_between(&user, "<entity_A_context>\n", "\n</entity_A_context>")?;
        let b = self.entity_between(&user, "<entity_B_context>\n", "\n</entity_B_context>")?;

        let delta = sign * (z[a] - z[b]) + self.noise(&[&user]);
        let (higher, magnitude) = if delta >= 0.0 {
            ("A", delta)
        } else {
            ("B", -delta)
        };
        let ratio = magnitude.exp().clamp(1.0, SYNTHETIC_RATIO_CEILING);
        let content = serde_json::json!({
            "higher_ranked": higher,
            "ratio": (ratio * 100.0).round() / 100.0,
            "confidence": 0.8,
        })
        .to_string();

        let input_tokens = ((system.len() + user.len()) / 4) as u32;
        let output_tokens = (content.len() / 4) as u32;
        let cost = i64::from(input_tokens) * SYNTH_ND_PER_INPUT_TOKEN
            + i64::from(output_tokens) * SYNTH_ND_PER_OUTPUT_TOKEN;
        Ok(ChatResponse {
            provider_call_id: None,
            provider_request_id: None,
            served_model: Some("synthetic/offline-judge".to_string()),
            content,
            reasoning: None,
            reasoning_tokens: None,
            input_tokens,
            output_tokens,
            cost_nanodollars: cost,
            cost_is_estimate: true,
            upstream_cost_nanodollars: None,
            latency: std::time::Duration::from_millis(0),
            finish_reason: FinishReason::Stop,
            output_logprobs: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
        })
    }
}

// ---------------------------------------------------------------------
//  Rank / regression metrics (n is tiny; direct implementations)
// ---------------------------------------------------------------------

fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).expect("finite latents"));
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &item in &idx[i..=j] {
            out[item] = avg;
        }
        i = j + 1;
    }
    out
}

fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va.sqrt() * vb.sqrt()).max(f64::MIN_POSITIVE)
}

/// Regression THROUGH THE ORIGIN of fitted latent differences on true
/// log-ratio differences over all unordered pairs: slope 1.0 = magnitude
/// calibrated; also the Pearson r² of the differences and the pair count.
fn pairwise_diff_fit(s: &[f64], t: &[f64]) -> (f64, f64, usize) {
    let n = s.len();
    let (mut stt, mut sst, mut sss) = (0.0f64, 0.0f64, 0.0f64);
    let mut pairs = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let ds = s[i] - s[j];
            let dt = t[i] - t[j];
            stt += dt * dt;
            sst += ds * dt;
            sss += ds * ds;
            pairs += 1;
        }
    }
    let slope = sst / stt.max(f64::MIN_POSITIVE);
    let r2 = (sst * sst) / (sss * stt).max(f64::MIN_POSITIVE);
    (slope, r2, pairs)
}

fn population_sd(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

// ---------------------------------------------------------------------
//  Per-run readouts from the comparison trace
// ---------------------------------------------------------------------

#[derive(Serialize, Clone, Copy)]
struct OrderStats {
    /// Unordered pairs observed in BOTH presentation orders.
    pairs_with_both_orientations: usize,
    /// Of those, pairs whose two orientations agree on direction
    /// (per-orientation mean canonical log-ratios share a sign).
    agreements: usize,
    agreement_frac: Option<f64>,
    /// Mean |Δ ln r| between the two orientations' mean canonical
    /// log-ratios, over the same pairs.
    mean_abs_delta_ln_r: Option<f64>,
}

/// Group traced solver observations by unordered pair and presentation
/// orientation. `solver_observation` is already in canonical (unswapped)
/// entity coordinates — replay contract, see examples/replay_trace.rs —
/// so the swapped/unswapped split isolates pure presentation effects.
fn order_stats(rows: &[ComparisonTrace]) -> OrderStats {
    let mut by_pair: HashMap<(usize, usize), (Vec<f64>, Vec<f64>)> = HashMap::new();
    for row in rows {
        let Some(obs) = row.solver_observation.as_ref() else {
            continue;
        };
        let (lo, hi) = (obs.i.min(obs.j), obs.i.max(obs.j));
        let m_lo_hi = if obs.i == lo {
            obs.ratio.ln()
        } else {
            -obs.ratio.ln()
        };
        let slot = by_pair.entry((lo, hi)).or_default();
        if row.swapped {
            slot.1.push(m_lo_hi);
        } else {
            slot.0.push(m_lo_hi);
        }
    }
    let mut pairs = 0usize;
    let mut agreements = 0usize;
    let mut deltas: Vec<f64> = Vec::new();
    for (unswapped, swapped) in by_pair.values() {
        if unswapped.is_empty() || swapped.is_empty() {
            continue;
        }
        let mean_u = unswapped.iter().sum::<f64>() / unswapped.len() as f64;
        let mean_s = swapped.iter().sum::<f64>() / swapped.len() as f64;
        pairs += 1;
        if mean_u * mean_s > 0.0 {
            agreements += 1;
        }
        deltas.push((mean_u - mean_s).abs());
    }
    OrderStats {
        pairs_with_both_orientations: pairs,
        agreements,
        agreement_frac: (pairs > 0).then(|| agreements as f64 / pairs as f64),
        mean_abs_delta_ln_r: (!deltas.is_empty())
            .then(|| deltas.iter().sum::<f64>() / deltas.len() as f64),
    }
}

#[derive(Serialize, Clone)]
struct SolverDiag {
    /// Traced solver observations re-fed to the production IRLS engine.
    observations: usize,
    /// Cyclic residual fraction of judgement energy (the gauge's "curl").
    hcr: f64,
    /// Hodge split of hcr: triangle-auditable vs triad-invisible cycles.
    local_curl_frac: f64,
    harmonic_frac: f64,
    filled_triangles: usize,
    harmonic_dim: usize,
    cycle_dim: usize,
    /// Spearman between the re-solved scores and the sort path's latents;
    /// ≈ 1.0 certifies the replay seam round-trips.
    resolve_vs_sort_spearman: f64,
}

/// Re-solve the traced observations through the production engine to reach
/// the Hodge/curl diagnostics the sort surface does not expose.
fn resolve_diagnostics(
    n: usize,
    rows: &[ComparisonTrace],
    sort_latents: &[f64],
    engine_spec_slot: &mut Option<EngineSpec>,
) -> Result<SolverDiag, Box<dyn std::error::Error>> {
    let obs: Vec<Observation> = rows
        .iter()
        .filter_map(|row| row.solver_observation.clone())
        .collect();
    let mut raters: HashMap<String, RaterParams> = HashMap::new();
    for o in &obs {
        raters.entry(o.rater_id.clone()).or_default();
    }
    if raters.is_empty() {
        raters.insert("unused".to_string(), RaterParams::default());
    }
    let mut engine = RatingEngine::new(n, AttributeParams::default(), raters, None)?;
    if engine_spec_slot.is_none() {
        *engine_spec_slot = Some(engine.spec());
    }
    engine.add_observations(&obs);
    let summary = engine.solve();
    Ok(SolverDiag {
        observations: obs.len(),
        hcr: summary.hcr,
        local_curl_frac: summary.hodge.local_curl_frac,
        harmonic_frac: summary.hodge.harmonic_frac,
        filled_triangles: summary.hodge.filled_triangles,
        harmonic_dim: summary.hodge.harmonic_dim,
        cycle_dim: summary.cycle_dim,
        resolve_vs_sort_spearman: spearman_rho(&summary.scores, sort_latents),
    })
}

// ---------------------------------------------------------------------
//  Report shapes
// ---------------------------------------------------------------------

#[derive(Serialize, Clone)]
struct ItemLatent {
    id: String,
    mean: f64,
    std: f64,
}

#[derive(Serialize)]
struct RunReadout {
    pool: String,
    model: String,
    wording: String,
    criterion: String,
    rng_seed: u64,
    comparison_budget: usize,
    comparisons_attempted: usize,
    comparisons_used: usize,
    comparisons_refused: usize,
    pairs_counterbalanced: usize,
    position_flips: usize,
    stop_reason: serde_json::Value,
    provider_input_tokens: u32,
    provider_output_tokens: u32,
    cost_nanodollars: i64,
    latency_ms: u128,
    latents: Vec<ItemLatent>,
    order: OrderStats,
    solver: SolverDiag,
}

#[derive(Serialize)]
struct BandCriterion {
    name: &'static str,
    value: Option<f64>,
    pass: bool,
}

#[derive(Serialize)]
struct GaugeReadout {
    pool: String,
    model: String,
    n_entities: usize,
    // Order invariance (attribute run).
    order_agreement: Option<f64>,
    order_pairs: usize,
    mean_abs_delta_ln_r: Option<f64>,
    // Multiplicative closure (attribute run).
    curl_hcr: f64,
    curl_observations: usize,
    filled_triangles: usize,
    harmonic_dim: usize,
    // Relation axes across runs (n = entities in both solves).
    polarity_spearman: f64,
    paraphrase_spearman: f64,
    relation_n: usize,
    // Signal (attribute run).
    signal_latent_sd: f64,
    signal_mean_posterior_std: f64,
    signal_ratio: f64,
    // Truth axis (anchors only).
    truth_spearman: f64,
    truth_slope: f64,
    truth_r2: f64,
    truth_pairs: usize,
    truth_spearman_paraphrase: f64,
    truth_spearman_negation: f64,
    truth_slope_negation: f64,
    // Verdict under the provisional thresholds.
    band: &'static str,
    band_passes: usize,
    band_criteria: Vec<BandCriterion>,
}

#[derive(Serialize)]
struct PoolReport {
    slug: &'static str,
    attribute: &'static str,
    paraphrase: &'static str,
    negation: &'static str,
    truth_source: &'static str,
    entities: Vec<PoolEntity>,
}

#[derive(Serialize)]
struct PoolEntity {
    name: &'static str,
    true_value: f64,
    ln_true: f64,
}

#[derive(Serialize)]
struct Report {
    generated_at: String,
    offline: bool,
    models: Vec<String>,
    seed: u64,
    spend_cap_usd: f64,
    band_rule: &'static str,
    pools: Vec<PoolReport>,
    engine_spec: EngineSpec,
    runs: Vec<RunReadout>,
    gauges: Vec<GaugeReadout>,
    total_cost_nanodollars: i64,
    caveats: Vec<String>,
}

const BAND_RULE: &str = "PROGRAM.md §2 provisional thresholds: grokked = order agreement >= 0.90 \
    AND curl (hcr) <= 0.10 AND polarity <= -0.80 AND paraphrase >= 0.80. The WST criterion (zero \
    violations beyond 2 SE) is UNMEASURED here: this protocol has no repeat draws. \
    Operationalization of the unnamed lower bands: 4 criteria pass = grokked, 2-3 = partial, \
    <= 1 = not.";

// ---------------------------------------------------------------------
//  Run driver
// ---------------------------------------------------------------------

struct SpendMeter {
    cap_nanodollars: i64,
    spent_nanodollars: i64,
    live: bool,
}

impl SpendMeter {
    fn add(&mut self, nanodollars: i64) -> Result<(), String> {
        self.spent_nanodollars += nanodollars;
        if self.live && self.spent_nanodollars > self.cap_nanodollars {
            return Err(format!(
                "spend cap exceeded: ${:.4} > ${:.2} — aborting",
                self.spent_nanodollars as f64 / 1e9,
                self.cap_nanodollars as f64 / 1e9
            ));
        }
        Ok(())
    }
}

struct RunCtx {
    gateway: Arc<dyn ChatGateway>,
    attribution: Attribution,
    out_dir: PathBuf,
    seed: u64,
}

/// Deterministic per-run planner seed: the base seed folded with the run
/// slug so every (pool, model, wording) run draws distinct pairs.
fn derive_seed(base: u64, slug: &str) -> u64 {
    let hash = blake3::hash(slug.as_bytes());
    let bytes: [u8; 8] = hash.as_bytes()[0..8].try_into().expect("8 bytes");
    base ^ u64::from_le_bytes(bytes)
}

async fn run_one(
    ctx: &RunCtx,
    pool: &Pool,
    model: &str,
    wording: Wording,
) -> Result<(SortedTexts, Vec<ComparisonTrace>, u64), Box<dyn std::error::Error>> {
    let slug = format!(
        "{}-{}-{}",
        pool.slug,
        model.replace('/', "_"),
        wording.slug()
    );
    let rng_seed = derive_seed(ctx.seed, &slug);
    let temp_path = ctx.out_dir.join(format!("run-{slug}.trace.jsonl"));
    let (sink, worker) = JsonlTraceSink::new(&temp_path)?;
    let documents: Vec<RerankDocument> = pool
        .entities
        .iter()
        .map(|(name, _)| RerankDocument {
            id: (*name).to_string(),
            text: (*name).to_string(),
        })
        .collect();
    let execution = RerankExecution::new(Arc::clone(&ctx.gateway), ctx.attribution.clone())
        .run_options(RerankRunOptions {
            rng_seed: Some(rng_seed),
            cache_only: false,
        })
        .trace(&sink);
    let sorted = sort_documents(
        documents,
        wording.text(pool),
        execution,
        SortOptions {
            model: Some(model.to_string()),
            comparison_budget: wording.budget(pool.entities.len()),
            ..SortOptions::default()
        },
    )
    .await?;
    drop(sink);
    worker.join()?;
    let text = std::fs::read_to_string(&temp_path)?;
    let rows: Vec<ComparisonTrace> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    std::fs::remove_file(&temp_path)?;
    Ok((sorted, rows, rng_seed))
}

fn latents_in_pool_order(pool: &Pool, sorted: &SortedTexts) -> Vec<ItemLatent> {
    let by_id: HashMap<&str, (f64, f64)> = sorted
        .items
        .iter()
        .map(|item| (item.id.as_str(), (item.latent_mean, item.latent_std)))
        .collect();
    pool.entities
        .iter()
        .map(|(name, _)| {
            let (mean, std) = *by_id.get(name).expect("sort returns every input id");
            ItemLatent {
                id: (*name).to_string(),
                mean,
                std,
            }
        })
        .collect()
}

#[derive(Serialize)]
struct TraceEnvelope<'a> {
    pool: &'a str,
    model: &'a str,
    wording: &'a str,
    trace: &'a ComparisonTrace,
}

// ---------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let pools = anchor_pools();
    std::fs::create_dir_all(&args.out_dir)?;

    let all_models: Vec<String> = args
        .models
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert!(!all_models.is_empty(), "at least one model required");
    let models: Vec<String> = if args.offline {
        vec![all_models[0].clone()]
    } else {
        all_models
    };

    let gateway: Arc<dyn ChatGateway> = if args.offline {
        Arc::new(SyntheticJudge::new(&pools, args.seed, args.synthetic_sigma))
    } else {
        Arc::new(ProviderGateway::from_env(Arc::new(NoopUsageSink))?)
    };
    let ctx = RunCtx {
        gateway,
        attribution: Attribution::new("llmsort::example::anchor_gauge"),
        out_dir: args.out_dir.clone(),
        seed: args.seed,
    };
    let mut meter = SpendMeter {
        cap_nanodollars: (args.spend_cap_usd * 1e9) as i64,
        spent_nanodollars: 0,
        live: !args.offline,
    };

    let planned_calls: usize =
        pools.iter().map(|p| 8 * p.entities.len()).sum::<usize>() * models.len();
    eprintln!(
        "{} pools x {} models x 3 wordings; comparison-call ceiling {} ({}); cap ${:.2}",
        pools.len(),
        models.len(),
        planned_calls,
        if args.offline { "offline" } else { "LIVE" },
        args.spend_cap_usd
    );

    // --- run the full protocol -------------------------------------------
    let mut engine_spec: Option<EngineSpec> = None;
    let mut runs: Vec<RunReadout> = Vec::new();
    // (pool, model, wording) -> (latent means in pool order, index into runs).
    let mut latent_index: HashMap<(String, String, &'static str), (Vec<f64>, usize)> =
        HashMap::new();
    let trace_path = args.out_dir.join("trace.jsonl");
    let mut trace = std::io::BufWriter::new(std::fs::File::create(&trace_path)?);

    for model in &models {
        for pool in &pools {
            for wording in Wording::ALL {
                let (sorted, rows, rng_seed) = run_one(&ctx, pool, model, wording).await?;
                meter.add(sorted.meta.provider_cost_nanodollars)?;
                for row in &rows {
                    serde_json::to_writer(
                        &mut trace,
                        &TraceEnvelope {
                            pool: pool.slug,
                            model,
                            wording: wording.slug(),
                            trace: row,
                        },
                    )?;
                    trace.write_all(b"\n")?;
                }
                let latents = latents_in_pool_order(pool, &sorted);
                let latent_means: Vec<f64> = latents.iter().map(|l| l.mean).collect();
                let solver = resolve_diagnostics(
                    pool.entities.len(),
                    &rows,
                    &latent_means,
                    &mut engine_spec,
                )?;
                let order = order_stats(&rows);
                eprintln!(
                    "{} {} {}: {} used / {} attempted (budget {}), ${:.4}, order {} agree / {} pairs, hcr {:.3}",
                    pool.slug,
                    model,
                    wording.slug(),
                    sorted.meta.comparisons_used,
                    sorted.meta.comparisons_attempted,
                    sorted.meta.comparison_budget,
                    sorted.meta.provider_cost_nanodollars as f64 / 1e9,
                    order.agreements,
                    order.pairs_with_both_orientations,
                    solver.hcr,
                );
                latent_index.insert(
                    (pool.slug.to_string(), model.clone(), wording.slug()),
                    (latent_means, runs.len()),
                );
                runs.push(RunReadout {
                    pool: pool.slug.to_string(),
                    model: model.clone(),
                    wording: wording.slug().to_string(),
                    criterion: wording.text(pool).to_string(),
                    rng_seed,
                    comparison_budget: sorted.meta.comparison_budget,
                    comparisons_attempted: sorted.meta.comparisons_attempted,
                    comparisons_used: sorted.meta.comparisons_used,
                    comparisons_refused: sorted.meta.comparisons_refused,
                    pairs_counterbalanced: sorted.meta.pairs_counterbalanced,
                    position_flips: sorted.meta.position_flips,
                    stop_reason: serde_json::to_value(sorted.meta.stop_reason)?,
                    provider_input_tokens: sorted.meta.provider_input_tokens,
                    provider_output_tokens: sorted.meta.provider_output_tokens,
                    cost_nanodollars: sorted.meta.provider_cost_nanodollars,
                    latency_ms: sorted.meta.latency_ms,
                    latents,
                    order,
                    solver,
                });
            }
        }
    }
    trace.flush()?;

    // --- gauge readouts per (pool, model) --------------------------------
    let mut gauges: Vec<GaugeReadout> = Vec::new();
    for model in &models {
        for pool in &pools {
            let key = |w: Wording| (pool.slug.to_string(), model.clone(), w.slug());
            let (s_main, main_idx) = latent_index[&key(Wording::Attribute)].clone();
            let (s_para, _) = latent_index[&key(Wording::Paraphrase)].clone();
            let (s_neg, _) = latent_index[&key(Wording::Negation)].clone();
            let main_run = &runs[main_idx];
            let t: Vec<f64> = pool.entities.iter().map(|(_, v)| v.ln()).collect();

            let polarity = spearman_rho(&s_main, &s_neg);
            let paraphrase = spearman_rho(&s_main, &s_para);
            let truth_rho = spearman_rho(&s_main, &t);
            let (truth_slope, truth_r2, truth_pairs) = pairwise_diff_fit(&s_main, &t);
            let (neg_slope, _, _) = pairwise_diff_fit(&s_neg, &t);
            let latent_sd = population_sd(&s_main);
            let mean_std =
                main_run.latents.iter().map(|l| l.std).sum::<f64>() / main_run.latents.len() as f64;

            let oa = main_run.order.agreement_frac;
            let hcr = main_run.solver.hcr;
            let criteria = vec![
                BandCriterion {
                    name: "order_agreement >= 0.90",
                    value: oa,
                    pass: oa.is_some_and(|v| v >= 0.90),
                },
                BandCriterion {
                    name: "curl_hcr <= 0.10",
                    value: Some(hcr),
                    pass: hcr <= 0.10,
                },
                BandCriterion {
                    name: "polarity <= -0.80",
                    value: Some(polarity),
                    pass: polarity <= -0.80,
                },
                BandCriterion {
                    name: "paraphrase >= 0.80",
                    value: Some(paraphrase),
                    pass: paraphrase >= 0.80,
                },
            ];
            let passes = criteria.iter().filter(|c| c.pass).count();
            let band = match passes {
                4 => "grokked",
                2 | 3 => "partial",
                _ => "not",
            };
            gauges.push(GaugeReadout {
                pool: pool.slug.to_string(),
                model: model.clone(),
                n_entities: pool.entities.len(),
                order_agreement: oa,
                order_pairs: main_run.order.pairs_with_both_orientations,
                mean_abs_delta_ln_r: main_run.order.mean_abs_delta_ln_r,
                curl_hcr: hcr,
                curl_observations: main_run.solver.observations,
                filled_triangles: main_run.solver.filled_triangles,
                harmonic_dim: main_run.solver.harmonic_dim,
                polarity_spearman: polarity,
                paraphrase_spearman: paraphrase,
                relation_n: pool.entities.len(),
                signal_latent_sd: latent_sd,
                signal_mean_posterior_std: mean_std,
                signal_ratio: latent_sd / mean_std.max(f64::MIN_POSITIVE),
                truth_spearman: truth_rho,
                truth_slope,
                truth_r2,
                truth_pairs,
                truth_spearman_paraphrase: spearman_rho(&s_para, &t),
                truth_spearman_negation: spearman_rho(&s_neg, &t),
                truth_slope_negation: neg_slope,
                band,
                band_passes: passes,
                band_criteria: criteria,
            });
        }
    }

    // --- verdict table ----------------------------------------------------
    println!("\n| pool | model | OA (pairs) | |Δln r| | hcr (obs) | polarity | paraphrase | signal | ρ_truth | slope | band |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for g in &gauges {
        println!(
            "| {} | {} | {} ({}) | {} | {:.3} ({}) | {:+.2} | {:+.2} | {:.1} | {:+.3} | {:+.2} | {} |",
            g.pool,
            g.model,
            g.order_agreement
                .map_or("n/a".to_string(), |v| format!("{v:.2}")),
            g.order_pairs,
            g.mean_abs_delta_ln_r
                .map_or("n/a".to_string(), |v| format!("{v:.2}")),
            g.curl_hcr,
            g.curl_observations,
            g.polarity_spearman,
            g.paraphrase_spearman,
            g.signal_ratio,
            g.truth_spearman,
            g.truth_slope,
            g.band,
        );
    }

    // --- report -----------------------------------------------------------
    let report = Report {
        generated_at: chrono::Utc::now().to_rfc3339(),
        offline: args.offline,
        models: models.clone(),
        seed: args.seed,
        spend_cap_usd: args.spend_cap_usd,
        band_rule: BAND_RULE,
        pools: pools
            .iter()
            .map(|pool| PoolReport {
                slug: pool.slug,
                attribute: pool.attribute,
                paraphrase: pool.paraphrase,
                negation: pool.negation,
                truth_source: pool.truth_source,
                entities: pool
                    .entities
                    .iter()
                    .map(|&(name, value)| PoolEntity {
                        name,
                        true_value: value,
                        ln_true: value.ln(),
                    })
                    .collect(),
            })
            .collect(),
        engine_spec: engine_spec.expect("at least one run solved"),
        runs,
        gauges,
        total_cost_nanodollars: meter.spent_nanodollars,
        caveats: vec![
            "WST/MST/SST transitivity is part of the PROGRAM.md band definition but is unmeasured here: the 4n/2n/2n budget has no repeat draws over triads. Bands are computed from the four measured criteria only.".to_string(),
            "The spend cap is enforced at run granularity (the sort path owns the call loop), so a breach can overshoot by at most one run's budget.".to_string(),
            "Cities truth values use UN urban-agglomeration figures; 'metro area' definitions diverge for London/Paris/Chicago/Vienna — a stated confound of that pool's truth axis.".to_string(),
            "Amazon river length is contested (6,400 km used; up to 6,992 km claimed); rank-adjacent anchor pairs in the rivers pool differ by ~1-2%, within any judge's plausible noise.".to_string(),
            "Order agreement compares per-orientation MEAN canonical log-ratios per unordered pair; with the default planner most pairs are asked once per orientation.".to_string(),
        ],
    };
    let report_path = args.out_dir.join("report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!(
        "\ndone: total ${:.4} ({}) -> {}",
        meter.spent_nanodollars as f64 / 1e9,
        if args.offline {
            "offline synthetic pricing"
        } else {
            "live"
        },
        report_path.display()
    );

    // --- offline validation: the readouts must recover planted truth ------
    if args.offline {
        let mut failures = Vec::new();
        for g in &report.gauges {
            let checks: Vec<(String, f64, bool)> = vec![
                (
                    format!("{}: truth Spearman >= 0.90", g.pool),
                    g.truth_spearman,
                    g.truth_spearman >= 0.90,
                ),
                (
                    format!("{}: paraphrase Spearman >= 0.80", g.pool),
                    g.paraphrase_spearman,
                    g.paraphrase_spearman >= 0.80,
                ),
                (
                    format!(
                        "{}: polarity Spearman <= -0.80 (negation flips sign)",
                        g.pool
                    ),
                    g.polarity_spearman,
                    g.polarity_spearman <= -0.80,
                ),
                (
                    format!("{}: negated-run truth Spearman <= -0.80", g.pool),
                    g.truth_spearman_negation,
                    g.truth_spearman_negation <= -0.80,
                ),
                (
                    format!("{}: truth slope > 0", g.pool),
                    g.truth_slope,
                    g.truth_slope > 0.0,
                ),
                (
                    format!("{}: negated-run truth slope < 0", g.pool),
                    g.truth_slope_negation,
                    g.truth_slope_negation < 0.0,
                ),
            ];
            for (name, value, pass) in checks {
                println!(
                    "  {} {} ({:+.3})",
                    if pass { "PASS" } else { "FAIL" },
                    name,
                    value
                );
                if !pass {
                    failures.push(name);
                }
            }
        }
        if !failures.is_empty() {
            return Err(format!("offline validation failed: {failures:?}").into());
        }
        println!("offline validation: all checks passed");
    }
    Ok(())
}
