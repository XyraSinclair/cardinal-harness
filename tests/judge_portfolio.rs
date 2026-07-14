//! Judge portfolio theory, planted: clones collapse the error-source
//! count and split one judge's weight; a noisier-but-independent judge
//! has strictly positive marginal information (diversification); and the
//! budgeted portfolio ranks by marginal information per dollar. Two
//! failed designs preceded this file's invariants (raw-correlation PR,
//! then eigen loadings) — both caught by these planted cases.

use cardinal_harness::rerank::judge_geometry;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn consensus(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 / (n as f64 - 1.0)) * 2.0 - 1.0)
        .collect()
}

fn judge(consensus: &[f64], noise: f64, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    consensus
        .iter()
        .map(|c| c + noise * rng.next() * 3.0f64.sqrt())
        .collect()
}

#[test]
fn a_clone_pair_shares_one_error_channel_and_one_weight() {
    // Six judges: four honest + a near-clone pair. Triad loadings are
    // clone-robust at this roster size (honest triads are the majority).
    let f = consensus(60);
    let honest: Vec<Vec<f64>> = (0..4).map(|k| judge(&f, 0.4, 7 + k)).collect();
    let a = judge(&f, 0.4, 21);
    let clone: Vec<f64> = {
        let mut rng = Lcg(99);
        a.iter().map(|x| x + 0.03 * rng.next()).collect()
    };
    let mut latents = honest.clone();
    latents.push(a);
    latents.push(clone);
    let names: Vec<String> = ["h1", "h2", "h3", "h4", "a", "a-clone"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let g = judge_geometry(&names, &latents, None).unwrap();
    assert!(
        g.effective_error_sources > 3.9 && g.effective_error_sources < 5.6,
        "six names, five error channels: {}",
        g.effective_error_sources
    );
    // The clone pair's combined weight is close to one honest judge's.
    let pair = g.judges[4].weight + g.judges[5].weight;
    let honest_mean = (0..4).map(|k| g.judges[k].weight).sum::<f64>() / 4.0;
    assert!(
        (pair - honest_mean).abs() < honest_mean,
        "duplication must not double the vote: pair {pair} vs honest {honest_mean}"
    );
    // Marginal information of the clone is near zero: its content is
    // already in the roster.
    assert!(
        g.judges[5].marginal_information < 0.5 * g.judges[0].marginal_information,
        "a clone adds far less than an honest judge: {:?}",
        g.judges
            .iter()
            .map(|e| e.marginal_information)
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_noisier_independent_judge_still_adds_information() {
    // Diversification: the noisy judge's marginal information is strictly
    // positive — an independent mind pays even when it is worse.
    let f = consensus(60);
    let latents = vec![judge(&f, 0.25, 7), judge(&f, 0.25, 11), judge(&f, 0.9, 13)];
    let names: Vec<String> = ["a", "b", "noisy"].iter().map(|s| s.to_string()).collect();
    let g = judge_geometry(&names, &latents, None).unwrap();
    assert!(
        g.judges[2].marginal_information > 0.05,
        "independent error must buy nonzero information: {:?}",
        g.judges[2]
    );
    assert!(
        g.judges[2].weight < g.judges[0].weight,
        "but proportionally less than a cleaner judge: {:?}",
        g.judges.iter().map(|e| e.weight).collect::<Vec<_>>()
    );
    assert!(
        g.judges[2].weight > 0.0,
        "never zero: diversification is not optional: {:?}",
        g.judges[2]
    );
}

#[test]
fn portfolio_order_ranks_by_marginal_information_per_dollar() {
    let f = consensus(60);
    let latents = vec![
        judge(&f, 0.2, 7),  // expensive-good
        judge(&f, 0.4, 11), // cheap-decent
        judge(&f, 0.4, 13), // cheap-decent-2
    ];
    let names: Vec<String> = ["expensive-good", "cheap-decent", "cheap-decent-2"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let costs = [0.20, 0.01, 0.01];
    let g = judge_geometry(&names, &latents, Some(&costs)).unwrap();
    let first = &g.judges[g.portfolio_order[0]];
    assert_ne!(
        first.judge, "expensive-good",
        "20x the price does not buy 20x the marginal information: {g:?}"
    );
}
