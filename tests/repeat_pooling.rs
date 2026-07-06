//! The DerSimonian–Laird floor, planted both ways: recovers (σ_w, σ_b);
//! goes to zero when there is no heterogeneity (no phantom floor); and
//! the misranking pin — a heavily-resampled frustrated pair drags the
//! naive k/σ² solve into the wrong order while the floored solve holds.

use cardinal_harness::repeat_pooling::{pool_repeats, RepeatDraws};

/// Deterministic LCG uniform in (−1, 1).
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

fn planted(
    latents: &[f64],
    pairs: &[(usize, usize)],
    k: usize,
    sigma_w: f64,
    sigma_b: f64,
    seed: u64,
) -> Vec<RepeatDraws> {
    let mut rng = Lcg(seed);
    pairs
        .iter()
        .map(|&(i, j)| {
            // Uniform(−1,1) has variance 1/3: scale to the target sigma.
            let bias = sigma_b * rng.next() * 3.0f64.sqrt();
            let draws = (0..k)
                .map(|_| latents[i] - latents[j] + bias + sigma_w * rng.next() * 3.0f64.sqrt())
                .collect();
            RepeatDraws { i, j, draws }
        })
        .collect()
}

fn all_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect()
}

#[test]
fn recovers_planted_variance_components() {
    let latents = [0.0, 0.4, 0.9, 1.5, 2.2, 2.6];
    let pooled = pool_repeats(
        6,
        &planted(&latents, &all_pairs(6), 40, 0.30, 0.25, 7),
    )
    .expect("solve");
    assert!(
        (pooled.sigma_w2.sqrt() - 0.30).abs() < 0.03,
        "sigma_w recovered: {}",
        pooled.sigma_w2.sqrt()
    );
    assert!(
        (pooled.sigma_b2.sqrt() - 0.25).abs() < 0.10,
        "sigma_b recovered (moment estimator, 15 pairs): {}",
        pooled.sigma_b2.sqrt()
    );
}

#[test]
fn no_heterogeneity_means_no_phantom_floor() {
    let latents = [0.0, 0.5, 1.1, 1.8, 2.3];
    let pooled = pool_repeats(
        5,
        &planted(&latents, &all_pairs(5), 30, 0.30, 0.0, 11),
    )
    .expect("solve");
    assert!(
        pooled.sigma_b2.sqrt() < 0.08,
        "no planted heterogeneity, no invented floor: {}",
        pooled.sigma_b2.sqrt()
    );
    // With sigma_b ≈ 0 the floored and naive solves must agree.
    for (a, b) in pooled.scores.iter().zip(pooled.scores_naive.iter()) {
        assert!((a - b).abs() < 0.05, "{a} vs {b}");
    }
}

#[test]
fn oversampled_frustrated_pair_misranks_naive_but_not_floored() {
    // Items 1 and 2 are truly separated by 0.30. One frustrated pair
    // (0,2) carries a +1.2 structural bias (pushing 2 DOWN, since the
    // bias inflates 0 over 2) and is sampled 200 times; every honest pair
    // is sampled 8 times. Naive pooling gives the frustrated pair 25x
    // the weight and flips the 1-vs-2 order; the DL floor caps it.
    let latents = [0.0, 0.6, 0.9, 1.5];
    let honest_pairs: Vec<(usize, usize)> = all_pairs(4)
        .into_iter()
        .filter(|&p| p != (0, 2))
        .collect();
    let mut draws = planted(&latents, &honest_pairs, 8, 0.25, 0.0, 13);
    let mut rng = Lcg(17);
    let frustrated: Vec<f64> = (0..200)
        .map(|_| latents[0] - latents[2] + 1.2 + 0.25 * rng.next() * 3.0f64.sqrt())
        .collect();
    draws.push(RepeatDraws {
        i: 0,
        j: 2,
        draws: frustrated,
    });

    let pooled = pool_repeats(4, &draws).expect("solve");
    assert!(
        pooled.sigma_b2 > 0.05,
        "the frustrated pair must surface as heterogeneity: {}",
        pooled.sigma_b2
    );
    // Naive: item 2 dragged below item 1.
    assert!(
        pooled.scores_naive[2] < pooled.scores_naive[1],
        "naive k/sigma^2 pooling must misrank (that is the bug this \
         estimator exists to prevent): {:?}",
        pooled.scores_naive
    );
    // Floored: true order preserved.
    assert!(
        pooled.scores[2] > pooled.scores[1],
        "the DL floor must hold the true order: {:?}",
        pooled.scores
    );
}
