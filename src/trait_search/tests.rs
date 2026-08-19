use super::*;

#[test]
fn test_compute_attribute_units_basic() {
    let scores = vec![10.0, 20.0, 30.0];
    let (scale, z, min_norm, pct) = compute_attribute_units(&scores);

    assert!((scale - 10.0).abs() < 1e-9);
    assert_eq!(min_norm.len(), 3);
    assert_eq!(pct.len(), 3);
    assert_eq!(z.len(), 3);

    assert!((min_norm[0] - 1.0).abs() < 1e-9);
    assert!((min_norm[1] - 11.0).abs() < 1e-9);
    assert!((min_norm[2] - 21.0).abs() < 1e-9);

    assert!((pct[0] - (0.5 / 3.0)).abs() < 1e-9);
    assert!((pct[1] - (1.5 / 3.0)).abs() < 1e-9);
    assert!((pct[2] - (2.5 / 3.0)).abs() < 1e-9);

    // Median is 20, MAD is 10 -> sigma = 14.826 -> z = +/- 0.67449...
    assert!((z[1] - 0.0).abs() < 1e-12);
    assert!((z[0] + (10.0 / (10.0 * MAD_TO_SIGMA))).abs() < 1e-9);
    assert!((z[2] - (10.0 / (10.0 * MAD_TO_SIGMA))).abs() < 1e-9);
}

#[test]
fn test_compute_attribute_units_all_equal() {
    let scores = vec![5.0, 5.0, 5.0];
    let (scale, z, min_norm, pct) = compute_attribute_units(&scores);

    assert!((scale - SCALE_FLOOR).abs() < 1e-12);
    assert_eq!(z, vec![0.0, 0.0, 0.0]);
    assert_eq!(min_norm, vec![1.0, 1.0, 1.0]);
    let mut pct_sorted = pct.clone();
    pct_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    assert_eq!(pct_sorted, vec![1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0]);
}

#[test]
fn test_compute_attribute_units_degenerate_mad() {
    // When >= 50% of values tie at the median, MAD can be zero even with non-trivial spread.
    // We should avoid exploding z-scores/weight normalization in this case.
    let scores = vec![0.0, 0.0, 0.097];
    let (scale, z, _min_norm, _pct) = compute_attribute_units(&scores);
    assert!(scale > 1e-3);
    assert!(z.iter().all(|v| v.is_finite()));
    assert!(z[2].abs() < 100.0);
}
