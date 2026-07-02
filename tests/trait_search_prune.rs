//! Deterministic engine-level tests for top-k exploration pruning.

use std::collections::HashMap;

use cardinal_harness::rating_engine::{
    AttributeParams, Observation, PlannerMode, RaterParams, RatingEngine,
};
use cardinal_harness::trait_search::{
    AttributeConfig, TopKConfig, TraitSearchConfig, TraitSearchManager,
};

fn sim_raters() -> HashMap<String, RaterParams> {
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    raters
}

/// A strong descending chain: item 0 best, item n-1 worst. Chain ends have
/// degree 1 (under the default min_explore_degree of 2), so both are
/// exploration candidates; only the hopeless bottom end should be pruned.
fn chain_engine(n: usize) -> RatingEngine {
    let mut engine = RatingEngine::new(n, AttributeParams::default(), sim_raters(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..(n - 1) {
        obs.push(Observation::new(i, i + 1, 3.0, 1.0, "sim", 1.0));
    }
    engine.ingest(&obs);
    engine
}

fn manager_with(k: usize, prune: Option<f64>, n: usize) -> TraitSearchManager {
    let attr_id = "quality";
    let mut engines = HashMap::new();
    engines.insert(attr_id.to_string(), chain_engine(n));
    let mut topk = TopKConfig::new(k);
    topk.prune_p_topk_below = prune;
    let config = TraitSearchConfig::new(n, vec![AttributeConfig::new(attr_id, 1.0)], topk, vec![]);
    TraitSearchManager::new(config, engines).unwrap()
}

#[test]
fn hopeless_chain_end_is_pruned_from_exploration() {
    let n = 8;
    let mut manager = manager_with(2, Some(0.2), n);
    manager.recompute_global_state().unwrap();

    let proposals = manager
        .propose_batch("sim", 16, PlannerMode::Hybrid)
        .unwrap();

    // The bottom chain end (rank n, one observation, essentially zero chance
    // of reaching the top-2) must be excluded from forced exploration.
    assert_eq!(
        manager.explore_pruned_count(),
        1,
        "proposals: {proposals:?}"
    );
    // The TOP chain end also has degree 1 but sits inside the top-k; it must
    // never be pruned.
    let touches_top = proposals.iter().any(|p| p.i == 0 || p.j == 0);
    assert!(
        touches_top,
        "top item must keep its exploration comparisons"
    );
}

#[test]
fn no_pruning_when_disabled() {
    let n = 8;
    let mut manager = manager_with(2, None, n);
    manager.recompute_global_state().unwrap();
    let _ = manager
        .propose_batch("sim", 16, PlannerMode::Hybrid)
        .unwrap();
    assert_eq!(manager.explore_pruned_count(), 0);
}

#[test]
fn pruned_entity_keeps_rank_and_scores() {
    let n = 8;
    let mut manager = manager_with(2, Some(0.2), n);
    manager.recompute_global_state().unwrap();
    let _ = manager
        .propose_batch("sim", 16, PlannerMode::Hybrid)
        .unwrap();
    assert_eq!(manager.explore_pruned_count(), 1);
    // Pruning is about not spending more queries, not about dropping items:
    // the pruned entity still has a rank and remains feasible.
    let state = manager.entity_state(n - 1);
    assert!(state.feasible);
    assert_eq!(state.rank, Some(n));
}
