use std::collections::HashMap;

use cardinal_harness::rating_engine::{AttributeParams, Observation, RaterParams, RatingEngine};
use cardinal_harness::trait_search::{
    AttributeConfig, GateSpec, TopKConfig, TraitSearchConfig, TraitSearchManager,
};

fn sim_raters() -> HashMap<String, RaterParams> {
    let mut raters = HashMap::new();
    raters.insert("sim".to_string(), RaterParams::default());
    raters
}

fn chain_engine(n: usize) -> RatingEngine {
    let mut engine = RatingEngine::new(n, AttributeParams::default(), sim_raters(), None).unwrap();
    let mut obs = Vec::new();
    for i in 0..(n - 1) {
        obs.push(Observation::new(i, i + 1, 3.0, 1.0, "sim", 1.0));
    }
    engine.ingest(&obs);
    engine
}

#[test]
fn percentile_gate_filters_infeasible_entities() {
    let n = 4;
    let attr_id = "quality";

    let mut engines = HashMap::new();
    engines.insert(attr_id.to_string(), chain_engine(n));

    let config = TraitSearchConfig::new(
        n,
        vec![AttributeConfig::new(attr_id, 1.0)],
        TopKConfig::new(1),
        vec![GateSpec::new(attr_id, "percentile", ">=", 0.8)],
    );

    let mut manager = TraitSearchManager::new(config, engines).unwrap();
    manager.recompute_global_state().unwrap();

    // Only the top entity should be feasible under the percentile gate.
    assert_eq!(manager.ranked_indices(), vec![0]);
    assert!(manager.entity_state(0).feasible);
    for idx in 1..n {
        assert!(!manager.entity_state(idx).feasible);
    }

    let pct = manager.attribute_percentiles(attr_id).unwrap();
    assert!(pct[0] >= 0.8);
    for &p in pct.iter().skip(1) {
        assert!(p < 0.8);
    }
}
