//! Multi-attribute trait search manager built on top of RatingEngine.
//!
//! Combines multiple attribute-specific rating engines into a unified
//! objective function with weighted combination and gate-based filtering.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::rating_engine::{self, plan_edges_for_rater, Observation, PlanProposal, RatingEngine};

mod manager;
mod math;
mod ranking;
mod solve;
mod types;

#[cfg(test)]
mod tests;

pub use math::compute_attribute_units;
pub(crate) type AttributeUnits = (f64, Vec<f64>, Vec<f64>, Vec<f64>);
#[cfg(test)]
use math::MAD_TO_SIGMA;
use math::{
    beta_from_tolerated_error, compute_attribute_scale, inversion_prob, robust_capped_sum,
    MAX_BATCH_SIZE, MAX_PLANNER_CANDIDATES, MAX_REFINED_ACTIVE, MIN_ATTR_UNCERTAINTY_WEIGHT,
    MIN_MEMBERSHIP_WEIGHT, MIN_PAIR_PROB, SCALE_FLOOR,
};
pub use types::{
    AttributeConfig, GateSpec, GlobalEntityState, GlobalPlanProposal, Result, TopKConfig,
    TraitSearchConfig, TraitSearchError, TraitSearchManager,
};
