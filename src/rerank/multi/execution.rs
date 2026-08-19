use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::cache::PairwiseCache;
use crate::gateway::{Attribution, ChatGateway};
use crate::rating_engine::Config as EngineConfig;
use crate::trait_search::{AttributeConfig, GateSpec, TopKConfig, TraitSearchConfig};

use super::super::hooks::{ComparisonObserver, WarmStartProvider};
use super::super::model_policy::ModelPolicy;
use super::super::options::RerankRunOptions;
use super::super::trace::TraceSink;
use super::super::types::MultiRerankRequest;
#[derive(Clone)]
pub struct RerankExecution<'a> {
    pub(super) gateway: Arc<dyn ChatGateway>,
    pub(super) cache: Option<&'a dyn PairwiseCache>,
    pub(super) model_policy: Option<Arc<dyn ModelPolicy>>,
    pub(super) run_options: RerankRunOptions,
    pub(super) attribution: Attribution,
    pub(super) warm_start: Option<&'a dyn WarmStartProvider>,
    pub(super) observer: Option<&'a dyn ComparisonObserver>,
    pub(super) trace: Option<&'a dyn TraceSink>,
    pub(super) cancel_flag: Option<&'a AtomicBool>,
}

/// The execution pieces a portable judgement run borrows from a [`RerankExecution`]:
/// gateway, optional trace sink, run options, and whether a cache is attached.
#[doc(hidden)]
pub type JudgementRunInstrumentation<'a> = (
    Arc<dyn ChatGateway>,
    Option<&'a dyn TraceSink>,
    RerankRunOptions,
    bool,
);

impl<'a> RerankExecution<'a> {
    #[must_use]
    pub fn new(gateway: Arc<dyn ChatGateway>, attribution: Attribution) -> Self {
        Self {
            gateway,
            cache: None,
            model_policy: None,
            run_options: RerankRunOptions::default(),
            attribution,
            warm_start: None,
            observer: None,
            trace: None,
            cancel_flag: None,
        }
    }

    #[must_use]
    pub fn cache(mut self, cache: &'a dyn PairwiseCache) -> Self {
        self.cache = Some(cache);
        self
    }

    #[must_use]
    pub fn model_policy(mut self, model_policy: Arc<dyn ModelPolicy>) -> Self {
        self.model_policy = Some(model_policy);
        self
    }

    #[must_use]
    pub fn run_options(mut self, run_options: RerankRunOptions) -> Self {
        self.run_options = run_options;
        self
    }

    #[must_use]
    pub fn warm_start(mut self, warm_start: &'a dyn WarmStartProvider) -> Self {
        self.warm_start = Some(warm_start);
        self
    }

    #[must_use]
    pub fn observer(mut self, observer: &'a dyn ComparisonObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    #[must_use]
    pub fn trace(mut self, trace: &'a dyn TraceSink) -> Self {
        self.trace = Some(trace);
        self
    }

    #[must_use]
    pub fn cancel_flag(mut self, cancel_flag: &'a AtomicBool) -> Self {
        self.cancel_flag = Some(cancel_flag);
        self
    }

    #[doc(hidden)]
    pub fn judgement_run_instrumentation(
        &self,
    ) -> Result<JudgementRunInstrumentation<'a>, &'static str> {
        if self.model_policy.is_some() {
            return Err("model policies have no portable v1 specification");
        }
        if self.warm_start.is_some() {
            return Err("warm starts have no complete comparison trace");
        }
        Ok((
            Arc::clone(&self.gateway),
            self.trace,
            self.run_options.clone(),
            self.cache.is_some(),
        ))
    }

    #[doc(hidden)]
    pub fn with_judgement_run_instrumentation<'b>(
        self,
        gateway: Arc<dyn ChatGateway>,
        trace: &'b dyn TraceSink,
    ) -> RerankExecution<'b>
    where
        'a: 'b,
    {
        RerankExecution {
            gateway,
            cache: self.cache,
            model_policy: self.model_policy,
            run_options: self.run_options,
            attribution: self.attribution,
            warm_start: self.warm_start,
            observer: self.observer,
            trace: Some(trace),
            cancel_flag: self.cancel_flag,
        }
    }
}

#[doc(hidden)]
pub fn build_trait_search_config(req: &MultiRerankRequest) -> (TraitSearchConfig, TopKConfig) {
    let attributes = req
        .attributes
        .iter()
        .map(|attribute| AttributeConfig::new(&attribute.id, attribute.weight))
        .collect();
    let topk = TopKConfig {
        k: req.topk.k,
        weight_exponent: req.topk.weight_exponent,
        tolerated_error: req.topk.tolerated_error,
        band_size: req.topk.band_size,
        effective_resistance_max_active: req.topk.effective_resistance_max_active,
        stop_sigma_inflate: req.topk.stop_sigma_inflate,
        stop_min_consecutive: req.topk.stop_min_consecutive,
        min_explore_degree: req.topk.min_explore_degree,
        prune_p_topk_below: req.topk.prune_p_topk_below,
    };
    let gates = req
        .gates
        .iter()
        .map(|gate| {
            GateSpec::new(
                &gate.attribute_id,
                gate.unit.to_ascii_lowercase(),
                &gate.op,
                gate.threshold,
            )
        })
        .collect();
    (
        TraitSearchConfig::new(req.entities.len(), attributes, topk.clone(), gates),
        topk,
    )
}

#[doc(hidden)]
pub fn build_engine_config(run_options: &RerankRunOptions, topk: &TopKConfig) -> EngineConfig {
    let mut config = EngineConfig::default();
    if let Some(seed) = run_options.rng_seed {
        config.rng_seed = seed;
    }
    config.top_k = Some(topk.k);
    if topk.k > 0 {
        config.tail_weight = (1.0 / topk.k as f64).clamp(0.05, 1.0);
    }
    config
}
