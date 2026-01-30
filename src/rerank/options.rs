//! Optional execution settings for reproducible runs.

#[derive(Debug, Clone, Default)]
pub struct RerankRunOptions {
    /// Override the internal RNG seed used by the planner.
    pub rng_seed: Option<u64>,
    /// Require all comparisons to be served from cache (no network calls).
    pub cache_only: bool,
}
