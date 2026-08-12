//! Transport-shape slice vendored from seriate's gateway.
//!
//! Cardinal has its own gateway; the instruments only need the *shape* of a
//! provider logprob position, so only that type crossed over in the
//! fold-back (notes/seriate-fold-2026-08-11.md).

use serde::{Deserialize, Serialize};

/// One completion-token position's logprob, plus its top-k alternatives.
///
/// The caller keeps the WHOLE array of positions from
/// `choices[0].logprobs.content[..]` — deciding which position is "the
/// answer" is a parsing concern for [`crate::seriate::instrument`], not a
/// transport concern here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The token the provider actually emitted at this position.
    pub token: String,
    /// Its logprob.
    pub logprob: f64,
    /// Up to top-k `(token, logprob)` alternatives the provider showed for
    /// this position, in provider order.
    pub top: Vec<(String, f64)>,
}
