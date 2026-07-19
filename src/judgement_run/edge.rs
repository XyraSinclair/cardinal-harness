//! Wire projection for consumers that need the finite-candidate rerank shape.
//!
//! These DTOs are deliberately outside the judgement-run kernel. They contain
//! no consumer-specific types and can be serialized directly at an API edge.

use serde::{Deserialize, Serialize};

use super::{
    JudgementPrivacy, JudgementRunRecord, JudgementRunResponse, NormalizedJudgementRunRequest,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankJudgementResponse {
    pub entities: Vec<RerankedEntity>,
    #[serde(default)]
    pub global_topk_error: Option<f64>,
    pub judgement_run: JudgementRun,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankedEntity {
    pub id: String,
    pub feasible: bool,
    #[serde(default)]
    pub p_flip: Option<f64>,
    pub attribute_score: RerankAttributeScore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankAttributeScore {
    #[serde(default)]
    pub latent_mean: Option<f64>,
    #[serde(default)]
    pub latent_std: Option<f64>,
    #[serde(default)]
    pub percentile: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgementRun {
    #[serde(default)]
    pub run_ref: Option<String>,
    pub privacy: JudgementPrivacy,
}

#[derive(Debug, thiserror::Error)]
pub enum EdgeMappingError {
    #[error("judgement run {run_ref} is {status}, not completed")]
    NotCompleted {
        run_ref: String,
        status: &'static str,
    },
}

impl TryFrom<&JudgementRunRecord> for RerankJudgementResponse {
    type Error = EdgeMappingError;

    fn try_from(record: &JudgementRunRecord) -> Result<Self, Self::Error> {
        let response =
            record
                .completed_response()
                .ok_or_else(|| EdgeMappingError::NotCompleted {
                    run_ref: record.run_ref.clone(),
                    status: record.terminal.status(),
                })?;
        Ok(project_response(
            response,
            &record.request,
            record.run_ref.clone(),
        ))
    }
}

fn project_response(
    response: &JudgementRunResponse,
    request: &NormalizedJudgementRunRequest,
    run_ref: String,
) -> RerankJudgementResponse {
    RerankJudgementResponse {
        entities: response
            .entities
            .iter()
            .map(|entity| RerankedEntity {
                id: entity.id.clone(),
                feasible: entity.feasible,
                p_flip: Some(entity.p_flip),
                attribute_score: RerankAttributeScore {
                    latent_mean: Some(entity.attribute_score.latent_mean),
                    latent_std: Some(entity.attribute_score.latent_std),
                    percentile: Some(entity.attribute_score.percentile),
                },
            })
            .collect(),
        global_topk_error: Some(response.global_topk_error),
        judgement_run: JudgementRun {
            run_ref: Some(run_ref),
            privacy: request.privacy,
        },
    }
}
