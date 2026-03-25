use std::collections::HashSet;

use super::multi::MultiRerankError;
use super::types::MultiRerankGateSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateUnit {
    Latent,
    Z,
    Percentile,
    MinNorm,
}

impl GateUnit {
    pub(crate) fn parse(unit: &str) -> Result<Self, MultiRerankError> {
        match unit.to_ascii_lowercase().as_str() {
            "latent" => Ok(Self::Latent),
            "z" => Ok(Self::Z),
            "percentile" => Ok(Self::Percentile),
            "min_norm" => Ok(Self::MinNorm),
            _ => Err(MultiRerankError::InvalidRequest(format!(
                "unsupported gate unit: {unit}"
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Latent => "latent",
            Self::Z => "z",
            Self::Percentile => "percentile",
            Self::MinNorm => "min_norm",
        }
    }

    pub(crate) fn select(self, latent: f64, z: f64, min_norm: f64, percentile: f64) -> f64 {
        match self {
            Self::Latent => latent,
            Self::Z => z,
            Self::Percentile => percentile,
            Self::MinNorm => min_norm,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateOp {
    GreaterThanOrEqual,
    LessThanOrEqual,
}

impl GateOp {
    pub(crate) fn parse(op: &str) -> Result<Self, MultiRerankError> {
        match op {
            ">=" => Ok(Self::GreaterThanOrEqual),
            "<=" => Ok(Self::LessThanOrEqual),
            _ => Err(MultiRerankError::InvalidRequest(format!(
                "unsupported gate op (expected \">=\" or \"<=\"): {op}"
            ))),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::GreaterThanOrEqual => ">=",
            Self::LessThanOrEqual => "<=",
        }
    }

    pub(crate) fn passes(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::GreaterThanOrEqual => value >= threshold,
            Self::LessThanOrEqual => value <= threshold,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ParsedGateSpec<'a> {
    pub(crate) attribute_id: &'a str,
    pub(crate) unit: GateUnit,
    pub(crate) op: GateOp,
    pub(crate) threshold: f64,
}

pub(crate) fn validate_gate_specs<'a>(
    gates: &'a [MultiRerankGateSpec],
    attribute_ids: &HashSet<&str>,
) -> Result<Vec<ParsedGateSpec<'a>>, MultiRerankError> {
    gates
        .iter()
        .map(|gate| {
            if !attribute_ids.contains(gate.attribute_id.as_str()) {
                return Err(MultiRerankError::InvalidRequest(format!(
                    "gate references unknown attribute: {}",
                    gate.attribute_id
                )));
            }

            let unit = GateUnit::parse(&gate.unit)?;
            let op = GateOp::parse(&gate.op)?;

            if matches!(unit, GateUnit::Percentile) && !(0.0..=1.0).contains(&gate.threshold) {
                return Err(MultiRerankError::InvalidRequest(format!(
                    "percentile gate threshold must be in [0,1]: {}",
                    gate.threshold
                )));
            }

            Ok(ParsedGateSpec {
                attribute_id: gate.attribute_id.as_str(),
                unit,
                op,
                threshold: gate.threshold,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_gate_specs_accepts_case_insensitive_units() {
        let attribute_ids = HashSet::from(["quality"]);
        let gates = vec![MultiRerankGateSpec {
            attribute_id: "quality".to_string(),
            unit: "Percentile".to_string(),
            op: ">=".to_string(),
            threshold: 0.6,
        }];

        let parsed = validate_gate_specs(&gates, &attribute_ids).expect("valid gate");
        assert_eq!(parsed[0].unit, GateUnit::Percentile);
        assert_eq!(parsed[0].op, GateOp::GreaterThanOrEqual);
    }
}
