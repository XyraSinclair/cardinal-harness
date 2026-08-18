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
}

pub(crate) fn validate_gate_specs(
    gates: &[MultiRerankGateSpec],
    attribute_ids: &HashSet<&str>,
) -> Result<(), MultiRerankError> {
    gates.iter().try_for_each(|gate| {
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

        let _ = (unit, op);
        Ok(())
    })
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

        validate_gate_specs(&gates, &attribute_ids).expect("valid gate");
    }
}
