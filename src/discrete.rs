use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightedValue<T> {
    pub value: T,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscreteDistribution<T> {
    pub support: Vec<WeightedValue<T>>,
    pub residual_probability: f64,
}

impl<T> DiscreteDistribution<T> {
    pub fn new(mut support: Vec<WeightedValue<T>>, residual_probability: f64) -> Self {
        support.retain(|entry| entry.probability.is_finite() && entry.probability > 0.0);
        support.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            support,
            residual_probability: residual_probability.clamp(0.0, 1.0),
        }
    }

    pub fn support_probability(&self) -> f64 {
        self.support.iter().map(|entry| entry.probability).sum()
    }

    pub fn total_probability(&self) -> f64 {
        self.support_probability() + self.residual_probability
    }

    pub fn entropy(&self) -> f64 {
        self.support
            .iter()
            .map(|entry| entry.probability)
            .chain(std::iter::once(self.residual_probability))
            .filter(|p| *p > 0.0)
            .map(|p| -p * p.ln())
            .sum()
    }

    pub fn top_probability(&self) -> f64 {
        self.support
            .first()
            .map(|entry| entry.probability)
            .unwrap_or(0.0)
    }

    pub fn probability_of<F>(&self, mut predicate: F) -> f64
    where
        F: FnMut(&T) -> bool,
    {
        self.support
            .iter()
            .filter(|entry| predicate(&entry.value))
            .map(|entry| entry.probability)
            .sum()
    }
}

impl<T: Clone> DiscreteDistribution<T> {
    pub fn renormalized_support(&self) -> Option<Self> {
        let support_probability = self.support_probability();
        if support_probability <= 0.0 {
            return None;
        }

        Some(Self {
            support: self
                .support
                .iter()
                .map(|entry| WeightedValue {
                    value: entry.value.clone(),
                    probability: entry.probability / support_probability,
                })
                .collect(),
            residual_probability: 0.0,
        })
    }

    pub fn expectation_by<F>(&self, mut map: F) -> Option<f64>
    where
        F: FnMut(&T) -> f64,
    {
        let normalized = self.renormalized_support()?;
        Some(
            normalized
                .support
                .iter()
                .map(|entry| entry.probability * map(&entry.value))
                .sum(),
        )
    }

    pub fn variance_by<F>(&self, mut map: F) -> Option<f64>
    where
        F: FnMut(&T) -> f64,
    {
        let mean = self.expectation_by(&mut map)?;
        let normalized = self.renormalized_support()?;
        Some(
            normalized
                .support
                .iter()
                .map(|entry| {
                    let centered = map(&entry.value) - mean;
                    entry.probability * centered * centered
                })
                .sum(),
        )
    }

    pub fn product<U, V, F>(
        &self,
        other: &DiscreteDistribution<U>,
        mut combine: F,
    ) -> DiscreteDistribution<V>
    where
        U: Clone,
        F: FnMut(&T, &U) -> V,
    {
        let mut support = Vec::with_capacity(self.support.len() * other.support.len());
        for left in &self.support {
            for right in &other.support {
                support.push(WeightedValue {
                    value: combine(&left.value, &right.value),
                    probability: left.probability * right.probability,
                });
            }
        }
        let residual_probability =
            (1.0 - self.support_probability() * other.support_probability()).clamp(0.0, 1.0);

        DiscreteDistribution::new(support, residual_probability)
    }
}

#[cfg(test)]
mod tests {
    use super::{DiscreteDistribution, WeightedValue};

    #[test]
    fn distribution_sorts_and_filters_invalid_entries() {
        let dist = DiscreteDistribution::new(
            vec![
                WeightedValue {
                    value: "b",
                    probability: 0.2,
                },
                WeightedValue {
                    value: "a",
                    probability: 0.8,
                },
                WeightedValue {
                    value: "bad",
                    probability: f64::NAN,
                },
            ],
            0.0,
        );

        assert_eq!(dist.support.len(), 2);
        assert_eq!(dist.support[0].value, "a");
        assert!(dist.support[0].probability > dist.support[1].probability);
    }

    #[test]
    fn distribution_entropy_and_total_probability_include_residual() {
        let dist = DiscreteDistribution::new(
            vec![WeightedValue {
                value: 1,
                probability: 0.75,
            }],
            0.25,
        );

        assert!((dist.total_probability() - 1.0).abs() < 1e-9);
        assert!(dist.entropy() > 0.0);
        assert!((dist.top_probability() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn distribution_probability_of_filters_support() {
        let dist = DiscreteDistribution::new(
            vec![
                WeightedValue {
                    value: "A",
                    probability: 0.6,
                },
                WeightedValue {
                    value: "B",
                    probability: 0.3,
                },
            ],
            0.1,
        );

        assert!((dist.probability_of(|value| *value == "A") - 0.6).abs() < 1e-9);
        assert!((dist.support_probability() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn renormalized_support_expectation_and_variance_ignore_residual_mass() {
        let dist = DiscreteDistribution::new(
            vec![
                WeightedValue {
                    value: 1.0,
                    probability: 0.4,
                },
                WeightedValue {
                    value: 3.0,
                    probability: 0.4,
                },
            ],
            0.2,
        );

        let normalized = dist
            .renormalized_support()
            .expect("support should normalize");
        assert!((normalized.total_probability() - 1.0).abs() < 1e-9);
        assert_eq!(normalized.residual_probability, 0.0);

        let expectation = dist.expectation_by(|value| *value).expect("expectation");
        let variance = dist.variance_by(|value| *value).expect("variance");
        assert!((expectation - 2.0).abs() < 1e-9);
        assert!((variance - 1.0).abs() < 1e-9);
    }

    #[test]
    fn product_builds_joint_distribution_and_tracks_residual() {
        let left = DiscreteDistribution::new(
            vec![WeightedValue {
                value: "A",
                probability: 0.8,
            }],
            0.2,
        );
        let right = DiscreteDistribution::new(
            vec![WeightedValue {
                value: 2,
                probability: 0.5,
            }],
            0.5,
        );

        let joint = left.product(&right, |label, ratio| format!("{label}:{ratio}"));
        assert_eq!(joint.support.len(), 1);
        assert_eq!(joint.support[0].value, "A:2");
        assert!((joint.support[0].probability - 0.4).abs() < 1e-9);
        assert!((joint.residual_probability - 0.6).abs() < 1e-9);
    }
}
