use super::*;

impl TraitSearchManager {
    pub(super) fn solve_attributes(&mut self) -> Result<()> {
        self.scales.clear();
        self.z_scores.clear();
        self.min_norm.clear();
        self.percentiles.clear();
        self.has_degraded = false;

        let n = self.n;
        let mut units_needed: HashSet<&str> = HashSet::new();
        for gate in &self.config.gates {
            if gate.unit != "latent" {
                units_needed.insert(gate.attribute_id.as_str());
            }
        }

        for attr in &self.config.attributes {
            let id = &attr.id;
            let engine =
                self.engines
                    .get_mut(id)
                    .ok_or_else(|| TraitSearchError::InternalError {
                        message: "engine map invariant violated".to_string(),
                    })?;
            let summary = engine.solve();
            if summary.degraded {
                self.has_degraded = true;
            }
            self.frustration.insert(id.clone(), summary.hcr);

            if summary.scores.len() != n || summary.diag_cov.len() != n {
                return Err(TraitSearchError::PosteriorLengthMismatch {
                    attribute_id: id.clone(),
                    scores_len: summary.scores.len(),
                    diag_cov_len: summary.diag_cov.len(),
                    expected_n: n,
                });
            }

            let needs_units = units_needed.contains(id.as_str());
            let (scale, z, min_norm, pct) = if needs_units {
                compute_attribute_units(&summary.scores)
            } else {
                (
                    compute_attribute_scale(&summary.scores),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };

            self.scales.insert(id.clone(), scale);
            if needs_units {
                self.z_scores.insert(id.clone(), z);
                self.min_norm.insert(id.clone(), min_norm);
                self.percentiles.insert(id.clone(), pct);
            }
        }

        Ok(())
    }

    pub(super) fn combine_attributes(&mut self) -> Result<()> {
        let n = self.n;
        let n_attrs = self.config.attributes.len();
        let mut u_mean = vec![0.0; n];
        let mut u_var = vec![0.0; n];

        // Collect per-attribute weighted variance contributions per entity
        // so we can cap outliers before summing.
        let mut attr_var_contributions: Vec<Vec<f64>> = vec![Vec::with_capacity(n_attrs); n];

        for attr in &self.config.attributes {
            let engine =
                self.engines
                    .get(&attr.id)
                    .ok_or_else(|| TraitSearchError::InternalError {
                        message: "engine map invariant violated".to_string(),
                    })?;
            let scores = engine
                .scores()
                .ok_or_else(|| TraitSearchError::InternalError {
                    message: "scores not available; call solve() first".to_string(),
                })?;
            let diag_cov = engine
                .diag_cov()
                .ok_or_else(|| TraitSearchError::InternalError {
                    message: "diag_cov not available; call solve() first".to_string(),
                })?;
            let scale =
                *self
                    .scales
                    .get(&attr.id)
                    .ok_or_else(|| TraitSearchError::InternalError {
                        message: "scales map invariant violated".to_string(),
                    })?;
            let w = attr.weight;

            let inv_scale = 1.0 / scale;
            let inv_scale2 = inv_scale * inv_scale;
            let w2 = w * w;

            for (i, ((mean, attr_contribs), &diag_var)) in u_mean
                .iter_mut()
                .zip(attr_var_contributions.iter_mut())
                .zip(diag_cov.iter())
                .enumerate()
            {
                *mean += w * (scores[i] * inv_scale);
                let contribution = w2 * (diag_var.max(0.0) * inv_scale2);
                attr_contribs.push(contribution);
            }
        }

        // For each entity, cap outlier variance contributions.
        // Use robust fence: cap at median + 3*IQR of that entity's contributions.
        for (var, contribs) in u_var.iter_mut().zip(attr_var_contributions.iter_mut()) {
            *var = robust_capped_sum(contribs);
        }

        let mut feasible_mask = vec![true; n];

        let gates = self.config.gates.clone();
        for gate in gates {
            let scores = self
                .engines
                .get(&gate.attribute_id)
                .and_then(|engine| engine.scores())
                .ok_or_else(|| TraitSearchError::GateUnknownAttribute {
                    attribute_id: gate.attribute_id.clone(),
                })?;

            let gate_vals: &[f64] = match gate.unit.as_str() {
                "latent" => scores,
                "z" => {
                    self.ensure_attribute_units(&gate.attribute_id)?;
                    self.z_scores.get(&gate.attribute_id).ok_or_else(|| {
                        TraitSearchError::InternalError {
                            message: "z_scores map invariant violated".to_string(),
                        }
                    })?
                }
                "percentile" => {
                    self.ensure_attribute_units(&gate.attribute_id)?;
                    self.percentiles.get(&gate.attribute_id).ok_or_else(|| {
                        TraitSearchError::InternalError {
                            message: "percentiles map invariant violated".to_string(),
                        }
                    })?
                }
                "min_norm" => {
                    self.ensure_attribute_units(&gate.attribute_id)?;
                    self.min_norm.get(&gate.attribute_id).ok_or_else(|| {
                        TraitSearchError::InternalError {
                            message: "min_norm map invariant violated".to_string(),
                        }
                    })?
                }
                _ => {
                    return Err(TraitSearchError::UnsupportedGateUnit {
                        unit: gate.unit.clone(),
                    })
                }
            };

            match gate.op.as_str() {
                ">=" => {
                    for (feasible, &gate_val) in feasible_mask.iter_mut().zip(gate_vals.iter()) {
                        *feasible &= gate_val >= gate.threshold;
                    }
                }
                "<=" => {
                    for (feasible, &gate_val) in feasible_mask.iter_mut().zip(gate_vals.iter()) {
                        *feasible &= gate_val <= gate.threshold;
                    }
                }
                _ => {
                    return Err(TraitSearchError::UnsupportedGateOp {
                        op: gate.op.clone(),
                    })
                }
            }
        }

        for (idx, (state, &feasible)) in self
            .entities
            .iter_mut()
            .zip(feasible_mask.iter())
            .enumerate()
        {
            state.feasible = feasible;
            if feasible {
                state.u_mean = u_mean[idx];
                state.u_var = u_var[idx].max(0.0);
            } else {
                state.u_mean = f64::NEG_INFINITY;
                state.u_var = f64::INFINITY;
            }
            state.rank = None;
            state.p_flip = 0.0;
        }

        Ok(())
    }

    pub(super) fn compute_bounds(&self, beta: f64) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut lcb = vec![f64::NEG_INFINITY; self.n];
        let mut ucb = vec![f64::NEG_INFINITY; self.n];
        let mut feasible = Vec::new();

        for (idx, state) in self.entities.iter().enumerate() {
            if !state.feasible {
                continue;
            }
            feasible.push(idx);
            let var = state.u_var.max(0.0);
            let std = var.sqrt();
            lcb[idx] = state.u_mean - beta * std;
            ucb[idx] = state.u_mean + beta * std;
        }

        (lcb, ucb, feasible)
    }

    pub(super) fn critical_pair(&self, lcb: &[f64], ucb: &[f64]) -> Option<(usize, usize)> {
        let k = self.config.topk.k.max(1);
        let topk: Vec<usize> = self.sorted_indices.iter().copied().take(k).collect();
        if topk.is_empty() {
            return None;
        }

        let mut i_star = topk[0];
        let mut l_min = lcb[i_star];
        for &idx in &topk {
            if lcb[idx] < l_min {
                l_min = lcb[idx];
                i_star = idx;
            }
        }

        let topk_set: HashSet<usize> = topk.iter().copied().collect();
        let mut j_star: Option<usize> = None;
        let mut u_max = f64::NEG_INFINITY;
        for &idx in &self.sorted_indices {
            if topk_set.contains(&idx) {
                continue;
            }
            let u = ucb[idx];
            if u > u_max {
                u_max = u;
                j_star = Some(idx);
            }
        }

        j_star.map(|j| (i_star, j))
    }

    pub(super) fn frontier_sets(&self, lcb: &[f64], ucb: &[f64]) -> (Vec<usize>, Vec<usize>) {
        let k = self.config.topk.k.max(1);
        let frontier_width = self.config.topk.band_size.max(1);

        let topk: Vec<usize> = self.sorted_indices.iter().copied().take(k).collect();
        if topk.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let topk_set: HashSet<usize> = topk.iter().copied().collect();
        let band_set: HashSet<usize> = self.band_indices.iter().copied().collect();

        let mut incumbents: Vec<usize> = topk
            .iter()
            .copied()
            .filter(|idx| band_set.contains(idx))
            .collect();
        incumbents.sort_by(|&a, &b| lcb[a].partial_cmp(&lcb[b]).unwrap_or(Ordering::Equal));
        incumbents.truncate(frontier_width.min(k));

        let mut challengers: Vec<usize> = self
            .band_indices
            .iter()
            .copied()
            .filter(|idx| !topk_set.contains(idx))
            .collect();
        challengers.sort_by(|&a, &b| ucb[b].partial_cmp(&ucb[a]).unwrap_or(Ordering::Equal));
        challengers.truncate(frontier_width);

        (incumbents, challengers)
    }

    pub(super) fn build_frontier_candidates(
        &self,
        incumbents: &[usize],
        challengers: &[usize],
    ) -> Vec<(usize, usize)> {
        if incumbents.is_empty() || challengers.is_empty() {
            return Vec::new();
        }

        let mut candidates = Vec::with_capacity(incumbents.len() * challengers.len());
        for &i in incumbents {
            for &j in challengers {
                if i == j {
                    continue;
                }
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                candidates.push((a, b));
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    pub(super) fn global_diff_var_safe(&self, i: usize, j: usize) -> f64 {
        let mut contribs: Vec<f64> = Vec::with_capacity(self.config.attributes.len());
        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = match self.engines.get(attr_id) {
                Some(e) => e,
                None => continue,
            };
            let diag = match engine.diag_cov() {
                Some(v) => v,
                None => continue,
            };
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let w = attr.weight;
            let sigma_i = diag[i].max(0.0).sqrt();
            let sigma_j = diag[j].max(0.0).sqrt();
            let diff_var = (sigma_i + sigma_j) * (sigma_i + sigma_j);
            contribs.push((w / scale).powi(2) * diff_var);
        }
        robust_capped_sum(&mut contribs)
    }

    pub(super) fn global_diff_var_diag(&self, i: usize, j: usize) -> f64 {
        let mut contribs: Vec<f64> = Vec::with_capacity(self.config.attributes.len());
        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = match self.engines.get(attr_id) {
                Some(e) => e,
                None => continue,
            };
            let diag = match engine.diag_cov() {
                Some(v) => v,
                None => continue,
            };
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let w = attr.weight;
            let diff_var = (diag[i].max(0.0) + diag[j].max(0.0)).max(0.0);
            contribs.push((w / scale).powi(2) * diff_var);
        }
        robust_capped_sum(&mut contribs)
    }

    pub(super) fn global_diff_var_effective(&self, i: usize, j: usize) -> Option<f64> {
        let mut contribs: Vec<f64> = Vec::new();
        let mut seen = false;
        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = self.engines.get(attr_id)?;
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let w = attr.weight;

            if let Some(diff_var) = engine.diff_var_for(i, j) {
                contribs.push((w / scale).powi(2) * diff_var.max(0.0));
                seen = true;
            }
        }
        if seen {
            Some(robust_capped_sum(&mut contribs))
        } else {
            None
        }
    }

    pub(super) fn refine_active_variances(&mut self, active: &[usize]) {
        if active.is_empty() {
            return;
        }

        let n_attrs = self.config.attributes.len();
        let mut per_entity_contribs: Vec<Vec<f64>> = (0..active.len())
            .map(|_| Vec::with_capacity(n_attrs))
            .collect();

        for attr in &self.config.attributes {
            let attr_id = &attr.id;
            let engine = match self.engines.get(attr_id) {
                Some(e) => e,
                None => continue,
            };
            let diag = match engine.diag_cov() {
                Some(v) => v,
                None => continue,
            };
            let scale = self
                .scales
                .get(attr_id)
                .copied()
                .unwrap_or(SCALE_FLOOR)
                .max(SCALE_FLOOR);
            let weight_factor = (attr.weight / scale).powi(2);

            let vars = engine
                .marginal_vars_for(active)
                .unwrap_or_else(|| active.iter().map(|&idx| diag[idx].max(0.0)).collect());

            for (pos, v) in vars.iter().enumerate() {
                per_entity_contribs[pos].push(weight_factor * v.max(0.0));
            }
        }

        for (&idx, contribs) in active.iter().zip(per_entity_contribs.iter_mut()) {
            self.entities[idx].u_var = robust_capped_sum(contribs);
        }
    }
}
