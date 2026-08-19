use super::*;

impl TraitSearchManager {
    pub fn certified_stop(&mut self) -> bool {
        if !self.state_valid {
            self.stop_streak = 0;
            return false;
        }
        if self.has_degraded {
            self.stop_streak = 0;
            return false;
        }
        // Certification assumes critical items are well-anchored in the comparison graph
        // (non-trivial degree and shared connectivity), avoiding premature stops on isolated items.
        let beta = beta_from_tolerated_error(self.config.topk.tolerated_error)
            * self.config.topk.stop_sigma_inflate.max(1.0);
        let (lcb, ucb, _feasible) = self.compute_bounds(beta);
        let (i_star, j_star) = match self.critical_pair(&lcb, &ucb) {
            Some(pair) => pair,
            None => {
                self.stop_streak = 0;
                return false;
            }
        };

        let min_degree = 2;
        let anchor_idx = self.sorted_indices.first().copied();
        for attr in &self.config.attributes {
            let engine = match self.engines.get(&attr.id) {
                Some(e) => e,
                None => {
                    self.stop_streak = 0;
                    return false;
                }
            };
            if !engine.has_min_degree(i_star, min_degree) {
                self.stop_streak = 0;
                return false;
            }
            if !engine.has_min_degree(j_star, min_degree) {
                self.stop_streak = 0;
                return false;
            }
            if let Some(anchor) = anchor_idx {
                if !engine.same_component(i_star, anchor) || !engine.same_component(j_star, anchor)
                {
                    self.stop_streak = 0;
                    return false;
                }
            }
        }

        let mut certified = lcb[i_star] > ucb[j_star];
        if certified {
            // Pre-stop verification on the critical pair using stronger variance.
            if let Some(var_eff) = self.global_diff_var_effective(i_star, j_star) {
                let delta = self.entities[i_star].u_mean - self.entities[j_star].u_mean;
                let margin = delta - beta * var_eff.max(0.0).sqrt();
                certified = margin > 0.0;
            }
        }

        if certified {
            self.stop_streak = self.stop_streak.saturating_add(1);
        } else {
            self.stop_streak = 0;
        }

        self.stop_streak >= self.config.topk.stop_min_consecutive.max(1)
    }

    pub(super) fn rank_entities(&mut self) {
        let mut feasible_indices: Vec<usize> = self
            .entities
            .iter()
            .filter(|s| s.feasible)
            .map(|s| s.idx)
            .collect();

        if feasible_indices.is_empty() {
            self.sorted_indices.clear();
            self.band_indices.clear();
            self.boundary_index = None;
            return;
        }

        feasible_indices.sort_by(|&a, &b| {
            let ua = self.entities[a].u_mean;
            let ub = self.entities[b].u_mean;
            ub.partial_cmp(&ua).unwrap_or(Ordering::Equal)
        });
        self.sorted_indices = feasible_indices.clone();

        for (rank, idx) in feasible_indices.iter().enumerate() {
            self.entities[*idx].rank = Some(rank + 1);
        }

        let k = self.config.topk.k.max(1);
        if feasible_indices.len() <= k {
            self.boundary_index = feasible_indices.last().copied();
            self.band_indices.clear();
            for &idx in &feasible_indices {
                self.entities[idx].p_flip = 0.0;
            }
            return;
        }

        let boundary_idx = feasible_indices[k - 1];
        self.boundary_index = Some(boundary_idx);

        let boundary_mean = self.entities[boundary_idx].u_mean;
        let beta = beta_from_tolerated_error(self.config.topk.tolerated_error);
        let (lcb, ucb, _feasible) = self.compute_bounds(beta);

        let mut lcb_vals: Vec<f64> = feasible_indices.iter().map(|&idx| lcb[idx]).collect();
        lcb_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let theta_l = lcb_vals[k - 1];

        let mut ucb_vals: Vec<f64> = feasible_indices.iter().map(|&idx| ucb[idx]).collect();
        ucb_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let theta_u = ucb_vals[k];

        self.band_indices = feasible_indices
            .iter()
            .copied()
            .filter(|&idx| ucb[idx] >= theta_l && lcb[idx] <= theta_u)
            .collect();

        for &idx in &feasible_indices {
            let delta_mu = self.entities[idx].u_mean - boundary_mean;
            let delta_var = self.global_diff_var_safe(idx, boundary_idx);
            self.entities[idx].p_flip = inversion_prob(delta_mu, delta_var);
        }
    }
}
