//! Estimator replay: re-derive solver observations from the raw ledger
//! draws persisted in a comparison trace, offline, and verify they
//! round-trip bit-for-bit against what the live run fed the solver.
//!
//! This is the consumer side of the replay seam: each trace row carries
//! `ledger_draws` (every parsed draw trajectory — exact chosen-token masses
//! plus sidebands per stochastic node — and the grammar version that minted
//! them). `decimal_ledger::analyze` over those draws is pure and
//! deterministic, so replaying it and re-applying the ingestion transform
//! (presentation-swap sign flip, variance floor) must reproduce the row's
//! recorded `solver_observation` EXACTLY. Any divergence means the seam is
//! broken — or the estimator changed, which is precisely what this tool
//! exists to measure against historical traces.
//!
//! Usage: cargo run --example replay_trace -- <trace.jsonl>

use ratiometer::rating_engine::Observation;
use ratiometer::rerank::decimal_ledger::{analyze, GRAMMAR_VERSION};
use ratiometer::rerank::multi::EVIDENCE_VAR_FLOOR;
use ratiometer::rerank::ComparisonTrace;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: replay_trace <trace.jsonl>")?;
    let text = std::fs::read_to_string(&path)?;

    let (mut rows, mut with_draws, mut replayed, mut exact, mut version_skip) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut mismatches = Vec::new();

    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let row: ComparisonTrace = serde_json::from_str(line)?;
        let Some(record) = row.ledger_draws.as_ref() else {
            continue;
        };
        with_draws += 1;
        if record.grammar_version != GRAMMAR_VERSION {
            version_skip += 1;
            eprintln!(
                "line {}: grammar {} != current {} — different instrument, not replayed",
                line_no + 1,
                record.grammar_version,
                GRAMMAR_VERSION
            );
            continue;
        }
        let Some(recorded) = row.solver_observation.as_ref() else {
            eprintln!(
                "line {}: ledger draws present but no solver observation recorded (solver \
                 rejection?) — skipped",
                line_no + 1
            );
            continue;
        };
        let Some(outcome) = analyze(&record.draws) else {
            mismatches.push(format!(
                "line {}: analyze() returned None over {} persisted draws, but the live run \
                 produced an observation",
                line_no + 1,
                record.draws.len()
            ));
            continue;
        };
        replayed += 1;

        // Re-apply the exact ingestion transform from multi.rs: moments are
        // in PRESENTED coordinates; a swapped presentation flips the sign;
        // the variance floor guards delta-certain PMFs.
        let mean_ij = if row.swapped {
            -outcome.mean
        } else {
            outcome.mean
        };
        let (i, j) = if row.swapped {
            (row.entity_b_index, row.entity_a_index)
        } else {
            (row.entity_a_index, row.entity_b_index)
        };
        let rebuilt = Observation::from_log_ratio_moments(
            i,
            j,
            mean_ij,
            outcome.var.max(EVIDENCE_VAR_FLOOR),
            recorded.rater_id.clone(),
            recorded.reps,
        );
        let rebuilt_v = serde_json::to_value(&rebuilt)?;
        let recorded_v = serde_json::to_value(recorded)?;
        if rebuilt_v == recorded_v {
            exact += 1;
            println!(
                "line {:>4}: EXACT  {} vs {}  mean={:+.6} var={:.6} e=[{:+.4},{:+.4}] gap={:.2e} ({} draws)",
                line_no + 1,
                row.entity_a_id,
                row.entity_b_id,
                mean_ij,
                outcome.var,
                outcome.e_lo,
                outcome.e_hi,
                outcome.conservation_gap,
                record.draws.len(),
            );
        } else {
            mismatches.push(format!(
                "line {}: MISMATCH\n  recorded: {recorded_v}\n  replayed: {rebuilt_v}",
                line_no + 1
            ));
        }
    }

    println!(
        "\n{rows} rows · {with_draws} with ledger draws · {version_skip} version-skipped · \
         {replayed} replayed · {exact} exact round-trips · {} mismatches",
        mismatches.len()
    );
    for m in &mismatches {
        eprintln!("{m}");
    }
    if !mismatches.is_empty() {
        return Err("replay round-trip failed".into());
    }
    if replayed == 0 {
        return Err("no rows were replayable (no ledger draws in trace?)".into());
    }
    Ok(())
}
