//! Terminal dashboard and proposal rendering for Commander.

use std::collections::HashMap;

use super::store::{
    CommanderStore, ModelRanking, Proposal, ProposalPriority, ProposalStatus, RunStatus,
};

// =============================================================================
// Dashboard
// =============================================================================

/// Render the full commander dashboard to stdout.
pub async fn render_dashboard(store: &CommanderStore) -> Result<(), super::store::StoreError> {
    let runs = store.list_runs().await?;
    let proposals = store.list_proposals(None).await?;
    let rankings = store.get_model_rankings().await?;
    let total_spend = store.total_spend().await?;
    let total_proposals = store.total_proposals().await?;
    let acceptance_rate = store.acceptance_rate(None).await?;
    let cost_per_accepted = store.cost_per_accepted().await?;

    let separator =
        "================================================================================";

    println!("{separator}");
    println!("                         CARDINAL COMMANDER DASHBOARD");
    println!("{separator}");
    println!(
        " Runs: {} total | ${:.2} spent | {} proposals generated",
        runs.len(),
        total_spend as f64 / 1e9,
        total_proposals
    );

    // Officer metrics
    let reviewed_count = proposals
        .iter()
        .filter(|p| p.status != ProposalStatus::Pending)
        .count();
    if reviewed_count > 0 {
        let cpa_str = match cost_per_accepted {
            Some(c) => format!("${:.2}", c / 1e9),
            None => "n/a".to_string(),
        };
        println!(
            " Acceptance: {:.0}% ({} reviewed) | Cost/accepted: {}",
            acceptance_rate * 100.0,
            reviewed_count,
            cpa_str,
        );
    }
    println!();

    // --- Recent Runs ---
    println!("--- Recent Runs ---");
    println!(
        " {:10} {:40} {:6} {:10} {}",
        "ID", "Directive", "Tasks", "Cost", "Status"
    );
    for run in runs.iter().take(10) {
        let directive_short = truncate_str(&run.directive, 38);
        let tasks = format!(
            "{}/{}",
            run.tasks_completed,
            run.tasks_completed + run.tasks_failed
        );
        let cost = format!("${:.2}", run.total_cost_nanodollars as f64 / 1e9);
        let status = match run.status {
            RunStatus::Running => "running",
            RunStatus::Completed => "completed",
            RunStatus::Failed => "failed",
            RunStatus::BudgetExceeded => "budget_exceeded",
        };
        println!(
            " {:10} {:40} {:6} {:10} {}",
            format!("#{}", run.id),
            directive_short,
            tasks,
            cost,
            status
        );
    }
    println!();

    // --- Pending Proposals ---
    let pending: Vec<&Proposal> = proposals
        .iter()
        .filter(|p| p.status == ProposalStatus::Pending)
        .collect();

    println!("--- Pending Proposals ({}) ---", pending.len());
    if pending.is_empty() {
        println!("  No pending proposals.");
    } else {
        // Group by priority
        let mut by_priority: HashMap<ProposalPriority, Vec<&Proposal>> = HashMap::new();
        for p in &pending {
            by_priority.entry(p.priority).or_default().push(*p);
        }

        let priority_order = [
            ProposalPriority::Critical,
            ProposalPriority::High,
            ProposalPriority::Medium,
            ProposalPriority::Low,
        ];

        for priority in &priority_order {
            if let Some(group) = by_priority.get(priority) {
                let label = match priority {
                    ProposalPriority::Critical => "CRITICAL",
                    ProposalPriority::High => "HIGH",
                    ProposalPriority::Medium => "MEDIUM",
                    ProposalPriority::Low => "LOW",
                };
                println!(" {} ({}):", label, group.len());
                for p in group.iter().take(5) {
                    let files = parse_affected_files(&p.affected_files);
                    let file_display = files.first().map(|f| f.as_str()).unwrap_or("?");
                    println!("   [{}] {}", p.short_id, truncate_str(&p.title, 60));
                    println!(
                        "            -> {} | effort: {}",
                        file_display,
                        effort_str(p.estimated_effort)
                    );
                }
                if group.len() > 5 {
                    println!("   ... and {} more", group.len() - 5);
                }
            }
        }
    }
    println!();

    // --- Model Performance ---
    if !rankings.is_empty() {
        println!("--- Model Performance ---");
        let model_stats = compute_model_stats(&rankings);
        let mut sorted: Vec<_> = model_stats.into_iter().collect();
        sorted.sort_by(|a, b| {
            b.1.win_rate
                .partial_cmp(&a.1.win_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (model, stats) in sorted.iter().take(6) {
            println!(
                " {:45} #1 wins: {}/{} ({:.1}%)",
                model,
                stats.first_place_wins,
                stats.total_tasks,
                stats.win_rate * 100.0
            );
        }
        println!();
    }

    // --- Cost Breakdown (last run) ---
    if let Some(last_run) = runs.first() {
        println!("--- Cost (Run #{}) ---", last_run.id);
        println!(
            " Decomposition: ${:.2} | Flywheel: ${:.2} | Extraction: ${:.2} | Total: ${:.2}",
            last_run.decompose_cost_nanodollars as f64 / 1e9,
            last_run.flywheel_cost_nanodollars as f64 / 1e9,
            last_run.extract_cost_nanodollars as f64 / 1e9,
            last_run.total_cost_nanodollars as f64 / 1e9,
        );
        println!();

        // --- Reflection (last run) ---
        if let Ok(Some(reflection)) = store.get_reflection(last_run.id).await {
            println!("--- Reflection (Run #{}) ---", last_run.id);
            println!(
                " Quality: {:.0}/100",
                reflection.quality_score.unwrap_or(0.0) * 100.0
            );
            println!(" {}", reflection.summary);
            if !reflection.efficiency_analysis.is_empty() {
                println!(" Efficiency: {}", reflection.efficiency_analysis);
            }
            // Parse and show recommendations
            if let Ok(recs) = serde_json::from_str::<Vec<String>>(&reflection.recommendations) {
                if !recs.is_empty() {
                    println!(" Recommendations:");
                    for rec in recs.iter().take(3) {
                        println!("   - {rec}");
                    }
                }
            }
        }
    }

    println!("{separator}");
    Ok(())
}

// =============================================================================
// Proposal detail rendering
// =============================================================================

/// Render a single proposal in detail.
pub fn render_proposal(p: &Proposal) {
    let separator = "────────────────────────────────────────────────────────────────";
    println!("{separator}");
    println!("Proposal [{}]", p.short_id);
    println!("{separator}");
    println!("Title:    {}", p.title);
    println!("Category: {:?}", p.category);
    println!("Priority: {:?}", p.priority);
    println!("Effort:   {}", effort_str(p.estimated_effort));
    println!("Status:   {:?}", p.status);
    println!("Task:     {}", p.task_id);
    println!("Run:      #{}", p.run_id);

    let files = parse_affected_files(&p.affected_files);
    if !files.is_empty() {
        println!("Files:");
        for f in &files {
            println!("  - {f}");
        }
    }

    println!();
    println!("{}", p.description);

    if let Some(ref notes) = p.reviewer_notes {
        println!();
        println!("Reviewer notes: {notes}");
    }
    println!("{separator}");
}

/// Render a table of proposals for `review --list`.
pub fn render_proposal_list(proposals: &[Proposal]) {
    if proposals.is_empty() {
        println!("No proposals found.");
        return;
    }

    println!(
        " {:10} {:8} {:10} {:12} {:50}",
        "ID", "Priority", "Category", "Effort", "Title"
    );
    println!("{}", "-".repeat(94));

    for p in proposals {
        let priority_str = match p.priority {
            ProposalPriority::Critical => "CRITICAL",
            ProposalPriority::High => "HIGH",
            ProposalPriority::Medium => "MEDIUM",
            ProposalPriority::Low => "LOW",
        };

        println!(
            " [{:8}] {:8} {:10} {:12} {}",
            p.short_id,
            priority_str,
            format!("{:?}", p.category),
            effort_str(p.estimated_effort),
            truncate_str(&p.title, 48)
        );
    }

    println!();
    println!("{} proposals total", proposals.len());
}

// =============================================================================
// Helpers
// =============================================================================

struct ModelStats {
    first_place_wins: usize,
    total_tasks: usize,
    win_rate: f64,
}

fn compute_model_stats(rankings: &[ModelRanking]) -> HashMap<String, ModelStats> {
    let mut task_count: HashMap<String, usize> = HashMap::new();
    let mut wins: HashMap<String, usize> = HashMap::new();

    // Count unique (run_id, task_id) pairs per model
    let mut seen_tasks: HashMap<String, std::collections::HashSet<(i64, String)>> = HashMap::new();

    for r in rankings {
        let key = (r.run_id, r.task_id.clone());
        seen_tasks.entry(r.model.clone()).or_default().insert(key);

        if r.rank == 1 {
            *wins.entry(r.model.clone()).or_default() += 1;
        }
    }

    for (model, tasks) in &seen_tasks {
        task_count.insert(model.clone(), tasks.len());
    }

    let mut stats = HashMap::new();
    for model in seen_tasks.keys() {
        let total = *task_count.get(model).unwrap_or(&0);
        let w = *wins.get(model).unwrap_or(&0);
        let rate = if total > 0 {
            w as f64 / total as f64
        } else {
            0.0
        };
        stats.insert(
            model.clone(),
            ModelStats {
                first_place_wins: w,
                total_tasks: total,
                win_rate: rate,
            },
        );
    }

    stats
}

fn truncate_str(s: &str, max: usize) -> String {
    let clean = strip_ansi(s);
    if clean.chars().count() <= max {
        clean
    } else {
        let truncated: String = clean.chars().take(max.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

/// Strip ANSI escape sequences from untrusted LLM-generated content.
/// Prevents terminal injection (clipboard writes, screen spoofing, deceptive links).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip ESC + '[' + params + final byte
            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                              // Consume parameter bytes (0x30-0x3F) and intermediate bytes (0x20-0x2F)
                while let Some(&next) = chars.peek() {
                    if next >= '\x20' && next <= '\x3f' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                // Consume final byte (0x40-0x7E)
                if let Some(&next) = chars.peek() {
                    if next >= '\x40' && next <= '\x7e' {
                        chars.next();
                    }
                }
            } else if chars.peek() == Some(&']') {
                // OSC sequence: ESC ] ... ST (or BEL)
                chars.next();
                while let Some(next) = chars.next() {
                    if next == '\x07' {
                        break; // BEL terminates OSC
                    }
                    if next == '\x1b' {
                        if chars.peek() == Some(&'\\') {
                            chars.next(); // consume '\\', ST terminates OSC
                            break;
                        }
                    }
                }
            }
            // else: lone ESC, skip it
        } else if c < '\x20' && c != '\n' && c != '\t' {
            // Strip other control characters (except newline, tab)
            continue;
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_affected_files(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

fn effort_str(e: super::store::EstimatedEffort) -> &'static str {
    match e {
        super::store::EstimatedEffort::Trivial => "trivial",
        super::store::EstimatedEffort::Small => "small",
        super::store::EstimatedEffort::Medium => "medium",
        super::store::EstimatedEffort::Large => "large",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi_removes_csi() {
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m"), "red");
        assert_eq!(strip_ansi("clean text"), "clean text");
    }

    #[test]
    fn test_strip_ansi_removes_osc() {
        // OSC clipboard write attempt
        assert_eq!(strip_ansi("\x1b]52;c;base64data\x07rest"), "rest");
    }

    #[test]
    fn test_strip_ansi_removes_control_chars() {
        assert_eq!(strip_ansi("a\x01b\x02c"), "abc");
        // But preserves newlines and tabs
        assert_eq!(strip_ansi("a\nb\tc"), "a\nb\tc");
    }

    #[test]
    fn test_truncate_strips_then_truncates() {
        let s = "\x1b[31mThis is a very long red string that should be truncated\x1b[0m";
        let result = truncate_str(s, 20);
        assert!(!result.contains('\x1b'));
        assert!(result.ends_with("..."));
        assert!(result.chars().count() <= 20);
    }
}
