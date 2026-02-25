#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use cardinal_harness::cache::SqlitePairwiseCache;
use cardinal_harness::commander;
use cardinal_harness::gateway::{NoopUsageSink, ProviderGateway};
use cardinal_harness::pipeline::{self, ModelPreset, PipelineRequest};
use cardinal_harness::rerank::{
    build_report, load_policy_from_path, render_report_markdown, JsonlTraceSink, ModelPolicy,
    PolicyRegistry, RerankRunOptions, TraceSink,
};
use cardinal_harness::rerank::{MultiRerankRequest, MultiRerankResponse, RerankReportOptions};

#[derive(Parser)]
#[command(name = "cardinal", version, about = "Cardinal harness CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Export SQLite cache to JSONL
    CacheExport {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Prune SQLite cache (by age and/or size)
    CachePrune {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        max_age_days: Option<u64>,
        #[arg(long)]
        max_rows: Option<usize>,
    },
    /// List or load model policies
    Policy {
        #[command(subcommand)]
        command: PolicyCommands,
    },
    /// Run synthetic evaluation suite
    Eval {
        #[arg(long)]
        case: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        curve_csv: Option<PathBuf>,
    },
    /// Run synthetic Likert baseline evaluation suite
    EvalLikert {
        #[arg(long)]
        case: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        curve_csv: Option<PathBuf>,
        /// Number of Likert levels (e.g. 5 or 10)
        #[arg(long, default_value_t = 10)]
        levels: usize,
        /// Multiplies the synthetic comparison budget when allocating ratings
        #[arg(long, default_value_t = 1.0)]
        budget_multiplier: f64,
    },
    /// Run ANP demo pipeline from JSON input
    AnpDemo {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run synthetic ANP typed-vs-forced benchmark
    EvalAnp {
        #[arg(long)]
        out: PathBuf,
    },
    /// Generate a report from a request + response JSON
    Report {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        response: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "md")]
        format: String,
        #[arg(long, default_value_t = 10)]
        top_n: usize,
        #[arg(long)]
        include_infeasible: bool,
        #[arg(long)]
        no_attr_scores: bool,
        #[arg(long)]
        rng_seed: Option<u64>,
        #[arg(long)]
        policy: Option<String>,
        #[arg(long)]
        cache_only: bool,
    },
    /// Run a rerank from JSON input (LLM calls)
    Rerank {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long)]
        lock_cache: bool,
        #[arg(long)]
        cache_only: bool,
        #[arg(long)]
        policy: Option<String>,
        #[arg(long)]
        policy_config: Option<PathBuf>,
        #[arg(long)]
        rng_seed: Option<u64>,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// Multi-model generate → rank → synthesize pipeline
    ///
    /// Sends the same prompt to N models, ranks outputs via pairwise comparison,
    /// then synthesizes the best response using an expert model.
    Pipeline {
        /// Path to pipeline request JSON
        #[arg(long, group = "input")]
        request: Option<PathBuf>,

        /// Inline prompt text (alternative to --request)
        #[arg(long, group = "input")]
        prompt: Option<String>,

        /// Read prompt from file (alternative to --request)
        #[arg(long, group = "input")]
        prompt_file: Option<PathBuf>,

        /// Comma-separated OpenRouter model IDs for generation
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,

        /// Model preset: frontier, balanced, or fast (used when --models is not given)
        #[arg(long, value_enum)]
        preset: Option<CliModelPreset>,

        /// Ranking attributes as "id:weight" pairs (comma-separated).
        /// Omit to use default assessment attributes.
        #[arg(long, value_delimiter = ',')]
        attrs: Option<Vec<String>>,

        /// Use extended attributes (adds verifiability + feasibility, narrows taste)
        #[arg(long)]
        extended_attrs: bool,

        /// Enable requirement_alignment gate (filters off-topic responses)
        #[arg(long)]
        use_gates: bool,

        /// Model for final synthesis (OpenRouter model ID)
        #[arg(long)]
        synthesizer: Option<String>,

        /// Output session JSON
        #[arg(long)]
        out: PathBuf,

        /// SQLite cache for pairwise judgments
        #[arg(long)]
        cache: Option<PathBuf>,

        /// JSONL trace output for ranking phase
        #[arg(long)]
        trace: Option<PathBuf>,

        /// Model policy for ranking phase
        #[arg(long)]
        policy: Option<String>,

        /// Model policy config file for ranking phase
        #[arg(long)]
        policy_config: Option<PathBuf>,

        /// Also write synthesis output to this file (plain text)
        #[arg(long)]
        synthesis_out: Option<PathBuf>,

        /// Context files to inject into generation prompts (glob patterns supported)
        #[arg(long, value_delimiter = ',')]
        context_files: Option<Vec<String>>,

        /// Maximum token budget for context files
        #[arg(long)]
        max_context_tokens: Option<usize>,

        /// Path to ANP network JSON for ANP-adjusted attribute weights
        #[arg(long)]
        anp_network: Option<PathBuf>,
    },
    /// Batch pipeline execution from a task manifest
    ///
    /// Reads a manifest of tasks and runs generate → rank → synthesize for each.
    Flywheel {
        /// Path to task manifest JSON
        #[arg(long)]
        manifest: PathBuf,

        /// Output directory for session JSONs
        #[arg(long)]
        out_dir: PathBuf,

        /// Model preset override: frontier, balanced, or fast
        #[arg(long, value_enum)]
        preset: Option<CliModelPreset>,

        /// SQLite cache for pairwise judgments (shared across tasks)
        #[arg(long)]
        cache: Option<PathBuf>,

        /// Trace output directory
        #[arg(long)]
        trace_dir: Option<PathBuf>,

        /// Number of tasks to run concurrently (default: 1)
        #[arg(long, default_value_t = 1)]
        parallel: usize,

        /// Directory for plain-text synthesis outputs
        #[arg(long)]
        synthesis_out_dir: Option<PathBuf>,

        /// Enable requirement_alignment gate
        #[arg(long)]
        use_gates: bool,

        /// Model policy for ranking phase
        #[arg(long)]
        policy: Option<String>,

        /// Model policy config file for ranking phase
        #[arg(long)]
        policy_config: Option<PathBuf>,
    },
    /// Run the Code Commander: brief → decompose → flywheel → extract → reflect
    Command {
        /// High-level improvement directive
        #[arg(long)]
        directive: String,

        /// SQLite store path (default: .cardinal_commander.sqlite)
        #[arg(long)]
        store: Option<PathBuf>,

        /// Model preset: frontier, balanced, or fast
        #[arg(long, value_enum, default_value = "balanced")]
        preset: CliModelPreset,

        /// Hard spend cap in dollars
        #[arg(long, default_value_t = 5.0)]
        budget: f64,

        /// SQLite cache for pairwise judgments
        #[arg(long)]
        cache: Option<PathBuf>,

        /// Number of concurrent flywheel tasks
        #[arg(long, default_value_t = 1)]
        parallel: usize,

        /// Model for decompose/extract (default: anthropic/claude-opus-4-6)
        #[arg(long, default_value = "anthropic/claude-opus-4-6")]
        commander_model: String,

        /// Enable requirement_alignment gate
        #[arg(long)]
        use_gates: bool,

        /// Skip the reflection phase for quick runs
        #[arg(long)]
        no_reflection: bool,

        /// Fresh start: skip briefing and dedup context from prior runs
        #[arg(long)]
        fresh: bool,

        /// Override the output directory (default: .cardinal_sessions/run_N/)
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
    /// Show the Commander dashboard
    Dashboard {
        /// SQLite store path
        #[arg(long)]
        store: Option<PathBuf>,
    },
    /// Review and manage Commander proposals
    Review {
        /// SQLite store path
        #[arg(long)]
        store: Option<PathBuf>,

        /// List proposals
        #[arg(long)]
        list: bool,

        /// Filter by status: pending, accepted, rejected, deferred
        #[arg(long)]
        status: Option<String>,

        /// Filter by run ID
        #[arg(long)]
        run: Option<i64>,

        /// Show a proposal by short ID
        #[arg(long)]
        show: Option<String>,

        /// Accept a proposal by short ID
        #[arg(long)]
        accept: Option<String>,

        /// Reject a proposal by short ID
        #[arg(long)]
        reject: Option<String>,

        /// Defer a proposal by short ID
        #[arg(long)]
        defer: Option<String>,

        /// Reviewer notes (used with --accept/--reject/--defer)
        #[arg(long)]
        notes: Option<String>,
    },
    /// View LLM traces for a run (full audit trail)
    Traces {
        /// SQLite store path
        #[arg(long)]
        store: Option<PathBuf>,

        /// Run ID to show traces for
        #[arg(long)]
        run: i64,

        /// Filter by phase: briefing, decompose, extract, reflect
        #[arg(long)]
        phase: Option<String>,
    },
    /// Show or re-run reflection for a run
    Reflect {
        /// SQLite store path
        #[arg(long)]
        store: Option<PathBuf>,

        /// Run ID to show reflection for
        #[arg(long)]
        run: i64,
    },
    /// Continuous codebase health scan: enumerate crates, fan out cheap LLM analysis
    Scan {
        /// Scan profile: compilation, dead-code, test-gaps, patterns, deps, security, all
        #[arg(long, value_enum, default_value = "all")]
        profile: CliScanProfile,

        /// Filter crates by name (exact or glob, comma-separated)
        #[arg(long, value_delimiter = ',')]
        crates: Option<Vec<String>>,

        /// Number of crates per LLM task group
        #[arg(long, default_value_t = 5)]
        group_size: usize,

        /// OpenRouter model ID for scan analysis
        #[arg(long, default_value = "google/gemini-3.1-pro-preview")]
        model: String,

        /// SQLite store path (default: .cardinal_commander.sqlite)
        #[arg(long)]
        store: Option<PathBuf>,

        /// Hard spend cap in dollars
        #[arg(long, default_value_t = 2.0)]
        budget: f64,

        /// Number of concurrent scan tasks
        #[arg(long, default_value_t = 4)]
        parallel: usize,

        /// Diff against previous scan run with same profiles
        #[arg(long)]
        diff: bool,

        /// Override the output directory
        #[arg(long)]
        output_dir: Option<PathBuf>,

        /// Workspace root (default: current directory)
        #[arg(long)]
        workspace_root: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum PolicyCommands {
    List,
    Load {
        #[arg(long)]
        config: PathBuf,
    },
}

/// CLI-facing model preset enum (clap::ValueEnum).
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CliModelPreset {
    Frontier,
    Balanced,
    Fast,
}

impl From<CliModelPreset> for ModelPreset {
    fn from(p: CliModelPreset) -> Self {
        match p {
            CliModelPreset::Frontier => ModelPreset::Frontier,
            CliModelPreset::Balanced => ModelPreset::Balanced,
            CliModelPreset::Fast => ModelPreset::Fast,
        }
    }
}

/// CLI-facing scan profile enum (clap::ValueEnum).
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CliScanProfile {
    Compilation,
    DeadCode,
    TestGaps,
    Patterns,
    Deps,
    Security,
    All,
}

impl CliScanProfile {
    fn into_profiles(self) -> Vec<commander::scan::ScanProfile> {
        use commander::scan::ScanProfile;
        match self {
            CliScanProfile::Compilation => vec![ScanProfile::Compilation],
            CliScanProfile::DeadCode => vec![ScanProfile::DeadCode],
            CliScanProfile::TestGaps => vec![ScanProfile::TestGaps],
            CliScanProfile::Patterns => vec![ScanProfile::Patterns],
            CliScanProfile::Deps => vec![ScanProfile::Deps],
            CliScanProfile::Security => vec![ScanProfile::Security],
            CliScanProfile::All => ScanProfile::all().to_vec(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::CacheExport { db, out } => {
            let path = db.unwrap_or_else(SqlitePairwiseCache::default_path);
            let cache = SqlitePairwiseCache::new(path)?;
            cache.export_jsonl(out).await?;
        }
        Commands::CachePrune {
            db,
            max_age_days,
            max_rows,
        } => {
            if max_age_days.is_none() && max_rows.is_none() {
                return Err("cache-prune requires --max-age-days and/or --max-rows".into());
            }
            if matches!(max_rows, Some(0)) {
                return Err("--max-rows must be >= 1".into());
            }
            let path = db.unwrap_or_else(SqlitePairwiseCache::default_path);
            let cache = SqlitePairwiseCache::new(path)?;
            let _lock = cache.lock_exclusive()?;
            let stats = cache.prune(max_age_days, max_rows).await?;
            println!(
                "pruned {} rows; {} rows remain",
                stats.deleted, stats.remaining
            );
        }
        Commands::Policy { command } => match command {
            PolicyCommands::List => {
                let registry = PolicyRegistry::default();
                for name in registry.list() {
                    println!("{name}");
                }
            }
            PolicyCommands::Load { config } => {
                let policy = load_policy_from_path(config)?;
                let description = policy.describe().unwrap_or_else(|| "unknown".to_string());
                println!("{description}");
            }
        },
        Commands::Eval {
            case,
            out,
            curve_csv,
        } => {
            let results =
                cardinal_harness::rerank::evaluation::run_synthetic_suite(case.as_deref());
            let mut file = File::create(out)?;
            for result in &results {
                let line = serde_json::to_string(result)?;
                writeln!(file, "{line}")?;
            }
            if let Some(csv_path) = curve_csv {
                let mut csv = File::create(csv_path)?;
                writeln!(csv, "case,step,error")?;
                for result in results {
                    for (idx, err) in result.error_trajectory.iter().enumerate() {
                        writeln!(csv, "{},{},{}", result.case_name, idx, err)?;
                    }
                }
            }
        }
        Commands::EvalLikert {
            case,
            out,
            curve_csv,
            levels,
            budget_multiplier,
        } => {
            let cfg = cardinal_harness::rerank::evaluation::LikertEvalConfig {
                levels,
                budget_multiplier,
            };
            let results = cardinal_harness::rerank::evaluation::run_likert_baseline_suite(
                case.as_deref(),
                cfg,
            );
            let mut file = File::create(out)?;
            for result in &results {
                let line = serde_json::to_string(result)?;
                writeln!(file, "{line}")?;
            }
            if let Some(csv_path) = curve_csv {
                let mut csv = File::create(csv_path)?;
                writeln!(csv, "case,step,error")?;
                for result in results {
                    for (idx, err) in result.error_trajectory.iter().enumerate() {
                        writeln!(csv, "{},{},{}", result.case_name, idx, err)?;
                    }
                }
            }
        }
        Commands::AnpDemo { input, out } => {
            let req: cardinal_harness::anp::AnpDemoRequest = read_json(&input)?;
            let resp = cardinal_harness::anp::run_demo(req)?;
            write_json(&out, &resp)?;
        }
        Commands::EvalAnp { out } => {
            let results = cardinal_harness::anp::run_synthetic_benchmark_suite()?;
            let mut file = File::create(out)?;
            for result in &results {
                let line = serde_json::to_string(result)?;
                writeln!(file, "{line}")?;
            }
        }
        Commands::Report {
            request,
            response,
            out,
            format,
            top_n,
            include_infeasible,
            no_attr_scores,
            rng_seed,
            policy,
            cache_only,
        } => {
            let req: MultiRerankRequest = read_json(&request)?;
            let resp: MultiRerankResponse = read_json(&response)?;
            let opts = RerankReportOptions {
                top_n,
                include_infeasible,
                include_attribute_scores: !no_attr_scores,
                rng_seed,
                model_policy: policy,
                cache_only,
            };
            let report = build_report(&req, &resp, &opts);
            if format == "json" {
                let json = serde_json::to_string_pretty(&report)?;
                std::fs::write(out, json)?;
            } else {
                let markdown = render_report_markdown(&report);
                std::fs::write(out, markdown)?;
            }
        }
        Commands::Rerank {
            request,
            out,
            cache,
            lock_cache,
            cache_only,
            policy,
            policy_config,
            rng_seed,
            report,
            trace,
        } => {
            let req: MultiRerankRequest = read_json(&request)?;
            let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
            let cache = SqlitePairwiseCache::new(cache_path)?;
            let _lock = if lock_cache {
                Some(cache.lock_exclusive()?)
            } else {
                None
            };

            let policy_obj: Option<Arc<dyn ModelPolicy>> = if let Some(path) = policy_config {
                Some(load_policy_from_path(path)?)
            } else if let Some(name) = policy {
                let registry = PolicyRegistry::default();
                match registry.get(&name) {
                    Some(policy) => Some(policy),
                    None => {
                        let available = registry.list().join(", ");
                        return Err(format!(
                            "unknown policy '{name}'; available policies: {available}"
                        )
                        .into());
                    }
                }
            } else {
                None
            };

            let options = RerankRunOptions {
                rng_seed,
                cache_only,
            };
            let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;

            let (trace_sink, trace_worker) = if let Some(path) = trace {
                let (sink, worker) = JsonlTraceSink::new(path)?;
                (Some(sink), Some(worker))
            } else {
                (None, None)
            };
            let trace_ref = trace_sink.as_ref().map(|sink| sink as &dyn TraceSink);

            let resp = cardinal_harness::rerank::multi_rerank_with_trace(
                Arc::new(gateway),
                Some(&cache),
                policy_obj.clone(),
                Some(&options),
                req.clone(),
                cardinal_harness::Attribution::new("cardinal::rerank"),
                None,
                None,
                trace_ref,
                None,
            )
            .await?;

            write_json(&out, &resp)?;

            drop(trace_sink);
            if let Some(worker) = trace_worker {
                worker.join()?;
            }

            if let Some(report_path) = report {
                let opts = RerankReportOptions {
                    top_n: 10,
                    include_infeasible: false,
                    include_attribute_scores: true,
                    rng_seed,
                    model_policy: policy_obj.and_then(|p| p.describe()),
                    cache_only,
                };
                let report = build_report(&req, &resp, &opts);
                let markdown = render_report_markdown(&report);
                std::fs::write(report_path, markdown)?;
            }
        }
        Commands::Pipeline {
            request,
            prompt,
            prompt_file,
            models,
            preset,
            attrs,
            extended_attrs,
            use_gates,
            synthesizer,
            out,
            cache,
            trace,
            policy,
            policy_config,
            synthesis_out,
            context_files,
            max_context_tokens,
            anp_network: _anp_network,
        } => {
            // Resolve context files from globs/paths
            let loaded_context = if let Some(patterns) = context_files {
                let expanded = pipeline::expand_context_globs(&patterns)?;
                eprintln!("[pipeline] loading {} context files...", expanded.len());
                pipeline::load_context_files(&expanded)?
            } else {
                Vec::new()
            };

            // Build the pipeline request from either JSON file or CLI args
            let mut pipeline_req: PipelineRequest = if let Some(req_path) = request {
                read_json(&req_path)?
            } else {
                // Build from CLI args
                let prompt_text = if let Some(p) = prompt {
                    p
                } else if let Some(path) = prompt_file {
                    std::fs::read_to_string(&path)?
                } else {
                    return Err("pipeline requires --request, --prompt, or --prompt-file".into());
                };

                // Models: from --models, --preset, or error
                let model_list = models.unwrap_or_default();

                let resolved_preset = preset.map(ModelPreset::from);

                let synth_model =
                    synthesizer.unwrap_or_else(|| "anthropic/claude-opus-4-6".to_string());

                let attributes = if let Some(attr_strs) = attrs {
                    parse_attribute_specs(&attr_strs)?
                } else if extended_attrs {
                    pipeline::default_extended_attributes()
                } else {
                    pipeline::default_assessment_attributes()
                };

                PipelineRequest {
                    prompt: prompt_text,
                    system_prompt: None,
                    models: model_list,
                    preset: resolved_preset,
                    context_files: Vec::new(),
                    max_context_tokens,
                    attributes,
                    synthesis_model: synth_model,
                    synthesis_system_prompt: None,
                    generation_temperature: 0.7,
                    synthesis_temperature: 0.3,
                    max_generation_tokens: 4096,
                    max_synthesis_tokens: 8192,
                    rank_config: pipeline::PipelineRankConfig::default(),
                }
            };

            // Merge CLI context files into request
            if !loaded_context.is_empty() {
                pipeline_req.context_files.extend(loaded_context);
            }
            if max_context_tokens.is_some() && pipeline_req.max_context_tokens.is_none() {
                pipeline_req.max_context_tokens = max_context_tokens;
            }

            // Build gates
            let gates = if use_gates {
                pipeline::default_gates()
            } else {
                Vec::new()
            };

            // Set up gateway
            let gateway = Arc::new(ProviderGateway::from_env(Arc::new(NoopUsageSink))?);

            // Set up cache
            let cache_inst = {
                let path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                SqlitePairwiseCache::new(path)?
            };

            // Set up model policy for ranking
            let policy_obj: Option<Arc<dyn ModelPolicy>> = if let Some(path) = policy_config {
                Some(load_policy_from_path(path)?)
            } else if let Some(name) = policy {
                let registry = PolicyRegistry::default();
                match registry.get(&name) {
                    Some(p) => Some(p),
                    None => {
                        let available = registry.list().join(", ");
                        return Err(
                            format!("unknown policy '{name}'; available: {available}").into()
                        );
                    }
                }
            } else {
                None
            };

            // Run the pipeline
            let session = pipeline::run_pipeline_with_trace_file(
                gateway,
                Some(&cache_inst),
                policy_obj,
                trace,
                pipeline_req,
                gates,
            )
            .await
            .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

            // Write session JSON
            write_json(&out, &session)?;
            eprintln!("[pipeline] session written to {}", out.display());

            // Optionally write synthesis text separately
            if let Some(synth_path) = synthesis_out {
                std::fs::write(&synth_path, &session.synthesis.content)?;
                eprintln!("[pipeline] synthesis written to {}", synth_path.display());
            }
        }
        Commands::Flywheel {
            manifest,
            out_dir,
            preset,
            cache,
            trace_dir,
            parallel,
            synthesis_out_dir,
            use_gates,
            policy,
            policy_config,
        } => {
            // Read manifest
            let flywheel_manifest: pipeline::FlywheelManifest = read_json(&manifest)?;

            // Create output directories
            std::fs::create_dir_all(&out_dir)?;
            if let Some(ref synth_dir) = synthesis_out_dir {
                std::fs::create_dir_all(synth_dir)?;
            }
            if let Some(ref trace) = trace_dir {
                std::fs::create_dir_all(trace)?;
            }

            // Set up gateway
            let gateway = Arc::new(ProviderGateway::from_env(Arc::new(NoopUsageSink))?);

            // Set up cache
            let cache_inst = {
                let path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                SqlitePairwiseCache::new(path)?
            };

            // Set up model policy
            let policy_obj: Option<Arc<dyn ModelPolicy>> = if let Some(path) = policy_config {
                Some(load_policy_from_path(path)?)
            } else if let Some(name) = policy {
                let registry = PolicyRegistry::default();
                match registry.get(&name) {
                    Some(p) => Some(p),
                    None => {
                        let available = registry.list().join(", ");
                        return Err(
                            format!("unknown policy '{name}'; available: {available}").into()
                        );
                    }
                }
            } else {
                None
            };

            // Build gates
            let gates = if use_gates {
                pipeline::default_gates()
            } else {
                Vec::new()
            };

            let preset_override = preset.map(ModelPreset::from);

            let summary = pipeline::run_flywheel(
                gateway,
                Some(&cache_inst),
                policy_obj,
                flywheel_manifest,
                &out_dir,
                synthesis_out_dir.as_deref(),
                trace_dir.as_deref(),
                preset_override,
                parallel,
                gates,
            )
            .await;

            // Write summary
            let summary_path = out_dir.join("_summary.json");
            write_json(&summary_path, &summary)?;
            eprintln!("[flywheel] summary written to {}", summary_path.display());

            // Print summary table
            println!("\n--- Flywheel Summary ---");
            println!(
                "Tasks: {} completed, {} failed",
                summary.tasks_completed, summary.tasks_failed
            );
            println!(
                "Total cost: ${:.4}",
                summary.total_cost_nanodollars as f64 / 1_000_000_000.0
            );
            println!();
            for ts in &summary.task_summaries {
                let status = if ts.success { "OK" } else { "FAIL" };
                let model = ts.top_model.as_deref().unwrap_or("-");
                let err = ts.error.as_deref().unwrap_or("");
                println!(
                    "  [{}] {} — top: {}, cost: ${:.4} {}",
                    status,
                    ts.task_id,
                    model,
                    ts.cost_nanodollars as f64 / 1_000_000_000.0,
                    err
                );
            }
        }
        Commands::Command {
            directive,
            store,
            preset,
            budget,
            cache,
            parallel,
            commander_model,
            use_gates,
            no_reflection,
            fresh,
            output_dir,
        } => {
            let config = commander::CommanderConfig {
                directive,
                store_path: store.unwrap_or_else(commander::store::CommanderStore::default_path),
                preset: ModelPreset::from(preset),
                budget_nanodollars: (budget * 1_000_000_000.0) as i64,
                cache_path: cache,
                parallel,
                commander_model,
                use_gates,
                no_reflection,
                fresh,
                output_dir,
            };

            commander::run_command(config)
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
        }
        Commands::Dashboard { store } => {
            let store_path = store.unwrap_or_else(commander::store::CommanderStore::default_path);
            let store_inst = commander::store::CommanderStore::new(&store_path)?;
            commander::dashboard::render_dashboard(&store_inst)
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
        }
        Commands::Review {
            store,
            list,
            status,
            run,
            show,
            accept,
            reject,
            defer,
            notes,
        } => {
            let store_path = store.unwrap_or_else(commander::store::CommanderStore::default_path);
            let store_inst = commander::store::CommanderStore::new(&store_path)?;

            if list {
                let status_filter = status
                    .as_deref()
                    .map(commander::store::ProposalStatus::from_str);
                let mut proposals = store_inst
                    .list_proposals(status_filter)
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
                if let Some(run_id) = run {
                    proposals.retain(|p| p.run_id == run_id);
                }
                commander::dashboard::render_proposal_list(&proposals);
            } else if let Some(id) = show {
                let proposal = store_inst
                    .get_proposal_by_short_id(&id)
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
                commander::dashboard::render_proposal(&proposal);
            } else if let Some(id) = accept {
                store_inst
                    .update_proposal_status(
                        &id,
                        commander::store::ProposalStatus::Accepted,
                        notes.as_deref(),
                    )
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
                println!("Proposal [{id}] accepted.");
            } else if let Some(id) = reject {
                store_inst
                    .update_proposal_status(
                        &id,
                        commander::store::ProposalStatus::Rejected,
                        notes.as_deref(),
                    )
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
                println!("Proposal [{id}] rejected.");
            } else if let Some(id) = defer {
                store_inst
                    .update_proposal_status(
                        &id,
                        commander::store::ProposalStatus::Deferred,
                        notes.as_deref(),
                    )
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
                println!("Proposal [{id}] deferred.");
            } else {
                // Default: list pending
                let proposals = store_inst
                    .list_proposals(Some(commander::store::ProposalStatus::Pending))
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
                commander::dashboard::render_proposal_list(&proposals);
            }
        }
        Commands::Traces { store, run, phase } => {
            let store_path = store.unwrap_or_else(commander::store::CommanderStore::default_path);
            let store_inst = commander::store::CommanderStore::new(&store_path)?;
            let traces = store_inst
                .get_traces_for_run(run, phase.as_deref())
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

            if traces.is_empty() {
                println!("No traces found for run #{run}.");
            } else {
                println!("--- LLM Traces for Run #{run} ({} total) ---", traces.len());
                println!();
                for trace in &traces {
                    let task_str = trace.task_id.as_deref().unwrap_or("-");
                    println!(
                        "[{}] phase={}, task={}, model={}, cost=${:.4}",
                        trace.id,
                        trace.phase,
                        task_str,
                        trace.model,
                        trace.cost_nanodollars as f64 / 1e9,
                    );
                    // Show truncated raw output
                    let preview: String = trace.raw_output.chars().take(200).collect();
                    println!("  output: {preview}");
                    if trace.raw_output.len() > 200 {
                        println!("  ... ({} chars total)", trace.raw_output.len());
                    }
                    println!();
                }
            }
        }
        Commands::Scan {
            profile,
            crates,
            group_size,
            model,
            store,
            budget,
            parallel,
            diff,
            output_dir,
            workspace_root,
        } => {
            let config = commander::scan::ScanConfig {
                profiles: profile.into_profiles(),
                crate_filter: crates.unwrap_or_default(),
                group_size,
                model: model.clone(),
                extract_model: model,
                store_path: store.unwrap_or_else(commander::store::CommanderStore::default_path),
                workspace_root: workspace_root.unwrap_or_else(|| PathBuf::from(".")),
                budget_nanodollars: (budget * 1_000_000_000.0) as i64,
                parallel,
                diff,
                output_dir,
            };

            commander::scan::run_scan(config)
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;
        }
        Commands::Reflect { store, run } => {
            let store_path = store.unwrap_or_else(commander::store::CommanderStore::default_path);
            let store_inst = commander::store::CommanderStore::new(&store_path)?;

            match store_inst
                .get_reflection(run)
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?
            {
                Some(reflection) => {
                    println!("=== Reflection for Run #{run} ===");
                    println!(
                        "Quality: {:.0}/100",
                        reflection.quality_score.unwrap_or(0.0) * 100.0
                    );
                    println!("Summary: {}", reflection.summary);
                    println!("Efficiency: {}", reflection.efficiency_analysis);
                    println!("Cost: ${:.4}", reflection.cost_nanodollars as f64 / 1e9);
                    // Parse recommendations
                    if let Ok(recs) =
                        serde_json::from_str::<Vec<String>>(&reflection.recommendations)
                    {
                        if !recs.is_empty() {
                            println!();
                            println!("Recommendations:");
                            for rec in &recs {
                                println!("  - {rec}");
                            }
                        }
                    }
                    // Parse model insights
                    if let Some(ref insights_json) = reflection.model_insights {
                        if let Ok(insights) = serde_json::from_str::<
                            std::collections::HashMap<String, String>,
                        >(insights_json)
                        {
                            if !insights.is_empty() {
                                println!();
                                println!("Model Insights:");
                                for (model, insight) in &insights {
                                    println!("  {model}: {insight}");
                                }
                            }
                        }
                    }
                }
                None => {
                    println!("No reflection found for run #{run}.");
                    println!(
                        "Run `cardinal command --directive ... --store {}` to generate one.",
                        store_path.display()
                    );
                }
            }
        }
    }

    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(
    path: &PathBuf,
) -> Result<T, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

fn write_json<T: serde::Serialize>(path: &PathBuf, value: &T) -> Result<(), io::Error> {
    let json = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    std::fs::write(path, json)
}

/// Parse "id:weight" attribute specs from CLI args.
/// If weight is omitted, defaults to 1.0.
/// Uses built-in prompts for known attribute IDs.
fn parse_attribute_specs(
    specs: &[String],
) -> Result<Vec<pipeline::PipelineAttribute>, Box<dyn std::error::Error>> {
    let defaults = pipeline::default_assessment_attributes();
    let default_map: std::collections::HashMap<&str, &str> = defaults
        .iter()
        .map(|a| (a.id.as_str(), a.prompt.as_str()))
        .collect();

    let mut attrs = Vec::new();
    for spec in specs {
        let parts: Vec<&str> = spec.splitn(2, ':').collect();
        let id = parts[0].trim();
        let weight: f64 = if parts.len() > 1 {
            parts[1].trim().parse()?
        } else {
            1.0
        };

        let prompt = if let Some(p) = default_map.get(id) {
            p.to_string()
        } else {
            format!("How well does this response demonstrate '{id}'?")
        };

        attrs.push(pipeline::PipelineAttribute {
            id: id.to_string(),
            prompt,
            weight,
        });
    }
    Ok(attrs)
}
