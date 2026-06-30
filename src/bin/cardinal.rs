#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use cardinal_harness::cache::SqlitePairwiseCache;
use cardinal_harness::gateway::{NoopUsageSink, ProviderGateway};
use cardinal_harness::rerank::model_policy::ModelPolicy;
use cardinal_harness::rerank::report::validate_report_inputs;
use cardinal_harness::rerank::{
    build_report, expand_prompt_experiment_request, load_policy_from_path, render_report_markdown,
    validate_multi_rerank_request, AttributeVariantSpec, JsonlTraceSink, MultiRerankRequest,
    MultiRerankResponse, PolicyRegistry, PromptExperimentConfig, RerankReportOptions,
    RerankRunOptions, TraceSink,
};
use cardinal_harness::Attribution;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PairwiseModeArg {
    Ratio,
    Ordinal,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ReportFormatArg {
    Md,
    Markdown,
    Json,
}

impl From<PairwiseModeArg> for cardinal_harness::rerank::evaluation::SyntheticPairwiseMode {
    fn from(mode: PairwiseModeArg) -> Self {
        match mode {
            PairwiseModeArg::Ratio => Self::Ratio,
            PairwiseModeArg::Ordinal => Self::Ordinal,
        }
    }
}

#[derive(Parser)]
#[command(name = "cardinal", version, about = "Canonical pairwise ratio CLI")]
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
    /// Prune SQLite cache by age and/or size
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
    /// Run the synthetic pairwise-ratio evaluation suite
    Eval {
        #[arg(long)]
        case: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        curve_csv: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "ratio")]
        mode: PairwiseModeArg,
    },
    /// Run the synthetic Likert baseline evaluation suite
    EvalLikert {
        #[arg(long)]
        case: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        curve_csv: Option<PathBuf>,
        #[arg(long, default_value_t = 10, value_parser = parse_likert_levels)]
        levels: usize,
        #[arg(long, default_value_t = 1.0, value_parser = parse_positive_finite_f64)]
        budget_multiplier: f64,
    },
    /// Compare cardinal pairwise evaluation against the Likert baseline
    EvalCompare {
        #[arg(long)]
        case: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10, value_parser = parse_likert_levels)]
        levels: usize,
        #[arg(long, default_value_t = 1.0, value_parser = parse_positive_finite_f64)]
        budget_multiplier: f64,
        #[arg(long, value_enum, default_value = "ratio")]
        mode: PairwiseModeArg,
    },
    /// Generate a report from a request + response JSON
    Report {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        response: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "md")]
        format: ReportFormatArg,
        #[arg(long, default_value_t = 10, value_parser = parse_report_top_n)]
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
    /// Expand one request across prompt templates and positive/negative attribute variants
    ExperimentExpand {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long = "prompt-template")]
        prompt_template_slugs: Vec<String>,
        #[arg(long)]
        include_negative: bool,
        #[arg(long = "variant-json")]
        variant_json: Vec<PathBuf>,
    },
    /// Validate a multi-rerank request JSON without touching the network or cache
    Validate {
        #[arg(long)]
        request: PathBuf,
    },
    /// Run a rerank from JSON input
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
}

#[derive(Subcommand)]
enum PolicyCommands {
    List,
    Load {
        #[arg(long)]
        config: PathBuf,
    },
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
            mode,
        } => {
            let cfg =
                cardinal_harness::rerank::evaluation::PairwiseEvalConfig { mode: mode.into() };
            let results = cardinal_harness::rerank::evaluation::run_synthetic_suite_with_config(
                case.as_deref(),
                cfg,
            )?;
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
            )?;
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
        Commands::EvalCompare {
            case,
            out,
            levels,
            budget_multiplier,
            mode,
        } => {
            let pairwise_cfg =
                cardinal_harness::rerank::evaluation::PairwiseEvalConfig { mode: mode.into() };
            let likert_cfg = cardinal_harness::rerank::evaluation::LikertEvalConfig {
                levels,
                budget_multiplier,
            };
            let summary = cardinal_harness::rerank::evaluation::run_evaluation_comparison_summary_with_config(
                case.as_deref(),
                pairwise_cfg,
                likert_cfg,
            )?;
            let mut file = File::create(out)?;
            serde_json::to_writer_pretty(&mut file, &summary)?;
            writeln!(file)?;
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
            validate_multi_rerank_request(&req)?;
            validate_report_inputs(&req, &resp)?;
            let opts = RerankReportOptions {
                top_n,
                include_infeasible,
                include_attribute_scores: !no_attr_scores,
                rng_seed,
                model_policy: policy,
                cache_only,
            };
            let report = build_report(&req, &resp, &opts);
            match format {
                ReportFormatArg::Json => {
                    let json = serde_json::to_string_pretty(&report)?;
                    std::fs::write(out, json)?;
                }
                ReportFormatArg::Md | ReportFormatArg::Markdown => {
                    let markdown = render_report_markdown(&report);
                    std::fs::write(out, markdown)?;
                }
            }
        }
        Commands::ExperimentExpand {
            request,
            out,
            prompt_template_slugs,
            include_negative,
            variant_json,
        } => {
            let req: MultiRerankRequest = read_json(&request)?;
            let mut variants = Vec::new();
            for path in variant_json {
                let mut loaded: Vec<AttributeVariantSpec> = read_json(&path)?;
                variants.append(&mut loaded);
            }
            let cfg = PromptExperimentConfig {
                prompt_template_slugs,
                include_negative,
                variants,
            };
            let expanded = expand_prompt_experiment_request(&req, &cfg)?;
            write_json(&out, &expanded)?;
            println!(
                "expanded request: {} attributes -> {} attributes",
                req.attributes.len(),
                expanded.attributes.len()
            );
        }
        Commands::Validate { request } => {
            let req: MultiRerankRequest = read_json(&request)?;
            validate_multi_rerank_request(&req)?;
            println!("valid request: {}", request.display());
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
            validate_multi_rerank_request(&req)?;
            let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
            let cache = SqlitePairwiseCache::new(cache_path)?;
            let _lock = if lock_cache {
                Some(cache.lock_exclusive()?)
            } else {
                None
            };

            let policy_obj = load_policy(policy, policy_config)?;
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

            let mut execution = cardinal_harness::rerank::RerankExecution::new(
                Arc::new(gateway),
                Attribution::new("cardinal::rerank"),
            )
            .cache(&cache)
            .run_options(options);
            if let Some(policy) = policy_obj.clone() {
                execution = execution.model_policy(policy);
            }
            if let Some(trace) = trace_ref {
                execution = execution.trace(trace);
            }

            let resp = cardinal_harness::rerank::multi_rerank(req.clone(), execution).await?;

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
                    model_policy: policy_obj.and_then(|policy| policy.describe()),
                    cache_only,
                };
                let report = build_report(&req, &resp, &opts);
                let markdown = render_report_markdown(&report);
                std::fs::write(report_path, markdown)?;
            }
        }
    }

    Ok(())
}

fn load_policy(
    policy: Option<String>,
    policy_config: Option<PathBuf>,
) -> Result<Option<Arc<dyn ModelPolicy>>, Box<dyn std::error::Error>> {
    if let Some(path) = policy_config {
        return Ok(Some(load_policy_from_path(path)?));
    }
    if let Some(name) = policy {
        let registry = PolicyRegistry::default();
        let available = registry.list().join(", ");
        let policy = registry
            .get(&name)
            .ok_or_else(|| format!("unknown policy '{name}'; available policies: {available}"))?;
        return Ok(Some(policy));
    }
    Ok(None)
}

fn read_json<T: serde::de::DeserializeOwned>(
    path: &PathBuf,
) -> Result<T, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read JSON from {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse JSON in {}: {err}", path.display()).into())
}

fn write_json<T: serde::Serialize>(path: &PathBuf, value: &T) -> Result<(), io::Error> {
    let json = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    std::fs::write(path, json)
}

fn parse_likert_levels(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid integer '{raw}': {err}"))?;
    if value >= 2 {
        Ok(value)
    } else {
        Err(format!("value must be at least 2, got {raw}"))
    }
}

fn parse_report_top_n(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid integer '{raw}': {err}"))?;
    if value >= 1 {
        Ok(value)
    } else {
        Err(format!("value must be at least 1, got {raw}"))
    }
}

fn parse_positive_finite_f64(raw: &str) -> Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|err| format!("invalid number '{raw}': {err}"))?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!(
            "value must be a finite number greater than 0, got {raw}"
        ))
    }
}
