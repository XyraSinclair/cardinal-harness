#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};

use cardinal_harness::cache::SqlitePairwiseCache;
use cardinal_harness::gateway::{NoopUsageSink, ProviderGateway};
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
