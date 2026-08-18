#![forbid(unsafe_code)]

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use llmsort::cache::SqlitePairwiseCache;
use llmsort::gateway::{NoopUsageSink, ProviderGateway};
use llmsort::rerank::model_policy::ModelPolicy;
use llmsort::rerank::report::validate_report_inputs;
use llmsort::rerank::{
    build_report, load_policy_from_path, render_report_markdown, validate_multi_rerank_request,
    JsonlTraceSink, MultiRerankRequest, MultiRerankResponse, PolicyRegistry, RerankReportOptions,
    RerankRunOptions, TraceSink,
};
use llmsort::Attribution;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ReportFormatArg {
    Md,
    Markdown,
    Json,
}

#[derive(Parser)]
#[command(
    name = "llmsort",
    version,
    about = "Canonical pairwise ratio CLI",
    after_help = "The stability-promised verbs are `sort` and `judge` (plus the judgment-packet \
format they emit). Verbs marked (research) are honest, provenanced instruments \
that are free to change shape without notice (AGENTS.md: canonical vs \
research-grade surface)."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SortFormatArg {
    Text,
    Json,
    Jsonl,
    Csv,
}

#[derive(Subcommand)]
enum Commands {
    /// Sort a list of items by a natural-language criterion
    ///
    /// Reads newline-delimited items (or a JSON array) from FILE or stdin and
    /// prints them sorted best-first. Requires OPENROUTER_API_KEY unless
    /// --cache-only is set and the cache already holds every judgement.
    ///
    /// Example: llmsort sort examples/sort-demo.txt --by "usefulness as advice"
    Sort {
        /// Input file; '-' or omitted reads stdin
        file: Option<PathBuf>,
        /// Criterion to sort by, e.g. "clarity of explanation"
        #[arg(long)]
        by: String,
        /// Model slug (OpenRouter), e.g. anthropic/claude-sonnet-4.6
        #[arg(long)]
        model: Option<String>,
        /// Built-in model policy name (see `llmsort policy list`)
        #[arg(long)]
        policy: Option<String>,
        /// Model policy JSON file
        #[arg(long)]
        policy_config: Option<PathBuf>,
        /// Maximum pairwise comparisons to spend
        #[arg(long)]
        budget: Option<usize>,
        /// Certify only the top K items (default: whole list)
        #[arg(long)]
        top_k: Option<usize>,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: SortFormatArg,
        /// In text mode, prefix each line with `mean±std<TAB>`
        #[arg(long)]
        scores: bool,
        /// Worst first instead of best first
        #[arg(long)]
        reverse: bool,
        /// Also judge the OPPOSITE of the criterion (`lack of <criterion>`),
        /// fold it in with weight -1, and report cross-side consistency
        #[arg(long)]
        two_sided: bool,
        /// Alternate phrasing of the criterion; judged as an extra attribute
        /// and reported as a paraphrase-consistency probe (repeatable)
        #[arg(long)]
        also_by: Vec<String>,
        /// Ask each planned pair in one random order only, instead of the
        /// default both-orders counterbalancing (halves cost, loses the
        /// position-bias diagnostic)
        #[arg(long)]
        no_counterbalance: bool,
        /// Prompt template: canonical_v2 (default), canonical_bucket_v1, or
        /// ratio_letter_v1 (single-token PMF evidence via answer logprobs;
        /// degrades loudly to sampled mode where providers hide them)
        #[arg(long)]
        template: Option<String>,
        /// First expand the criterion into a precise judging rubric with one
        /// LLM call, print it to stderr, then sort by the rubric
        #[arg(long)]
        elaborate: bool,
        /// Stop spending exploration comparisons on items whose probability
        /// of reaching the top-k drops below this (requires --top-k intent;
        /// pruned count is reported in the run summary)
        #[arg(long)]
        prune_below: Option<f64>,
        /// RNG seed for reproducible planning
        #[arg(long)]
        seed: Option<u64>,
        /// Judgements in flight at once (default 8). Lower it for
        /// rate-limited rails: a subscription CLI rail that 429s under a burst
        /// backs off for minutes, so 8-wide bursts cost more wall-clock than
        /// a 2-wide steady stream
        #[arg(long)]
        concurrency: Option<usize>,
        /// Serve judgements from cache only; error on any cache miss
        #[arg(long)]
        cache_only: bool,
        /// Do not read or write the pairwise cache
        #[arg(long)]
        no_cache: bool,
        /// SQLite cache path (default: shared user cache)
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Write a JSONL trace of every comparison
        #[arg(long)]
        trace: Option<PathBuf>,
        /// Suppress the run summary on stderr
        #[arg(long)]
        quiet: bool,
        /// Print the worst-case comparison count and dollar cost, then exit
        /// without touching the network or cache
        #[arg(long)]
        estimate: bool,
    },
    /// One pairwise judgement between two items, fully transparent
    ///
    /// The lowest-level primitive: see exactly what the judge is asked
    /// (--show-prompt) and exactly what it answered. Items are literal text
    /// or @path to read a file.
    Judge {
        /// First item (literal text, or @path)
        item_a: String,
        /// Second item (literal text, or @path)
        item_b: String,
        /// Criterion to judge by
        #[arg(long)]
        by: String,
        /// Model slug (OpenRouter)
        #[arg(long)]
        model: Option<String>,
        /// Prompt template slug
        #[arg(long, default_value = "canonical_v2")]
        template: String,
        /// Print the fully rendered system + user prompt to stderr first
        #[arg(long)]
        show_prompt: bool,
        /// Structured JSON output on stdout
        #[arg(long)]
        json: bool,
        /// Do not read or write the pairwise cache
        #[arg(long)]
        no_cache: bool,
        /// SQLite cache path (default: shared user cache)
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Susceptibility probe: judge under neutral, pro-first, and
        /// pro-second requester framings (each in both presentation orders,
        /// 6 comparisons) and report whether the belief survives the spin
        #[arg(long)]
        spin: bool,
        /// Sweep framing intensity from -3 to +3 (14 comparisons) and fit
        /// the response line: chi as a slope plus a linearity R² — separates
        /// a genuinely rigid judge from a threshold sycophant. Implies the
        /// full sweep instead of the 6-call --spin probe.
        #[arg(long)]
        sweep: bool,
        /// Orbit transform: measure the judgment under the full Z₂³ group
        /// (order × polarity × wording, 8 comparisons), pull back through
        /// the known equivariances, and report the character decomposition
        /// — belief = the invariant coefficient, every bias a named
        /// orthogonal coefficient, Parseval as the energy budget
        #[arg(long)]
        orbit: bool,
        /// Repeat the judgement N times varying only a suffix nonce
        /// (cache-friendly: the long prefix stays byte-identical, so
        /// provider prompt caching bills it at the cached rate) and report
        /// the mean, the spread sigma_w — the within-pair
        /// context-sensitivity noise the DL floor consumes — and the
        /// provider's cached-token count
        #[arg(long)]
        draws: Option<usize>,
        /// Sampling temperature for --draws (default 0: spread = pure
        /// context sensitivity, not sampling noise)
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Wording-invariance probe: ask the same question as "times more",
        /// "what fraction", and "which has LESS" (6 comparisons) — a
        /// coherent ratio judge must recover the same signed log-ratio
        /// through all three; disagreement separates inversion failure
        /// from numerical framing bias
        #[arg(long)]
        wordings: bool,
        /// Consortium verdict: judge models, comma-separated (≥ 2). Each
        /// judge measures the full Z₂³ orbit (8 comparisons); complete
        /// orbits become judgment packets and the belief is computed by
        /// FUSING them — one number, an explicit error budget (within-judge
        /// orbit bias + cross-judge spread), and portable evidence
        #[arg(long)]
        consortium: Option<String>,
        /// Write one judgment packet JSON per usable judge to this
        /// directory (with --consortium)
        #[arg(long)]
        packets_out: Option<PathBuf>,
    },
    /// (research) Explain an existing ranking: which attributes reconstruct it?
    ///
    /// FILE (or stdin) holds items in YOUR believed order, best first.
    /// Each --candidate attribute is measured with pairwise judgements and
    /// scored on how well it — alone and in weighted combination —
    /// reconstructs your order.
    Explain {
        /// Input file in believed order, best first; '-' or omitted reads stdin
        file: Option<PathBuf>,
        /// Candidate attribute (repeatable)
        #[arg(long)]
        candidate: Vec<String>,
        /// Ask an LLM to propose this many additional candidate attributes
        #[arg(long)]
        propose: Option<usize>,
        /// Model slug (OpenRouter)
        #[arg(long)]
        model: Option<String>,
        /// Total comparison budget across all candidates
        #[arg(long)]
        budget: Option<usize>,
        /// Structured JSON output on stdout
        #[arg(long)]
        format_json: bool,
        /// Do not read or write the pairwise cache
        #[arg(long)]
        no_cache: bool,
        /// SQLite cache path (default: shared user cache)
        #[arg(long)]
        cache: Option<PathBuf>,
        /// RNG seed for reproducible planning
        #[arg(long)]
        seed: Option<u64>,
    },
    /// (research) Export SQLite cache to JSONL
    CacheExport {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// (research) Prune SQLite cache by age and/or size
    CachePrune {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        max_age_days: Option<u64>,
        #[arg(long)]
        max_rows: Option<usize>,
    },
    /// (research) List or load model policies
    Policy {
        #[command(subcommand)]
        command: PolicyCommands,
    },
    /// (research) Generate a report from a request + response JSON
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
    /// (research) Validate a multi-rerank request JSON without touching the network or cache
    Validate {
        #[arg(long)]
        request: PathBuf,
    },
    /// (research) Run a rerank from JSON input
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
        Commands::Sort {
            file,
            by,
            model,
            policy,
            policy_config,
            budget,
            top_k,
            format,
            scores,
            reverse,
            two_sided,
            also_by,
            no_counterbalance,
            template,
            elaborate,
            prune_below,
            seed,
            concurrency,
            cache_only,
            no_cache,
            cache,
            trace,
            quiet,
            estimate,
        } => {
            if cache_only && no_cache {
                return Err("--cache-only and --no-cache are mutually exclusive".into());
            }
            let raw = read_sort_input(file.as_deref())?;
            let documents = parse_sort_items(&raw)?;
            if documents.is_empty() {
                return Err("no items to sort: input is empty".into());
            }

            if estimate {
                let opts = llmsort::rerank::SortOptions {
                    model: model.clone(),
                    comparison_budget: budget,
                    top_k,
                    counterbalance: !no_counterbalance,
                    two_sided,
                    also_by: also_by.clone(),
                    prune_p_topk_below: prune_below,
                    prompt_template_slug: template.clone(),
                    ..Default::default()
                };
                let simple = llmsort::rerank::sort::sort_request(documents.clone(), &by, &opts);
                let multi = llmsort::rerank::simple::to_multi_request(&simple);
                let charge = llmsort::rerank::estimate_max_rerank_charge(&multi);
                println!(
                    "worst case: {} comparisons · ~{} input + {} output tokens each · provider max ${:.4}",
                    charge.comparison_budget,
                    charge.input_tokens_per_comparison,
                    charge.output_tokens_per_comparison,
                    charge.provider_cost_max_nanodollars as f64 / 1e9,
                );
                eprintln!(
                    "estimate only — no network, no cache; actual runs stop earlier on certified top-k or cache hits"
                );
                return Ok(());
            }

            let gateway = provider_gateway(cache_only)?;

            let cache_store = if no_cache {
                None
            } else {
                let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                Some(SqlitePairwiseCache::new(cache_path)?)
            };
            let policy_obj = load_policy(policy, policy_config)?;

            let (trace_sink, trace_worker) = if let Some(path) = trace {
                let (sink, worker) = JsonlTraceSink::new(path)?;
                (Some(sink), Some(worker))
            } else {
                (None, None)
            };
            let trace_ref = trace_sink.as_ref().map(|sink| sink as &dyn TraceSink);

            let gateway = Arc::new(gateway);
            let mut execution = llmsort::rerank::RerankExecution::new(
                gateway.clone(),
                Attribution::new("llmsort::sort"),
            )
            .run_options(RerankRunOptions {
                rng_seed: seed,
                cache_only,
            });
            if let Some(store) = cache_store.as_ref() {
                execution = execution.cache(store);
            }
            if let Some(policy) = policy_obj {
                execution = execution.model_policy(policy);
            }
            if let Some(trace) = trace_ref {
                execution = execution.trace(trace);
            }

            let opts = llmsort::rerank::SortOptions {
                model: model.clone(),
                comparison_budget: budget,
                top_k,
                counterbalance: !no_counterbalance,
                two_sided,
                also_by,
                prune_p_topk_below: prune_below,
                prompt_template_slug: template,
                comparison_concurrency: concurrency,
                ..Default::default()
            };
            let criterion = if elaborate {
                let rubric = llmsort::rerank::elaborate_criterion(
                    gateway.as_ref(),
                    model.as_deref(),
                    &by,
                    Attribution::new("llmsort::sort::elaborate"),
                )
                .await?;
                if !quiet {
                    eprintln!(
                        "elaborated criterion ({}, ${:.4}):
{}
",
                        rubric.model_used,
                        rubric.provider_cost_nanodollars as f64 / 1e9,
                        rubric.elaborated
                    );
                }
                rubric.elaborated
            } else {
                by.clone()
            };
            let mut sorted =
                llmsort::rerank::sort_documents(documents, &criterion, execution, opts).await?;

            drop(trace_sink);
            if let Some(worker) = trace_worker {
                worker.join()?;
            }

            // A sort where every comparison failed or was refused is not a
            // sort; refuse to emit uninformative output on stdout.
            if sorted.meta.comparisons_attempted > 0 && sorted.meta.comparisons_used == 0 {
                return Err(format!(
                    "all {} comparison attempts failed ({} refused); output would be \
                     uninformative. Re-run with --trace <path> to see per-comparison \
                     errors (bad model slug and invalid API key are the usual causes).",
                    sorted.meta.comparisons_attempted, sorted.meta.comparisons_refused,
                )
                .into());
            }

            if reverse {
                sorted.items.reverse();
            }
            let stdout = io::stdout();
            let mut out = stdout.lock();
            render_sorted(&mut out, &sorted, format, scores)?;

            if !quiet {
                let meta = &sorted.meta;
                let cost_usd = meta.provider_cost_nanodollars as f64 / 1e9;
                let estimate = if meta.provider_cost_is_estimate {
                    "~"
                } else {
                    ""
                };
                let evidence = if meta.evidence_judgements > 0 {
                    let residual = meta
                        .evidence_order_residual_mean_abs
                        .map(|r| format!(", order-residual {r:.3} nats"))
                        .unwrap_or_default();
                    format!(
                        " · evidence: {}/{} logprob-mode, visible {:.2}{residual}",
                        meta.logprob_mode_judgements,
                        meta.evidence_judgements,
                        meta.evidence_visible_mass_mean.unwrap_or(0.0)
                    )
                } else {
                    String::new()
                };
                let frustration = meta
                    .judgement_frustration_mean
                    .map(|f| format!(" · frustration {f:.3}"))
                    .unwrap_or_default();
                let flips = if meta.pairs_counterbalanced > 0 {
                    format!(
                        " · order flips: {}/{}",
                        meta.position_flips, meta.pairs_counterbalanced
                    )
                } else {
                    String::new()
                };
                eprintln!(
                    "sorted {} items by \"{by}\" · {} comparisons ({} cached, {} refused) · {estimate}${cost_usd:.4}{flips}{evidence}{frustration} · stop: {}",
                    sorted.items.len(),
                    meta.comparisons_used,
                    meta.comparisons_cached,
                    meta.comparisons_refused,
                    serde_json::to_value(meta.stop_reason)?.as_str().unwrap_or("unknown"),
                );
                // Error budget, experimentalist-style: statistical and
                // systematic components side by side, each in its native
                // unit — never silently pooled.
                {
                    let stat = if sorted.items.is_empty() {
                        None
                    } else {
                        Some(
                            sorted.items.iter().map(|i| i.latent_std).sum::<f64>()
                                / sorted.items.len() as f64,
                        )
                    };
                    let mut parts: Vec<String> = Vec::new();
                    if let Some(stat) = stat {
                        parts.push(format!("stat ±{stat:.3} (posterior, mean)"));
                    }
                    if let Some(residual) = meta.evidence_order_residual_mean_abs {
                        parts.push(format!("syst order {residual:.3} nats/pair"));
                    }
                    if let Some(hcr) = meta.judgement_frustration_mean {
                        parts.push(format!("syst cyclic {:.1}% of energy", hcr * 100.0));
                    }
                    if meta.topk_error > 0.0 {
                        parts.push(format!(
                            "rank risk {:.3} (top-k flip probability)",
                            meta.topk_error
                        ));
                    }
                    if parts.len() > 1 {
                        eprintln!("error budget: {}", parts.join(" · "));
                    }
                }
                for probe in &sorted.probes {
                    let kind = match probe.kind {
                        llmsort::rerank::SortProbeKind::Opposite => "opposite",
                        llmsort::rerank::SortProbeKind::Paraphrase => "paraphrase",
                    };
                    match probe.consistency {
                        Some(c) => {
                            let verdict = if c >= 0.7 {
                                "consistent"
                            } else if c >= 0.3 {
                                "shaky"
                            } else {
                                "INCOHERENT for this judge"
                            };
                            eprintln!(
                                "probe [{kind}] \"{}\": consistency {c:+.2} — {verdict}",
                                probe.prompt
                            );
                        }
                        None => eprintln!(
                            "probe [{kind}] \"{}\": not enough shared scores to assess",
                            probe.prompt
                        ),
                    }
                }
            }
        }
        Commands::Judge {
            item_a,
            item_b,
            by,
            model,
            template,
            show_prompt,
            json,
            no_cache,
            cache,
            spin,
            sweep,
            orbit,
            wordings,
            draws,
            temperature,
            consortium,
            packets_out,
        } => {
            let text_a = read_item_arg(&item_a)?;
            let text_b = read_item_arg(&item_b)?;
            let model = model
                .as_deref()
                .unwrap_or("openai/gpt-5.4-mini")
                .to_string();

            if let Some(consortium) = consortium {
                let models: Vec<String> = consortium
                    .split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect();
                require_openrouter_key()?;
                let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
                let cache_store = if no_cache {
                    None
                } else {
                    let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                    Some(SqlitePairwiseCache::new(cache_path)?)
                };
                let cache_ref = cache_store
                    .as_ref()
                    .map(|c| c as &dyn llmsort::cache::PairwiseCache);
                // Stable entity labels: packets accrete across runs by
                // id + content hash, so @path items keep their file stem
                // and literals get a content-derived label.
                let label = |arg: &str, text: &str| -> String {
                    arg.strip_prefix('@')
                        .and_then(|p| {
                            std::path::Path::new(p)
                                .file_stem()
                                .map(|s| s.to_string_lossy().into_owned())
                        })
                        .unwrap_or_else(|| {
                            llmsort::packet::entity_text_hash(text)[..12].to_string()
                        })
                };
                let id_a = label(&item_a, &text_a);
                let id_b = label(&item_b, &text_b);
                let created = chrono::Utc::now().to_rfc3339();
                let report = llmsort::rerank::consortium_verdict(
                    &gateway,
                    cache_ref,
                    &models,
                    &by,
                    (&id_a, &text_a),
                    (&id_b, &text_b),
                    &template,
                    &created,
                    Attribution::new("llmsort::judge::consortium"),
                )
                .await?;
                let written = if let Some(dir) = packets_out.as_ref() {
                    std::fs::create_dir_all(dir)?;
                    let mut paths = Vec::new();
                    for packet in &report.packets {
                        let path = dir.join(format!(
                            "packet-{}-{}.json",
                            packet.judge.replace('/', "-"),
                            &packet.id().0[..12],
                        ));
                        std::fs::write(&path, serde_json::to_string_pretty(packet)?)?;
                        paths.push(path);
                    }
                    paths
                } else {
                    Vec::new()
                };
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    println!(
                        "{:<34} {:>8} {:>9}  top bias",
                        "judge", "belief", "coherence"
                    );
                    for j in &report.judges {
                        match (j.belief, j.coherence, &j.top_bias) {
                            (Some(b), Some(c), Some((name, coef))) => {
                                println!("{:<34} {b:+8.3} {c:9.3}  {name} {coef:+.3}", j.model)
                            }
                            _ => println!(
                                "{:<34} orbit incomplete ({} refusals) — excluded",
                                j.model, j.refusals
                            ),
                        }
                    }
                    match (report.belief, report.ratio) {
                        (Some(b), Some(r)) => {
                            let toward = if b >= 0.0 { &id_a } else { &id_b };
                            println!(
                                "belief (fused, toward {toward}): {:+.3} nats · ratio {r:.2}×",
                                b
                            );
                            let spread = report
                                .judge_spread_nats
                                .map(|s| format!("{s:.3}"))
                                .unwrap_or_else(|| "n/a (1 judge)".into());
                            let bias = report
                                .orbit_bias_rms
                                .map(|s| format!("{s:.3}"))
                                .unwrap_or_else(|| "n/a".into());
                            let unanimity = match report.direction_unanimous {
                                Some(true) => format!(
                                    "unanimous ({}/{})",
                                    report.usable_judges, report.usable_judges
                                ),
                                Some(false) => "SPLIT".into(),
                                None => "n/a".into(),
                            };
                            println!(
                                "error budget: syst orbit-bias rms {bias} · syst judge spread \
                                 {spread} (nats) · direction {unanimity}"
                            );
                        }
                        _ => println!("no usable judge completed its orbit — no verdict"),
                    }
                    if let Some(matrix) = &report.residual_correlation {
                        let rows: Vec<String> = matrix
                            .iter()
                            .map(|row| {
                                row.iter()
                                    .map(|v| format!("{v:+.2}"))
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            })
                            .collect();
                        println!(
                            "shared-bias correlation (orbit residuals, n = 8 cells): [{}]",
                            rows.join(" | ")
                        );
                    }
                    for path in &written {
                        println!("packet: {}", path.display());
                    }
                }
                eprintln!(
                    "{} judges ({} usable) · {} comparisons ({} cached) · ${:.4}",
                    report.judges.len(),
                    report.usable_judges,
                    report.comparisons,
                    report.comparisons_cached,
                    report.cost_nanodollars as f64 / 1e9,
                );
                return Ok(());
            }

            if let Some(k) = draws {
                require_openrouter_key()?;
                let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
                let report = llmsort::rerank::nonce_draws(
                    &gateway,
                    &model,
                    &template,
                    &by,
                    ("A", &text_a),
                    ("B", &text_b),
                    k,
                    temperature,
                    7,
                    Attribution::new("llmsort::judge::draws"),
                )
                .await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    for (i, d) in report.draws.iter().enumerate() {
                        match d {
                            Some(m) => println!("draw {i}: {m:+.3} nats"),
                            None => println!("draw {i}: refused"),
                        }
                    }
                    match (report.mean, report.sigma_w) {
                        (Some(m), Some(s)) => println!(
                            "mean {m:+.3} nats · sigma_w {s:.3} (n = {})",
                            report.comparisons - report.refusals
                        ),
                        (Some(m), None) => println!("mean {m:+.3} nats (single usable draw)"),
                        _ => println!("no usable draws"),
                    }
                    println!(
                        "cache: {} of {} input tokens billed as cached",
                        report.cache_read_tokens_total, report.input_tokens_total
                    );
                }
                eprintln!(
                    "{} draws ({} refused) · ${:.4}",
                    report.comparisons,
                    report.refusals,
                    report.cost_nanodollars as f64 / 1e9,
                );
                return Ok(());
            }

            if orbit {
                require_openrouter_key()?;
                let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
                let cache_store = if no_cache {
                    None
                } else {
                    let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                    Some(SqlitePairwiseCache::new(cache_path)?)
                };
                let cache_ref = cache_store
                    .as_ref()
                    .map(|c| c as &dyn llmsort::cache::PairwiseCache);
                let report = llmsort::rerank::orbit_transform(
                    &gateway,
                    cache_ref,
                    &model,
                    &by,
                    ("A", &text_a),
                    ("B", &text_b),
                    &template,
                    Attribution::new("llmsort::judge::orbit"),
                )
                .await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else if report.refusals > 0 {
                    println!(
                        "orbit incomplete: {} refusals in 8 variants — no transform",
                        report.refusals
                    );
                } else {
                    let total: f64 = report.energies.iter().sum();
                    for (idx, name) in llmsort::rerank::CHARACTERS.iter().enumerate() {
                        println!(
                            "{name:<26} {:+.3} nats  ({:.1}% of energy)",
                            report.coefficients[idx],
                            100.0 * report.energies[idx] / total.max(1e-12)
                        );
                    }
                    if let Some(c) = report.coherence {
                        println!("coherence (invariant fraction): {c:.3}");
                    }
                    println!("parseval residual: {:.2e}", report.parseval_residual);
                }
                eprintln!(
                    "{} comparisons ({} cached) · ${:.4}",
                    report.comparisons,
                    report.comparisons_cached,
                    report.cost_nanodollars as f64 / 1e9,
                );
                return Ok(());
            }

            if wordings {
                require_openrouter_key()?;
                let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
                let cache_store = if no_cache {
                    None
                } else {
                    let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                    Some(SqlitePairwiseCache::new(cache_path)?)
                };
                let cache_ref = cache_store
                    .as_ref()
                    .map(|c| c as &dyn llmsort::cache::PairwiseCache);
                let report = llmsort::rerank::wording_invariance(
                    &gateway,
                    cache_ref,
                    &model,
                    &by,
                    ("A", &text_a),
                    ("B", &text_b),
                    Attribution::new("llmsort::judge::wordings"),
                )
                .await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    for r in &report.readings {
                        match r.mean_log_ratio {
                            Some(m) => println!("{:<14} {m:+.3} nats", r.template),
                            None => println!("{:<14} refused", r.template),
                        }
                    }
                    match report.sign_consistent {
                        Some(true) => {
                            println!("inversion: OK — the judge can mirror its own scale")
                        }
                        Some(false) => println!(
                            "inversion: FAILS — asking \"which has less\" flips the belief"
                        ),
                        None => println!("inversion: undetermined"),
                    }
                    if let Some(d) = report.max_disagreement_nats {
                        println!(
                            "max wording disagreement: {d:.3} nats{}",
                            if d > 0.5 {
                                " — numerical framing bias"
                            } else {
                                ""
                            }
                        );
                    }
                }
                eprintln!(
                    "{} comparisons ({} cached) · ${:.4}",
                    report.comparisons,
                    report.comparisons_cached,
                    report.cost_nanodollars as f64 / 1e9,
                );
                return Ok(());
            }

            if sweep {
                require_openrouter_key()?;
                let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
                let cache_store = if no_cache {
                    None
                } else {
                    let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                    Some(SqlitePairwiseCache::new(cache_path)?)
                };
                let cache_ref = cache_store
                    .as_ref()
                    .map(|c| c as &dyn llmsort::cache::PairwiseCache);
                let report = llmsort::rerank::spin_sweep(
                    &gateway,
                    cache_ref,
                    &model,
                    &template,
                    &by,
                    ("A", &text_a),
                    ("B", &text_b),
                    Attribution::new("llmsort::judge::sweep"),
                )
                .await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    for r in &report.readings {
                        match r.mean_log_ratio {
                            Some(m) => println!("field {:+}: {m:+.3} nats", r.field),
                            None => println!("field {:+}: refused", r.field),
                        }
                    }
                    match (report.chi_slope, report.linearity_r2) {
                        (Some(chi), Some(r2)) => {
                            let even = report
                                .even_response_mean
                                .map(|e| format!(" · even component {e:+.3} nats"))
                                .unwrap_or_default();
                            println!(
                                "response: odd slope {chi:+.3} nats/step · linear R² {r2:.3}{even}"
                            );
                        }
                        _ => println!("response: unmeasurable (refusals)"),
                    }
                    match report.belief_survives_sweep {
                        Some(true) => println!("sign(m) constant over the sweep: yes"),
                        Some(false) => println!("sign(m) constant over the sweep: no"),
                        None => {
                            println!("sign(m) constant over the sweep: undetermined (m(0) = 0)")
                        }
                    }
                }
                eprintln!(
                    "{} comparisons ({} cached) · ${:.4}",
                    report.comparisons,
                    report.comparisons_cached,
                    report.cost_nanodollars as f64 / 1e9,
                );
                return Ok(());
            }

            if spin {
                require_openrouter_key()?;
                let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
                let cache_store = if no_cache {
                    None
                } else {
                    let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                    Some(SqlitePairwiseCache::new(cache_path)?)
                };
                let cache_ref = cache_store
                    .as_ref()
                    .map(|c| c as &dyn llmsort::cache::PairwiseCache);
                let report = llmsort::rerank::spin_probe(
                    &gateway,
                    cache_ref,
                    &model,
                    &template,
                    &by,
                    ("A", &text_a),
                    ("B", &text_b),
                    Attribution::new("llmsort::judge::spin"),
                )
                .await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    for reading in &report.readings {
                        let label = match reading.framing {
                            llmsort::rerank::SpinFraming::Neutral => "neutral   ",
                            llmsort::rerank::SpinFraming::ProFirst => "pro-A spin",
                            llmsort::rerank::SpinFraming::ProSecond => "pro-B spin",
                        };
                        match reading.mean_log_ratio {
                            Some(m) => {
                                let winner = if m >= 0.0 { "A" } else { "B" };
                                let order = if reading.flipped_by_order {
                                    " · ORDER-FLIPPED"
                                } else {
                                    ""
                                };
                                println!("{label}: {winner} wins · {:+.3} nats{order}", m);
                            }
                            None => println!("{label}: refused"),
                        }
                    }
                    match report.susceptibility_nats {
                        Some(chi) => println!("susceptibility (secant): {chi:+.3} nats/spin"),
                        None => println!("susceptibility: unmeasurable (refusals)"),
                    }
                    match report.belief_survives_spin {
                        Some(true) => println!("sign(m) constant across framings: yes"),
                        Some(false) => println!("sign(m) constant across framings: no"),
                        None => println!("sign(m) constant across framings: undetermined"),
                    }
                }
                eprintln!(
                    "{} comparisons ({} cached) · ${:.4}",
                    report.comparisons,
                    report.comparisons_cached,
                    report.cost_nanodollars as f64 / 1e9,
                );
                return Ok(());
            }

            let spec = llmsort::rerank::PairwiseComparisonSpec {
                model: &model,
                attribute: llmsort::rerank::PairwiseComparisonAttribute {
                    id: "judge",
                    prompt: &by,
                    prompt_template_slug: Some(&template),
                },
                entity_a: llmsort::rerank::PairwiseComparisonEntity {
                    id: "A",
                    text: &text_a,
                },
                entity_b: llmsort::rerank::PairwiseComparisonEntity {
                    id: "B",
                    text: &text_b,
                },
            };

            if show_prompt {
                let rendered = spec.prompt_instance();
                eprintln!(
                    "--- system ---
{}
--- user ---
{}
---",
                    rendered.system, rendered.user
                );
            }

            require_openrouter_key()?;
            let gateway = ProviderGateway::from_env(Arc::new(NoopUsageSink))?;
            let cache_store = if no_cache {
                None
            } else {
                let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                Some(SqlitePairwiseCache::new(cache_path)?)
            };
            let cache_ref = cache_store
                .as_ref()
                .map(|c| c as &dyn llmsort::cache::PairwiseCache);

            let (judgement, usage) = llmsort::rerank::compare_pair(
                &gateway,
                cache_ref,
                llmsort::rerank::PairwiseComparisonRequest {
                    spec,
                    cache_only: false,
                    attribution: Attribution::new("llmsort::judge"),
                },
            )
            .await?;

            let cost_usd = usage.provider_cost_nanodollars as f64 / 1e9;
            match judgement {
                llmsort::rerank::PairwiseJudgement::Observation {
                    higher_ranked,
                    ratio,
                    confidence,
                } => {
                    let winner = match higher_ranked {
                        llmsort::rerank::HigherRanked::A => "A",
                        llmsort::rerank::HigherRanked::B => "B",
                    };
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({
                                "higher_ranked": winner,
                                "ratio": ratio,
                                "confidence": confidence,
                                "refused": false,
                                "model": model,
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "cost_nanodollars": usage.provider_cost_nanodollars,
                                "cached": usage.cached,
                            })
                        );
                    } else {
                        let cached = if usage.cached { " · cached" } else { "" };
                        println!(
                            "{winner} wins · ratio {ratio} · confidence {confidence:.2} · ${cost_usd:.4}{cached}"
                        );
                    }
                }
                llmsort::rerank::PairwiseJudgement::Refused => {
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({
                                "refused": true,
                                "model": model,
                                "cost_nanodollars": usage.provider_cost_nanodollars,
                                "cached": usage.cached,
                            })
                        );
                    } else {
                        println!("REFUSED · ${cost_usd:.4}");
                    }
                }
            }
        }
        Commands::Explain {
            file,
            candidate,
            propose,
            model,
            budget,
            format_json,
            no_cache,
            cache,
            seed,
        } => {
            let raw = read_sort_input(file.as_deref())?;
            let documents = parse_sort_items(&raw)?;
            if documents.len() < 3 {
                return Err(
                    "explain requires at least 3 items (in your believed order, best first)".into(),
                );
            }
            require_openrouter_key()?;
            let gateway = Arc::new(ProviderGateway::from_env(Arc::new(NoopUsageSink))?);

            let mut candidates = candidate;
            if let Some(count) = propose {
                let (proposed, usage) = llmsort::rerank::propose_candidates(
                    gateway.as_ref(),
                    model.as_deref().unwrap_or("openai/gpt-5.4-mini"),
                    &documents,
                    count,
                    Attribution::new("llmsort::explain::propose"),
                )
                .await?;
                eprintln!(
                    "proposed {} candidate attributes (${:.4}):",
                    proposed.len(),
                    usage.cost_nanodollars as f64 / 1e9
                );
                for c in &proposed {
                    eprintln!("  - {c}");
                }
                candidates.extend(proposed);
            }
            if candidates.is_empty() {
                return Err(
                    "no candidate attributes: pass --candidate \"<attribute>\" (repeatable) and/or --propose <n>"
                        .into(),
                );
            }

            let cache_store = if no_cache {
                None
            } else {
                let cache_path = cache.unwrap_or_else(SqlitePairwiseCache::default_path);
                Some(SqlitePairwiseCache::new(cache_path)?)
            };
            let mut execution = llmsort::rerank::RerankExecution::new(
                gateway.clone(),
                Attribution::new("llmsort::explain"),
            )
            .run_options(RerankRunOptions {
                rng_seed: seed,
                cache_only: false,
            });
            if let Some(store) = cache_store.as_ref() {
                execution = execution.cache(store);
            }

            let explanation = llmsort::rerank::explain_ranking(
                documents,
                candidates,
                execution,
                llmsort::rerank::ExplainOptions {
                    model,
                    comparison_budget: budget,
                    ..Default::default()
                },
            )
            .await?;

            if format_json {
                println!("{}", serde_json::to_string_pretty(&explanation)?);
            } else {
                println!("attribute                                    | alone ρ | weight");
                println!("---------------------------------------------|---------|-------");
                for attr in &explanation.attributes {
                    let rho = attr
                        .spearman_alone
                        .map(|r| format!("{r:+.2}"))
                        .unwrap_or_else(|| "  n/a".into());
                    let prompt: String = attr.prompt.chars().take(44).collect();
                    println!("{prompt:<45}| {rho:>7} | {:.2}", attr.fitted_weight);
                }
                match explanation.combined_spearman {
                    Some(c) => println!(
                        "
weighted combination reconstructs your ranking at ρ = {c:+.2}"
                    ),
                    None => println!(
                        "
no combination of these attributes reconstructs your ranking"
                    ),
                }
            }
            let meta = &explanation.meta;
            eprintln!(
                "{} comparisons ({} cached, {} refused) · ${:.4} · order flips: {}/{}",
                meta.comparisons_used,
                meta.comparisons_cached,
                meta.comparisons_refused,
                meta.provider_cost_nanodollars as f64 / 1e9,
                meta.position_flips,
                meta.pairs_counterbalanced,
            );
        }
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
            let gateway = provider_gateway(cache_only)?;

            let (trace_sink, trace_worker) = if let Some(path) = trace {
                let (sink, worker) = JsonlTraceSink::new(path)?;
                (Some(sink), Some(worker))
            } else {
                (None, None)
            };
            let trace_ref = trace_sink.as_ref().map(|sink| sink as &dyn TraceSink);

            let mut execution = llmsort::rerank::RerankExecution::new(
                Arc::new(gateway),
                Attribution::new("llmsort::rerank"),
            )
            .cache(&cache)
            .run_options(options);
            if let Some(policy) = policy_obj.clone() {
                execution = execution.model_policy(policy);
            }
            if let Some(trace) = trace_ref {
                execution = execution.trace(trace);
            }

            let resp = llmsort::rerank::multi_rerank(req.clone(), execution).await?;

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

/// Resolve a judge item argument: literal text, or `@path` file contents.
fn read_item_arg(raw: &str) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(path) = raw.strip_prefix('@') {
        std::fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}").into())
    } else {
        Ok(raw.to_string())
    }
}

/// Read raw sort input from a file or stdin (`-` or omitted).
fn read_sort_input(file: Option<&std::path::Path>) -> Result<String, Box<dyn std::error::Error>> {
    match file {
        Some(path) if path.as_os_str() != "-" => std::fs::read_to_string(path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()).into()),
        _ => {
            let mut raw = String::new();
            io::Read::read_to_string(&mut io::stdin(), &mut raw)?;
            Ok(raw)
        }
    }
}

/// Parse sort input: newline-delimited plain text, or a JSON array of strings
/// or `{"id", "text"}` objects when the first non-whitespace byte is `[`.
fn parse_sort_items(
    raw: &str,
) -> Result<Vec<llmsort::rerank::RerankDocument>, Box<dyn std::error::Error>> {
    use llmsort::rerank::RerankDocument;

    if raw.trim_start().starts_with('[') {
        let value: serde_json::Value = serde_json::from_str(raw)
            .map_err(|err| format!("input looks like JSON but failed to parse: {err}"))?;
        let arr = value
            .as_array()
            .ok_or("JSON input must be an array of strings or {id, text} objects")?;
        let mut documents = Vec::with_capacity(arr.len());
        for (idx, elem) in arr.iter().enumerate() {
            if let Some(text) = elem.as_str() {
                documents.push(RerankDocument {
                    id: format!("item-{idx:04}"),
                    text: text.to_string(),
                });
            } else if let Some(obj) = elem.as_object() {
                let text = obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("JSON element {idx} needs a string \"text\" field"))?;
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("item-{idx:04}"));
                documents.push(RerankDocument {
                    id,
                    text: text.to_string(),
                });
            } else {
                return Err(format!(
                    "JSON element {idx} must be a string or an object with a \"text\" field"
                )
                .into());
            }
        }
        Ok(documents)
    } else {
        Ok(raw
            .lines()
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(idx, line)| RerankDocument {
                id: format!("item-{idx:04}"),
                text: line.to_string(),
            })
            .collect())
    }
}

/// Render sorted output in the requested format.
fn render_sorted(
    out: &mut impl Write,
    sorted: &llmsort::rerank::SortedTexts,
    format: SortFormatArg,
    scores: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        SortFormatArg::Text => {
            for item in &sorted.items {
                if scores {
                    writeln!(
                        out,
                        "{:.3}\u{b1}{:.3}\t{}",
                        item.latent_mean, item.latent_std, item.text
                    )?;
                } else {
                    writeln!(out, "{}", item.text)?;
                }
            }
        }
        SortFormatArg::Json => {
            serde_json::to_writer_pretty(&mut *out, sorted)?;
            writeln!(out)?;
        }
        SortFormatArg::Jsonl => {
            for item in &sorted.items {
                serde_json::to_writer(&mut *out, item)?;
                writeln!(out)?;
            }
        }
        SortFormatArg::Csv => {
            writeln!(
                out,
                "rank,id,latent_mean,latent_std,z_score,percentile,text"
            )?;
            for item in &sorted.items {
                writeln!(
                    out,
                    "{},{},{:.6},{:.6},{:.6},{:.6},{}",
                    item.rank,
                    csv_field(&item.id),
                    item.latent_mean,
                    item.latent_std,
                    item.z_score,
                    item.percentile,
                    csv_field(&item.text),
                )?;
            }
        }
    }
    Ok(())
}

/// Quote a CSV field when it contains a comma, quote, or newline.
fn csv_field(raw: &str) -> String {
    if raw.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", raw.replace('"', "\"\""))
    } else {
        raw.to_string()
    }
}

fn provider_gateway(
    cache_only: bool,
) -> Result<ProviderGateway<NoopUsageSink>, Box<dyn std::error::Error>> {
    if cache_only {
        let adapter = llmsort::gateway::openrouter::OpenRouterAdapter::with_config(
            "cache-only",
            "http://127.0.0.1:9",
            std::time::Duration::from_secs(1),
            None,
            None,
        )?;
        return Ok(ProviderGateway::with_config(
            adapter,
            Arc::new(NoopUsageSink),
            llmsort::gateway::GatewayConfig::default(),
        ));
    }

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        return Err("OPENROUTER_API_KEY is not set. Create a key at \
             https://openrouter.ai/keys and `export OPENROUTER_API_KEY=...`, \
             or use --cache-only to replay cached judgements."
            .into());
    }

    Ok(ProviderGateway::from_env(Arc::new(NoopUsageSink))?)
}

fn require_openrouter_key() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        return Err("OPENROUTER_API_KEY is not set. Create a key at \
             https://openrouter.ai/keys and `export OPENROUTER_API_KEY=...`."
            .into());
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
