//! Pathological-judge taxonomy.
//!
//! Each test wires a deliberately broken pairwise judge behind a wiremock
//! `OpenRouterAdapter` and drives it through the *real* sort path
//! (`sort_texts` / `sort_documents`, i.e. `simple::rerank` /
//! `multi::multi_rerank` under the hood — the same code CLI `cardinal sort`
//! uses). The claim under test in each case is a property the diagnostics (or
//! the recovered order) must exhibit no matter how the judge misbehaves:
//! the machinery must fail informatively, not silently or catastrophically.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cardinal_harness::gateway::openrouter::OpenRouterAdapter;
use cardinal_harness::gateway::{Attribution, GatewayConfig, NoopUsageSink, ProviderGateway};
use cardinal_harness::rerank::{
    sort_documents, sort_texts, RerankDocument, RerankExecution, RerankRunOptions, SortOptions,
    SortedTexts,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// =============================================================================
// Shared plumbing
// =============================================================================

fn extract_between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let start_idx = s.find(start)? + start.len();
    let rest = &s[start_idx..];
    let end_idx = rest.find(end)?;
    Some(&rest[..end_idx])
}

/// Pull the raw `<entity_A_context>` / `<entity_B_context>` bodies out of an
/// OpenRouter chat-completion request, as every prompt template embeds them.
fn contexts(request: &Request) -> (String, String) {
    let parsed: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
    let user_content = parsed
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|messages| {
            messages
                .iter()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .map(str::to_string)
        })
        .unwrap_or_default();
    let a = extract_between(&user_content, "<entity_A_context>", "</entity_A_context>")
        .unwrap_or("")
        .trim()
        .to_string();
    let b = extract_between(&user_content, "<entity_B_context>", "</entity_B_context>")
        .unwrap_or("")
        .trim()
        .to_string();
    (a, b)
}

/// Score a context string by which of a best-to-worst tag list it contains.
/// First tag = highest score. Zero if none match.
fn tag_score(ctx: &str, tags_best_to_worst: &[&str]) -> i32 {
    tags_best_to_worst
        .iter()
        .position(|t| ctx.contains(t))
        .map(|idx| (tags_best_to_worst.len() - idx) as i32)
        .unwrap_or(0)
}

fn decisive_json(a_score: i32, b_score: i32, strong: f64, weak: f64, confidence: f64) -> String {
    let (higher, ratio) = if a_score >= b_score {
        ("A", if a_score - b_score >= 2 { strong } else { weak })
    } else {
        ("B", if b_score - a_score >= 2 { strong } else { weak })
    };
    format!(r#"{{"higher_ranked":"{higher}","ratio":{ratio},"confidence":{confidence}}}"#)
}

fn chat_ok_body(content: &str) -> serde_json::Value {
    json!({
        "choices": [{
            "message": { "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 10 }
    })
}

fn gateway_for(server: &MockServer) -> Arc<ProviderGateway<NoopUsageSink>> {
    // A generous timeout: these are in-process wiremock calls that normally
    // resolve in milliseconds, but this suite runs against a shared,
    // non-isolated CI/dev host where sibling agents' concurrent builds can
    // starve the scheduler for several seconds at a time. A short timeout
    // here doesn't test anything about judge behavior — it just risks
    // spuriously turning every in-flight comparison into a transport error
    // (`comparisons_used == 0`) under host contention, which is a test-harness
    // flake, not a finding about the code under test.
    let adapter = OpenRouterAdapter::with_config(
        "sk-test",
        server.uri(),
        Duration::from_secs(20),
        None,
        None,
    )
    .expect("adapter");
    Arc::new(ProviderGateway::with_config(
        adapter,
        Arc::new(NoopUsageSink),
        GatewayConfig {
            max_retries: 0,
            retry_base_delay: Duration::from_millis(0),
        },
    ))
}

fn base_opts(budget: usize) -> SortOptions {
    SortOptions {
        model: Some("test/judge".into()),
        comparison_budget: Some(budget),
        comparison_concurrency: Some(1),
        ..Default::default()
    }
}

fn doc(id: &str, text: &str) -> RerankDocument {
    RerankDocument {
        id: id.to_string(),
        text: text.to_string(),
    }
}

fn ids_by_rank(sorted: &SortedTexts) -> Vec<String> {
    let mut items: Vec<&cardinal_harness::rerank::SortedItem> = sorted.items.iter().collect();
    items.sort_by_key(|i| i.rank);
    items.into_iter().map(|i| i.id.clone()).collect()
}

fn mean_std(items: &[cardinal_harness::rerank::SortedItem]) -> f64 {
    items.iter().map(|i| i.latent_std).sum::<f64>() / items.len() as f64
}

fn latent_mean_of(items: &[cardinal_harness::rerank::SortedItem], id: &str) -> f64 {
    items
        .iter()
        .find(|i| i.id == id)
        .unwrap_or_else(|| panic!("missing item {id}"))
        .latent_mean
}

// =============================================================================
// 1. PURE POSITION BIAS
// =============================================================================

/// A judge with pure position bias: always prefers whichever entity is
/// presented as "A", regardless of content.
#[derive(Clone, Copy)]
struct AlwaysAJudge;

impl Respond for AlwaysAJudge {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(chat_ok_body(
            r#"{"higher_ranked":"A","ratio":2.5,"confidence":0.9}"#,
        ))
    }
}

const METAL_TAGS: [&str; 4] = ["GOLD", "SILVER", "BRONZE", "TIN"];

fn metal_docs(order: &[&str]) -> Vec<RerankDocument> {
    order
        .iter()
        .map(|tag| doc(&tag.to_lowercase(), &format!("a {tag} coloured trinket")))
        .collect()
}

/// Claim: with counterbalance on, a judge with zero real signal and pure
/// position bias must disagree with itself on literally every
/// counterbalanced pair (each pair is asked A/B then B/A, and "always A"
/// flips its answer both times), so `position_flips == pairs_counterbalanced`.
#[tokio::test]
async fn pure_position_bias_flips_every_counterbalanced_pair() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(AlwaysAJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::pos_bias"));

    let sorted = sort_documents(
        metal_docs(&METAL_TAGS),
        "shininess",
        execution,
        base_opts(48),
    )
    .await
    .unwrap();

    assert!(
        sorted.meta.pairs_counterbalanced > 0,
        "expected at least one counterbalanced pair, meta={:?}",
        sorted.meta
    );
    assert_eq!(
        sorted.meta.position_flips, sorted.meta.pairs_counterbalanced,
        "pure position bias must flip every counterbalanced pair: meta={:?}",
        sorted.meta
    );
}

/// Claim: the final ordering produced against a zero-signal, pure-position
/// judge must not simply echo whatever order the caller happened to submit
/// the items in. We drive two input permutations of the same identities and
/// require either (a) they land on the same order (the counterbalanced
/// contradictory observations cancel identically both times), or (b) both
/// runs are recognizably uninformative (latent means collapse near each
/// other) rather than each parroting its own input order.
#[tokio::test]
async fn pure_position_bias_order_is_not_a_function_of_input_order() {
    let server_a = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(AlwaysAJudge)
        .mount(&server_a)
        .await;
    let server_b = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(AlwaysAJudge)
        .mount(&server_b)
        .await;

    let order_a = ["gold", "silver", "bronze", "tin"];
    let order_b = ["tin", "bronze", "silver", "gold"];
    let docs_a: Vec<RerankDocument> = order_a
        .iter()
        .map(|id| doc(id, &format!("a {} coloured trinket", id.to_uppercase())))
        .collect();
    let docs_b: Vec<RerankDocument> = order_b
        .iter()
        .map(|id| doc(id, &format!("a {} coloured trinket", id.to_uppercase())))
        .collect();

    let opts = || SortOptions {
        model: Some("test/judge".into()),
        comparison_budget: Some(48),
        comparison_concurrency: Some(1),
        ..Default::default()
    };

    let run_a = sort_documents(
        docs_a,
        "shininess",
        RerankExecution::new(gateway_for(&server_a), Attribution::new("test::pos_bias_a"))
            .run_options(RerankRunOptions {
                rng_seed: Some(1234),
                ..Default::default()
            }),
        opts(),
    )
    .await
    .unwrap();
    let run_b = sort_documents(
        docs_b,
        "shininess",
        RerankExecution::new(gateway_for(&server_b), Attribution::new("test::pos_bias_b"))
            .run_options(RerankRunOptions {
                rng_seed: Some(1234),
                ..Default::default()
            }),
        opts(),
    )
    .await
    .unwrap();

    let mut order_by_id_a: Vec<String> = run_a.items.iter().map(|i| i.id.clone()).collect();
    order_by_id_a.sort_by_key(|id| run_a.items.iter().find(|i| &i.id == id).unwrap().rank);
    let mut order_by_id_b: Vec<String> = run_b.items.iter().map(|i| i.id.clone()).collect();
    order_by_id_b.sort_by_key(|id| run_b.items.iter().find(|i| &i.id == id).unwrap().rank);

    let orders_agree = order_by_id_a == order_by_id_b;

    let spread = |items: &[cardinal_harness::rerank::SortedItem]| -> f64 {
        let means: Vec<f64> = items.iter().map(|i| i.latent_mean).collect();
        means.iter().cloned().fold(f64::MIN, f64::max)
            - means.iter().cloned().fold(f64::MAX, f64::min)
    };
    // A judge with a genuine 4-tier signal recovered by a truthful judge
    // spreads latent means by several units (see the scale-compressed and
    // metal-judge tests below); a pure-position judge whose counterbalanced
    // observations cancel should stay far tighter than that.
    const UNINFORMATIVE_SPREAD: f64 = 1.5;
    let both_uninformative =
        spread(&run_a.items) < UNINFORMATIVE_SPREAD && spread(&run_b.items) < UNINFORMATIVE_SPREAD;

    assert!(
        orders_agree || both_uninformative,
        "pure position bias must either agree across input permutations or be \
         recognizably uninformative in both; order_a={order_by_id_a:?} order_b={order_by_id_b:?} \
         spread_a={:.3} spread_b={:.3}",
        spread(&run_a.items),
        spread(&run_b.items)
    );
}

// =============================================================================
// 2. INTRANSITIVE JUDGE
// =============================================================================

/// A judge that is decisive everywhere except a designated 3-item cycle,
/// where it enforces cycle_a > cycle_b > cycle_c > cycle_a by construction,
/// with a real (non-random) ratio each time.
#[derive(Clone, Copy)]
struct IntransitiveJudge;

fn cycle_tag(ctx: &str) -> Option<&'static str> {
    if ctx.contains("CYCLE_A") {
        Some("CYCLE_A")
    } else if ctx.contains("CYCLE_B") {
        Some("CYCLE_B")
    } else if ctx.contains("CYCLE_C") {
        Some("CYCLE_C")
    } else {
        None
    }
}

impl Respond for IntransitiveJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let (a_ctx, b_ctx) = contexts(request);
        if let (Some(a_cycle), Some(b_cycle)) = (cycle_tag(&a_ctx), cycle_tag(&b_ctx)) {
            // Both inside the cycle: enforce A > B > C > A regardless of
            // which side of the prompt each landed on.
            let beats = |x: &str, y: &str| -> bool {
                matches!(
                    (x, y),
                    ("CYCLE_A", "CYCLE_B") | ("CYCLE_B", "CYCLE_C") | ("CYCLE_C", "CYCLE_A")
                )
            };
            let higher = if beats(a_cycle, b_cycle) { "A" } else { "B" };
            let content =
                format!(r#"{{"higher_ranked":"{higher}","ratio":1.6,"confidence":0.85}}"#);
            return ResponseTemplate::new(200).set_body_json(chat_ok_body(&content));
        }
        // Decisive elsewhere: CLEAR_HIGH > any cycle item > CLEAR_LOW.
        let score = |ctx: &str| -> i32 {
            if ctx.contains("CLEAR_HIGH") {
                2
            } else if cycle_tag(ctx).is_some() {
                0
            } else {
                -2
            }
        };
        let a_score = score(&a_ctx);
        let b_score = score(&b_ctx);
        let content = decisive_json(a_score, b_score, 4.0, 4.0, 0.9);
        ResponseTemplate::new(200).set_body_json(chat_ok_body(&content))
    }
}

fn cycle_docs() -> Vec<RerankDocument> {
    vec![
        doc("high", "a CLEAR_HIGH quality specimen"),
        doc("cycle_a", "an item, CYCLE_A variant"),
        doc("cycle_b", "an item, CYCLE_B variant"),
        doc("cycle_c", "an item, CYCLE_C variant"),
        doc("low", "a CLEAR_LOW quality specimen"),
    ]
}

/// Claim: an intransitive judge (a 3-cycle embedded in otherwise-decisive
/// comparisons) must not make the run panic or hang. It must still produce
/// a complete, uniquely-ranked total order over every item and spend a
/// nonzero number of comparisons.
#[tokio::test]
async fn intransitive_judge_completes_and_produces_total_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(IntransitiveJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::intransitive"));

    let sorted = sort_documents(cycle_docs(), "quality", execution, base_opts(80))
        .await
        .unwrap();

    assert!(sorted.meta.comparisons_used > 0);
    let mut ranks: Vec<usize> = sorted.items.iter().map(|i| i.rank).collect();
    ranks.sort_unstable();
    assert_eq!(
        ranks,
        vec![1, 2, 3, 4, 5],
        "expected a total order, got {ranks:?}"
    );
    for item in &sorted.items {
        assert!(item.latent_mean.is_finite());
        assert!(item.latent_std.is_finite());
    }
}

/// Claim: the robust solver averages through an intransitive cycle rather
/// than exploding — the three cyclic items' latent means must sit closer to
/// each other than any of them sits to either clearly-decisive anchor item.
#[tokio::test]
async fn intransitive_cycle_items_cluster_together() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(IntransitiveJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::intransitive_cluster"));

    let sorted = sort_documents(cycle_docs(), "quality", execution, base_opts(80))
        .await
        .unwrap();

    let high = latent_mean_of(&sorted.items, "high");
    let low = latent_mean_of(&sorted.items, "low");
    let a = latent_mean_of(&sorted.items, "cycle_a");
    let b = latent_mean_of(&sorted.items, "cycle_b");
    let c = latent_mean_of(&sorted.items, "cycle_c");
    assert!(
        high > low,
        "sanity: anchors must still be well ordered ({high} vs {low})"
    );

    let cycle_spread = [a, b, c].iter().cloned().fold(f64::MIN, f64::max)
        - [a, b, c].iter().cloned().fold(f64::MAX, f64::min);
    let cycle_avg = (a + b + c) / 3.0;
    let dist_to_high = (high - cycle_avg).abs();
    let dist_to_low = (cycle_avg - low).abs();

    assert!(
        cycle_spread < dist_to_high && cycle_spread < dist_to_low,
        "cycle items must cluster tighter than their distance to either anchor: \
         spread={cycle_spread:.3} dist_to_high={dist_to_high:.3} dist_to_low={dist_to_low:.3} \
         (a={a:.3} b={b:.3} c={c:.3} high={high:.3} low={low:.3})"
    );
}

// =============================================================================
// 3. SCALE-COMPRESSED JUDGE
// =============================================================================

/// A judge that always gets the direction right but reports the smallest
/// possible ratio (1.05) no matter how large the true gap is: magnitude
/// information is destroyed, only sign survives.
#[derive(Clone, Copy)]
struct ScaleCompressedJudge;

const TIER_TAGS: [&str; 5] = ["TIER1", "TIER2", "TIER3", "TIER4", "TIER5"];

impl Respond for ScaleCompressedJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let (a_ctx, b_ctx) = contexts(request);
        let a_score = tag_score(&a_ctx, &TIER_TAGS);
        let b_score = tag_score(&b_ctx, &TIER_TAGS);
        let higher = if a_score >= b_score { "A" } else { "B" };
        let content = format!(r#"{{"higher_ranked":"{higher}","ratio":1.05,"confidence":0.6}}"#);
        ResponseTemplate::new(200).set_body_json(chat_ok_body(&content))
    }
}

fn tier_docs() -> Vec<RerankDocument> {
    TIER_TAGS
        .iter()
        .map(|tag| doc(&tag.to_lowercase(), &format!("a {tag} ranked item")))
        .collect()
}

/// Claim: direction-only information is sufficient to recover the true
/// order on a clearly separated item set, given enough budget, even though
/// every single observed ratio is compressed to 1.05.
#[tokio::test]
async fn scale_compressed_judge_recovers_order_given_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ScaleCompressedJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::scale_compressed"));

    let sorted = sort_documents(tier_docs(), "rank", execution, base_opts(120))
        .await
        .unwrap();

    let order: Vec<String> = ids_by_rank(&sorted);
    assert_eq!(
        order,
        vec!["tier1", "tier2", "tier3", "tier4", "tier5"],
        "expected true order recovered from direction-only signal, got {order:?}"
    );
}

/// Claim: magnitude degrades gracefully rather than catastrophically under a
/// tight budget — the gross polarity (best item isn't ranked last, worst
/// item isn't ranked first) must survive even when there isn't enough
/// budget to fully resolve every adjacent pair.
#[tokio::test]
async fn scale_compressed_judge_degrades_gracefully_under_tight_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ScaleCompressedJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::scale_compressed_tight"));

    let sorted = sort_documents(tier_docs(), "rank", execution, base_opts(6))
        .await
        .unwrap();

    assert!(sorted.meta.comparisons_used > 0);
    for item in &sorted.items {
        assert!(
            item.latent_mean.is_finite() && item.latent_std.is_finite(),
            "compressed ratios must not blow up the solver: {item:?}"
        );
    }
    let order = ids_by_rank(&sorted);
    assert_ne!(
        order.first().map(String::as_str),
        Some("tier5"),
        "worst item must not land first under a merely tight budget: {order:?}"
    );
    assert_ne!(
        order.last().map(String::as_str),
        Some("tier1"),
        "best item must not land last under a merely tight budget: {order:?}"
    );
}

// =============================================================================
// 4. REFUSER
// =============================================================================

/// Refuses any comparison that involves the "POISON" item; decisive
/// otherwise, ranking GOLD > SILVER > BRONZE > TIN.
#[derive(Clone, Copy)]
struct RefuserJudge;

impl Respond for RefuserJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let (a_ctx, b_ctx) = contexts(request);
        if a_ctx.contains("POISON") || b_ctx.contains("POISON") {
            return ResponseTemplate::new(200).set_body_json(chat_ok_body(r#"{"refused":true}"#));
        }
        let a_score = tag_score(&a_ctx, &METAL_TAGS);
        let b_score = tag_score(&b_ctx, &METAL_TAGS);
        let content = decisive_json(a_score, b_score, 3.9, 1.5, 0.9);
        ResponseTemplate::new(200).set_body_json(chat_ok_body(&content))
    }
}

fn poisoned_docs() -> Vec<RerankDocument> {
    vec![
        doc("poison", "a POISON tainted specimen"),
        doc("gold", "a GOLD coloured trinket"),
        doc("silver", "a SILVER coloured trinket"),
        doc("bronze", "a BRONZE coloured trinket"),
        doc("tin", "a TIN coloured trinket"),
    ]
}

/// Claim: a run against a judge that refuses every pair touching one
/// poisoned item still completes, surfaces the refusals in run metadata,
/// and still recovers the correct order among the unpoisoned items.
#[tokio::test]
async fn refuser_completes_and_orders_unpoisoned_items_correctly() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(RefuserJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::refuser"));

    let sorted = sort_documents(poisoned_docs(), "shininess", execution, base_opts(80))
        .await
        .unwrap();

    assert!(
        sorted.meta.comparisons_refused > 0,
        "expected refusals in run metadata: {:?}",
        sorted.meta
    );
    let unpoisoned_order: Vec<String> = ids_by_rank(&sorted)
        .into_iter()
        .filter(|id| id != "poison")
        .collect();
    assert_eq!(
        unpoisoned_order,
        vec!["gold", "silver", "bronze", "tin"],
        "unpoisoned items must still be ordered correctly"
    );
}

/// Claim: the refusal count in run metadata is not just nonzero but
/// *exact* — it equals the number of attempted comparisons that actually
/// touched the poisoned item, no more (no phantom refusals) and no fewer
/// (no swallowed refusals).
#[tokio::test]
async fn refuser_refused_count_matches_poisoned_pair_attempts_exactly() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(RefuserJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::refuser_exact"));

    let sorted = sort_documents(poisoned_docs(), "shininess", execution, base_opts(80))
        .await
        .unwrap();

    let received = server.received_requests().await.unwrap();
    let poisoned_attempts = received
        .iter()
        .filter(|req| {
            let (a, b) = contexts(req);
            a.contains("POISON") || b.contains("POISON")
        })
        .count();
    assert!(
        poisoned_attempts > 0,
        "test setup must actually probe the poisoned item"
    );
    assert_eq!(
        sorted.meta.comparisons_refused, poisoned_attempts,
        "every refusal must correspond 1:1 to a poisoned-pair attempt and vice versa"
    );
    assert_eq!(received.len(), sorted.meta.comparisons_attempted);
}

// =============================================================================
// 5. GASLIGHTER
// =============================================================================

/// Reports high confidence (0.99) but the direction is a pseudo-random
/// function of the pair *as presented* (not canonicalized by identity) and
/// a run seed — uncorrelated with any real signal in the text, and, unlike
/// a merely alternate-but-internally-coherent ranking, liable to disagree
/// with itself when the same logical pair is re-asked in the opposite
/// presentation order (exactly the channel `counterbalance` exists to
/// expose). A hash that canonicalizes by pair identity first was tried and
/// rejected here: it produces an internally self-consistent (if wrong)
/// total order indistinguishable in principle from truth by the data
/// alone, which tests something else entirely (see the comment on
/// `gaslighter_yields_larger_posterior_uncertainty_than_truthful_judge_on_average`
/// below for why that variant is not a fair test of calibration).
#[derive(Clone)]
struct GaslighterJudge {
    seed: u64,
}

fn presented_a_wins(a: &str, b: &str, seed: u64) -> bool {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    a.hash(&mut hasher);
    b.hash(&mut hasher);
    hasher.finish().is_multiple_of(2)
}

impl Respond for GaslighterJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let (a_ctx, b_ctx) = contexts(request);
        let a_wins = presented_a_wins(&a_ctx, &b_ctx, self.seed);
        let higher = if a_wins { "A" } else { "B" };
        let content = format!(r#"{{"higher_ranked":"{higher}","ratio":2.0,"confidence":0.99}}"#);
        ResponseTemplate::new(200).set_body_json(chat_ok_body(&content))
    }
}

/// Same item set and prompt shape, but honest: high confidence AND correct
/// direction. Isolates "wrong but confident" from "confident" as the
/// variable under test.
#[derive(Clone, Copy)]
struct TruthfulHighConfidenceJudge;

impl Respond for TruthfulHighConfidenceJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let (a_ctx, b_ctx) = contexts(request);
        let a_score = tag_score(&a_ctx, &METAL_TAGS);
        let b_score = tag_score(&b_ctx, &METAL_TAGS);
        let higher = if a_score >= b_score { "A" } else { "B" };
        let content = format!(r#"{{"higher_ranked":"{higher}","ratio":2.0,"confidence":0.99}}"#);
        ResponseTemplate::new(200).set_body_json(chat_ok_body(&content))
    }
}

/// Claim: a confidently-wrong judge whose stated direction carries no real
/// signal must, on average over a seeded ensemble of runs, leave the
/// machinery reporting *larger* posterior uncertainty than an equally
/// confident but truthful judge on the identical items and budget. If the
/// solver ever reports tight posteriors while being fed pure noise, that is
/// a real finding: confidence values must not be taken as ground truth.
#[tokio::test]
async fn gaslighter_yields_larger_posterior_uncertainty_than_truthful_judge_on_average() {
    const SEEDS: u64 = 12;
    let mut gaslighter_stds = Vec::new();
    let mut truthful_stds = Vec::new();

    for seed in 0..SEEDS {
        let docs = || metal_docs(&METAL_TAGS);

        let server_g = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(GaslighterJudge { seed })
            .mount(&server_g)
            .await;
        let run_g = sort_documents(
            docs(),
            "shininess",
            RerankExecution::new(gateway_for(&server_g), Attribution::new("test::gaslighter"))
                .run_options(RerankRunOptions {
                    rng_seed: Some(seed),
                    ..Default::default()
                }),
            base_opts(24),
        )
        .await
        .unwrap();
        gaslighter_stds.push(mean_std(&run_g.items));

        let server_t = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(TruthfulHighConfidenceJudge)
            .mount(&server_t)
            .await;
        let run_t = sort_documents(
            docs(),
            "shininess",
            RerankExecution::new(gateway_for(&server_t), Attribution::new("test::truthful"))
                .run_options(RerankRunOptions {
                    rng_seed: Some(seed),
                    ..Default::default()
                }),
            base_opts(24),
        )
        .await
        .unwrap();
        truthful_stds.push(mean_std(&run_t.items));
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let avg_gaslighter = avg(&gaslighter_stds);
    let avg_truthful = avg(&truthful_stds);

    assert!(
        avg_gaslighter.is_finite() && avg_truthful.is_finite(),
        "posterior stds must stay finite: gaslighter={gaslighter_stds:?} truthful={truthful_stds:?}"
    );
    assert!(
        avg_gaslighter > avg_truthful,
        "a confidently-wrong judge with no real signal must leave the system MORE \
         uncertain on average than a confidently-correct judge, not less: \
         avg_gaslighter={avg_gaslighter:.4} avg_truthful={avg_truthful:.4} \
         gaslighter_stds={gaslighter_stds:?} truthful_stds={truthful_stds:?}"
    );
}

// =============================================================================
// 6. FORMAT VANDAL
// =============================================================================

/// Returns a completely malformed HTTP body (not even valid JSON) on every
/// third call; answers truthfully and well-formed otherwise.
#[derive(Clone)]
struct FormatVandalJudge {
    call_count: Arc<AtomicUsize>,
}

impl Respond for FormatVandalJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
        if n.is_multiple_of(3) {
            return ResponseTemplate::new(200).set_body_string("{ this is not json at all !!");
        }
        let (a_ctx, b_ctx) = contexts(request);
        let a_score = tag_score(&a_ctx, &METAL_TAGS);
        let b_score = tag_score(&b_ctx, &METAL_TAGS);
        let content = decisive_json(a_score, b_score, 3.9, 1.5, 0.9);
        ResponseTemplate::new(200).set_body_json(chat_ok_body(&content))
    }
}

/// Claim: a judge whose transport-level responses are garbage on 1-in-3
/// calls must not corrupt the run. The failed calls must surface in the
/// run metadata as attempted-but-not-used, and the order recovered from the
/// surviving well-formed calls must still be correct given enough budget.
#[tokio::test]
async fn format_vandal_surfaces_failed_calls_and_recovers_correct_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(FormatVandalJudge {
            call_count: Arc::new(AtomicUsize::new(0)),
        })
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::format_vandal"));

    let sorted = sort_documents(
        metal_docs(&METAL_TAGS),
        "shininess",
        execution,
        base_opts(90),
    )
    .await
    .unwrap();

    assert!(
        sorted.meta.comparisons_attempted > sorted.meta.comparisons_used,
        "malformed responses must surface as attempted-but-not-used: {:?}",
        sorted.meta
    );
    let order = ids_by_rank(&sorted);
    assert_eq!(
        order,
        vec!["gold", "silver", "bronze", "tin"],
        "correct order must still be recoverable from the surviving well-formed calls"
    );
}

/// Claim: the "attempted but not used" gap tracks the actual number of
/// garbage responses the transport served (1 in 3 of everything the run
/// sent), not some unrelated accounting artifact.
#[tokio::test]
async fn format_vandal_attempted_minus_used_tracks_garbage_response_rate() {
    let counter = Arc::new(AtomicUsize::new(0));
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(FormatVandalJudge {
            call_count: counter.clone(),
        })
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::format_vandal_count"));

    let sorted = sort_documents(
        metal_docs(&METAL_TAGS),
        "shininess",
        execution,
        base_opts(90),
    )
    .await
    .unwrap();

    let received = server.received_requests().await.unwrap();
    let total_calls = received.len();
    let garbage_calls = total_calls / 3;
    assert!(
        garbage_calls > 0,
        "test setup must actually trigger at least one garbage call"
    );

    let gap = sorted.meta.comparisons_attempted - sorted.meta.comparisons_used;
    assert_eq!(
        sorted.meta.comparisons_attempted, total_calls,
        "every transport call must be counted as attempted"
    );
    // Every garbage call fails to parse and is never counted as "used"; a
    // healthy call can also fail to be "used" only via the (disjoint)
    // refusal path, which this judge never takes, so the gap must be
    // exactly the garbage-call count.
    assert_eq!(
        gap, garbage_calls,
        "attempted-used gap must equal exactly the garbage response count: \
         attempted={} used={} garbage_calls={garbage_calls} total_calls={total_calls}",
        sorted.meta.comparisons_attempted, sorted.meta.comparisons_used
    );
    assert_eq!(
        sorted.meta.comparisons_refused, 0,
        "transport-level garbage is a distinct failure mode from an honest judge refusal"
    );
}

// =============================================================================
// Cross-cutting sanity: a plain sort_texts smoke test through this file's
// own plumbing, so a bug in the shared helpers can't hide behind the more
// elaborate adversarial cases above.
// =============================================================================

/// Claim: the shared helper plumbing in this file (context extraction, tag
/// scoring, decisive-JSON formatting) correctly drives an ordinary decisive
/// judge to the expected order — a control against which the adversarial
/// results above are calibrated.
#[tokio::test]
async fn control_decisive_judge_recovers_expected_order_via_shared_helpers() {
    #[derive(Clone, Copy)]
    struct ControlJudge;
    impl Respond for ControlJudge {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let (a_ctx, b_ctx) = contexts(request);
            let a_score = tag_score(&a_ctx, &METAL_TAGS);
            let b_score = tag_score(&b_ctx, &METAL_TAGS);
            let content = decisive_json(a_score, b_score, 3.9, 1.5, 0.9);
            ResponseTemplate::new(200).set_body_json(chat_ok_body(&content))
        }
    }
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ControlJudge)
        .mount(&server)
        .await;
    let gateway = gateway_for(&server);
    let execution = RerankExecution::new(gateway, Attribution::new("test::control"));

    let sorted = sort_texts(
        vec![
            "a TIN coloured trinket".into(),
            "a GOLD coloured trinket".into(),
            "a BRONZE coloured trinket".into(),
            "a SILVER coloured trinket".into(),
        ],
        "shininess",
        execution,
        base_opts(24),
    )
    .await
    .unwrap();

    let texts: Vec<&str> = sorted.items.iter().map(|i| i.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            "a GOLD coloured trinket",
            "a SILVER coloured trinket",
            "a BRONZE coloured trinket",
            "a TIN coloured trinket",
        ]
    );
    assert_eq!(sorted.meta.comparisons_refused, 0);
    assert!(sorted.meta.comparisons_used > 0);
}
