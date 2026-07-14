//! Slate: the intake pipeline's front door.
//!
//! A person sees something in the wild, rambles a note about why it's
//! interesting, and wants it indexed in the space of attributes that are
//! alive in social reality. The pipeline: (1) name the stakeholders whose
//! judgments would matter for this item (given, or proposed by the
//! model); (2) each stakeholder — simulated separately, blind to the
//! others — proposes the attributes THEY would rate it by; (3) one merge
//! pass clusters near-duplicates into a slate, each attribute carrying
//! its BACKERS.
//!
//! Breadth (how many independent perspectives converge on an attribute)
//! is the measurable proxy for "alive in social reality" — a slate
//! attribute backed by four unlike stakeholders is a different object
//! from one backed by one. The slate is a hypothesis set, priced in
//! pennies; measurement is the existing machinery's job (`canonize` for
//! transmissibility, `sort`/`distinguish` for the ratings, `explain` for
//! the inverse mode where a believed ranking comes first).

use serde::Serialize;

use crate::gateway::{Attribution, ChatGateway, ChatModel, ChatRequest, Message};

/// One slate entry: an attribute and the stakeholders who want it.
#[derive(Debug, Clone, Serialize)]
pub struct SlateEntry {
    pub attribute: String,
    pub backers: Vec<String>,
}

/// Result of [`propose_slate`].
#[derive(Debug, Serialize)]
pub struct SlateReport {
    pub stakeholders: Vec<String>,
    /// Attributes sorted by breadth (backer count) descending.
    pub slate: Vec<SlateEntry>,
    pub calls: usize,
    pub cost_nanodollars: i64,
}

/// Errors from [`propose_slate`].
#[derive(Debug, thiserror::Error)]
pub enum SlateError {
    #[error("provider error: {0}")]
    Provider(#[from] crate::gateway::error::ProviderError),
    #[error("could not parse model output: {0}")]
    Parse(String),
}

async fn chat_json(
    gateway: &dyn ChatGateway,
    model: &str,
    system: &str,
    user: String,
    attribution: &Attribution,
) -> Result<(serde_json::Value, i64), SlateError> {
    let response = gateway
        .chat(ChatRequest {
            model: ChatModel::openrouter(model),
            messages: vec![Message::system(system), Message::user(user)],
            temperature: 0.6,
            max_tokens: Some(900),
            json_mode: true,
            attribution: attribution.clone(),
            logprobs: false,
            top_logprobs: None,
            reasoning: None,
            prompt_cache_key: None,
        })
        .await?;
    let content = response.content.trim();
    let start = content.find(['[', '{']).unwrap_or(0);
    let end = content
        .rfind([']', '}'])
        .map(|e| e + 1)
        .unwrap_or(content.len());
    let value: serde_json::Value = serde_json::from_str(&content[start..end])
        .map_err(|err| SlateError::Parse(format!("{err}: {content}")))?;
    Ok((value, response.cost_nanodollars))
}

fn string_array(value: &serde_json::Value) -> Result<Vec<String>, SlateError> {
    // json_mode output varies: a bare array, an object wrapping one, or an
    // object whose VALUES are the strings. Accept all three; strings may
    // also arrive as objects with a single string value.
    let element = |v: &serde_json::Value| -> Option<String> {
        v.as_str().map(str::to_string).or_else(|| {
            v.as_object()
                .and_then(|o| o.values().find_map(serde_json::Value::as_str))
                .map(str::to_string)
        })
    };
    if let Some(array) = value.as_array().or_else(|| {
        value
            .as_object()
            .and_then(|o| o.values().find_map(serde_json::Value::as_array))
    }) {
        return array
            .iter()
            .map(|v| {
                element(v)
                    .ok_or_else(|| SlateError::Parse(format!("non-string element in {value}")))
            })
            .collect();
    }
    if let Some(object) = value.as_object() {
        let strings: Vec<String> = object
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if !strings.is_empty() {
            return Ok(strings);
        }
    }
    Err(SlateError::Parse(format!("no string array in {value}")))
}

/// Build an attribute slate for one item.
///
/// `stakeholders`: pass the perspectives that matter, or empty to have
/// the model propose `stakeholder_count` of them from the item itself.
/// `note`: the collector's voice-note ramble, verbatim — it carries the
/// human's felt sense of why this item is interesting.
pub async fn propose_slate(
    gateway: &dyn ChatGateway,
    model: &str,
    item: &str,
    note: Option<&str>,
    mut stakeholders: Vec<String>,
    stakeholder_count: usize,
    per_stakeholder: usize,
) -> Result<SlateReport, SlateError> {
    let attribution = Attribution::new("cardinal::slate");
    let mut calls = 0usize;
    let mut cost = 0i64;
    let note_block = note
        .map(|n| format!("\n\n<collector_note>\n{n}\n</collector_note>"))
        .unwrap_or_default();

    // Stage 0: name the stakeholders, if none given.
    if stakeholders.is_empty() {
        let system = "You name stakeholders. Given an item someone collected \
            from the world, name the distinct kinds of people or perspectives \
            for whom judging this item well genuinely matters — unlike each \
            other, concrete, alive in current social reality (not generic \
            roles like 'expert'). Output only JSON of the form {\"items\": [name, ...]} with short \
            stakeholder names (2-5 words each).";
        let user = format!(
            "<item>\n{item}\n</item>{note_block}\n\nName exactly \
             {stakeholder_count} stakeholders.\n\njson:"
        );
        let (value, c) = chat_json(gateway, model, system, user, &attribution).await?;
        calls += 1;
        cost += c;
        stakeholders = string_array(&value)?;
    }

    // Stage 1: each stakeholder proposes, blind to the others.
    let mut proposals: Vec<(String, Vec<String>)> = Vec::new();
    for stakeholder in &stakeholders {
        let system = format!(
            "You are exactly this stakeholder: {stakeholder}. You judge items \
             by what actually matters to someone in your position. Propose the \
             attributes YOU would want this item rated by — each a short, \
             judgeable criterion phrase (2-8 words) usable in the question \
             'which of these two items has more of X?'. Genuinely different \
             from each other. Output only JSON of the form {{\"items\": [attribute, ...]}}."
        );
        let user = format!(
            "<item>\n{item}\n</item>{note_block}\n\nPropose exactly \
             {per_stakeholder} attributes.\n\njson:"
        );
        let (value, c) = chat_json(gateway, model, &system, user, &attribution).await?;
        calls += 1;
        cost += c;
        proposals.push((stakeholder.clone(), string_array(&value)?));
    }

    // Stage 2: one merge pass clusters near-duplicates, preserving backers.
    let listing = proposals
        .iter()
        .map(|(who, attrs)| format!("{who}: {}", attrs.join(" | ")))
        .collect::<Vec<_>>()
        .join("\n");
    let system = "You merge attribute slates. Different stakeholders proposed \
        attributes for the same item; cluster ONLY true rewordings of the same \
        underlying dimension into one attribute with the best wording, and \
        record which stakeholders backed each cluster. Merging is the \
        exception: genuinely different dimensions stay separate, and a typical \
        output keeps most of the input attributes. Output only JSON of the form {\"slate\": [{\"attribute\": string, \
        \"backers\": [stakeholder names]}, ...]}, one object per surviving \
        dimension.";
    let user = format!("<proposals>\n{listing}\n</proposals>\n\njson:");
    let (value, c) = chat_json(gateway, model, system, user, &attribution).await?;
    calls += 1;
    cost += c;

    // Merge output shapes seen in the wild: a bare array of entries; an
    // object wrapping such an array; or a SINGLE entry object (whose
    // backers array must not be mistaken for the entry list).
    let single;
    let merged: &[serde_json::Value] = if let Some(a) = value.as_array() {
        a
    } else if value.get("attribute").is_some() || value.get("name").is_some() {
        single = [value.clone()];
        &single
    } else if let Some(a) = value.as_object().and_then(|o| {
        o.values()
            .find(|v| {
                v.as_array()
                    .is_some_and(|a| a.iter().all(serde_json::Value::is_object))
            })
            .and_then(serde_json::Value::as_array)
    }) {
        a
    } else {
        return Err(SlateError::Parse(format!(
            "merge: expected array in {value}"
        )));
    };
    let mut slate: Vec<SlateEntry> = merged
        .iter()
        .map(|v| {
            // Field-name tolerance: models vary between attribute/name/
            // wording and backers/stakeholders/supporters.
            let attribute = ["attribute", "name", "wording"]
                .iter()
                .find_map(|k| v[*k].as_str())
                .or_else(|| v.as_object()?.values().find_map(serde_json::Value::as_str))
                .ok_or_else(|| {
                    SlateError::Parse(format!("merge: no attribute in {v} (full: {value})"))
                })?
                .to_string();
            let backers = ["backers", "stakeholders", "supporters"]
                .iter()
                .find_map(|k| v[*k].as_array().map(|_| string_array(&v[*k])))
                .transpose()?
                .unwrap_or_default();
            Ok(SlateEntry { attribute, backers })
        })
        .collect::<Result<_, SlateError>>()?;
    slate.sort_by(|a, b| b.backers.len().cmp(&a.backers.len()));

    Ok(SlateReport {
        stakeholders,
        slate,
        calls,
        cost_nanodollars: cost,
    })
}
