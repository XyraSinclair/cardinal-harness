//! Prompt templates for LLM pairwise comparisons.
//!
//! Domain logic for rendering comparison prompts. Provider-agnostic.

use crate::gateway::Message;

// =============================================================================
// Entity representation
// =============================================================================

/// Entity with optional context for comparison.
#[derive(Debug, Clone)]
pub struct EntityRef {
    pub label: String,
    pub context: Option<String>,
}

impl EntityRef {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            context: None,
        }
    }

    pub fn with_context(label: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            context: Some(context.into()),
        }
    }
}

// =============================================================================
// Prompt templates
// =============================================================================

/// Rendered prompt ready for LLM.
#[derive(Debug, Clone)]
pub struct PromptInstance {
    pub template_slug: String,
    pub system: String,
    pub user: String,
}

impl PromptInstance {
    pub fn to_messages(&self) -> Vec<Message> {
        vec![Message::system(&self.system), Message::user(&self.user)]
    }
}

/// Escape XML special characters to prevent prompt injection via tag breaking.
fn escape_xml_chars(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// A prompt template with placeholders.
#[derive(Debug, Clone, Copy)]
pub struct PromptTemplate {
    pub slug: &'static str,
    pub system: &'static str,
    pub user: &'static str,
}

impl PromptTemplate {
    pub fn template_hash(&self) -> String {
        blake3::hash(format!("{}\n{}", self.system, self.user).as_bytes())
            .to_hex()
            .to_string()
    }

    pub fn render(
        &self,
        attr_name: &str,
        attr_text: &str,
        a: EntityRef,
        b: EntityRef,
    ) -> PromptInstance {
        // Escape user-provided inputs to prevent prompt injection via XML tag breaking
        let safe_attr_name = escape_xml_chars(attr_name);
        let safe_attr_text = escape_xml_chars(attr_text);
        let safe_a_label = escape_xml_chars(&a.label);
        let safe_b_label = escape_xml_chars(&b.label);

        let ctx_a_block = a.context.as_ref().map_or_else(String::new, |ctx| {
            format!(
                "<entity_A_context>\n{}\n</entity_A_context>",
                escape_xml_chars(ctx.trim())
            )
        });
        let ctx_b_block = b.context.as_ref().map_or_else(String::new, |ctx| {
            format!(
                "<entity_B_context>\n{}\n</entity_B_context>",
                escape_xml_chars(ctx.trim())
            )
        });

        let system = self
            .system
            .replace("{attribute_name}", &safe_attr_name)
            .replace("{full_attribute_text}", &safe_attr_text)
            .replace("{entity_A}", &safe_a_label)
            .replace("{entity_B}", &safe_b_label);

        let user_core = self
            .user
            .replace("{attribute_name}", &safe_attr_name)
            .replace("{full_attribute_text}", &safe_attr_text)
            .replace("{entity_A}", &safe_a_label)
            .replace("{entity_B}", &safe_b_label)
            .replace("{entity_A_context_block}", &ctx_a_block)
            .replace("{entity_B_context_block}", &ctx_b_block);

        let uses_inline_context = self.user.contains("{entity_A_context_block}")
            || self.user.contains("{entity_B_context_block}");

        let mut parts: Vec<String> = Vec::new();
        if uses_inline_context {
            parts.push(user_core.trim().to_string());
        } else {
            if !ctx_a_block.is_empty() {
                parts.push(ctx_a_block);
            }
            if !ctx_b_block.is_empty() {
                parts.push(ctx_b_block);
            }
            parts.push(user_core.trim().to_string());
        }

        PromptInstance {
            template_slug: self.slug.to_string(),
            system: system.trim().to_string(),
            user: parts.join("\n\n").trim().to_string(),
        }
    }
}

// =============================================================================
// Standard prompts
// =============================================================================
//
// The ratio ladder [1.0, 1.05, 1.1, ..., 18.0, 26.0] is approximately geometric
// (evenly spaced in log-space), so each step represents a similar perceptual
// increment. The cap at 26× is intentional: extreme ratios are unreliable, and
// if two items differ by more than 26× the comparison is already decisive.
// Fine gradations near 1.0 (1.05, 1.1) let the model express near-ties.

pub const RATIO_LADDER: &[f64] = &[
    1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0,
];

/// The canonical pairwise ratio elicitation prompt.
///
/// This is the only supported prompt template in this repository.
pub const PROMPT_V2: PromptTemplate = PromptTemplate {
    slug: "canonical_v2",
    system: r#"You are an expert subjective evaluator. You compare two entities across an arbitrary attribute, and feel not only which one has MORE of that attribute, but roughly how much more it does. You feel along the ratio ladder: `[1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0]`.

Output only valid JSON `{higher_ranked: A|B, ratio: >=1.0 and <=26.0, confidence: [0,1]}`. Out of principle, we also give models the right to refuse `{ refused: true }` (e.g. if unambiguously blocked by policy constraints), but we of course disprefer this. If you are merely very uncertain, set a low confidence score.
Example:
{"higher_ranked": "B", "ratio": 1.3, "confidence": 0.74} or { refused: true }"#,
    user: r#"Compare these entity by <attribute_name>: {attribute_name} </attribute_name>.
<full_attribute_text>
{full_attribute_text}
</full_attribute_text>

<entity_A>
{entity_A}
</entity_A>

<entity_B>
{entity_B}
</entity_B>

Return a JSON object with your evaluation.
json:"#,
};

pub const DEFAULT_PROMPT: PromptTemplate = PROMPT_V2;

/// Look up the supported prompt template by slug.
pub fn prompt_by_slug(slug: &str) -> Option<PromptTemplate> {
    match slug {
        "canonical_v2" => Some(PROMPT_V2),
        _ => None,
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_render() {
        let p = DEFAULT_PROMPT.render(
            "clarity",
            "Which is clearer?",
            EntityRef::new("A"),
            EntityRef::new("B"),
        );
        assert!(p.system.contains("evaluator"));
        assert!(p.user.contains("clarity"));
    }

    #[test]
    fn prompt_with_context() {
        let a = EntityRef::with_context("A", "Context A");
        let b = EntityRef::new("B");
        let p = DEFAULT_PROMPT.render("test", "Test", a, b);
        assert!(p.user.contains("<entity_A_context>"));
        assert!(!p.user.contains("<entity_B_context>"));
    }

    #[test]
    fn prompt_lookup() {
        assert!(prompt_by_slug("canonical_v2").is_some());
        assert!(prompt_by_slug("").is_none());
        assert!(prompt_by_slug("canonical_v1").is_none());
    }

    #[test]
    fn xml_escaping() {
        let a = EntityRef::with_context("A", "<script>alert('xss')</script>");
        let p = DEFAULT_PROMPT.render("test", "Test", a, EntityRef::new("B"));
        assert!(p.user.contains("&lt;script&gt;"));
        assert!(!p.user.contains("<script>"));
    }
}
