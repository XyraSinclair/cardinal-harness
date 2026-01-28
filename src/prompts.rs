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
            .replace("{entity_B}", &safe_b_label);

        let mut parts: Vec<String> = Vec::new();
        if let Some(ctx) = &a.context {
            parts.push(format!(
                "<entity_A_context>\n{}\n</entity_A_context>",
                escape_xml_chars(ctx.trim())
            ));
        }
        if let Some(ctx) = &b.context {
            parts.push(format!(
                "<entity_B_context>\n{}\n</entity_B_context>",
                escape_xml_chars(ctx.trim())
            ));
        }
        parts.push(user_core.trim().to_string());

        PromptInstance {
            template_slug: self.slug.to_string(),
            system: system.trim().to_string(),
            user: parts.join("\n\n"),
        }
    }
}

// =============================================================================
// Standard prompts
// =============================================================================

pub const PROMPT_V1: PromptTemplate = PromptTemplate {
    slug: "canonical_v1",
    system: r#"You are an expert subjective evaluator. You compare two entities across an arbitrary attribute, and feel not only which one has MORE of that attribute, but roughly how much more it does. You feel along the ratio ladder: `[1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.1, 2.5, 3.1, 3.9, 5.1, 6.8, 9.2, 12.7, 18.0, 26.0]`.

Output only valid JSON with higher_ranked, ratio (>=1.0 and <=26.0), and confidence (0.5-1.0).
Example:
{"higher_ranked": "B", "ratio": 1.3, "confidence": 0.74}"#,
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

pub const PROMPT_V3: PromptTemplate = PromptTemplate {
    slug: "canonical_v3",
    system: r#"You are an expert subjective evaluator comparing two entities on one attribute. Decide which has more of it and how much more using ratio ladder R=[1,1.05,1.1,1.2,1.3,1.5,1.75,2.1,2.5,3.1,3.9,5.1,6.8,9.2,12.7,18,26].
Return only JSON: {higher_ranked:A|B,ratio:1-26,confidence:0..1}. If policy-blocked, return {refused:true}. If uncertain, lower confidence."#,
    user: r#"Compare by <attribute_name>{attribute_name}</attribute_name>.
<full_attribute_text>{full_attribute_text}</full_attribute_text>
<entity_A>{entity_A}</entity_A>
<entity_B>{entity_B}</entity_B>

json:"#,
};

pub const PROMPTS: &[PromptTemplate] = &[PROMPT_V1, PROMPT_V2, PROMPT_V3];
pub const DEFAULT_PROMPT: PromptTemplate = PROMPT_V2;

pub fn prompt_by_slug(slug: &str) -> Option<PromptTemplate> {
    PROMPTS.iter().find(|t| t.slug == slug).copied()
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
        assert!(prompt_by_slug("canonical_v1").is_some());
        assert!(prompt_by_slug("nonexistent").is_none());
    }

    #[test]
    fn xml_escaping() {
        let a = EntityRef::with_context("A", "<script>alert('xss')</script>");
        let p = DEFAULT_PROMPT.render("test", "Test", a, EntityRef::new("B"));
        assert!(p.user.contains("&lt;script&gt;"));
        assert!(!p.user.contains("<script>"));
    }
}
