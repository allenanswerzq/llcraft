//! # VM Schema for LLM Agent Execution
//!
//! This module provides prompt generation for the LLcraft VM.
//! Fixed prompts are in `prompts/*.md`, dynamic content uses placeholders.

use crate::session::PageIndex;
use serde::{Deserialize, Serialize};

// ============================================================================
// PROMPT TEMPLATES - Loaded from external markdown files
// ============================================================================

/// System prompt - the complete VM specification (fixed)
pub const SYSTEM_PROMPT: &str = include_str!("prompts/system.md");

/// User prompt template with placeholders: {{TASK}}, {{PAGES}}, {{TRACE}}
pub const USER_PROMPT_TEMPLATE: &str = include_str!("prompts/user.md");

// ============================================================================
// DYNAMIC CONTENT FORMATTING
// ============================================================================

/// Format available pages section for the user prompt
pub fn format_pages_section<'a>(pages: impl Iterator<Item = (&'a String, &'a PageIndex)>) -> String {
    let pages: Vec<_> = pages.collect();
    if pages.is_empty() {
        return String::new();
    }

    let total_tokens: usize = pages.iter().map(|(_, idx)| idx.tokens).sum();
    let mut out = format!(
        "\n\nAVAILABLE PAGES FROM PREVIOUS TASKS (~{} tokens total):\n",
        total_tokens
    );

    for (page_id, idx) in pages {
        out.push_str(&format!(
            "- Page '{}' (~{} tokens): {}\n",
            page_id, idx.tokens, idx.summary
        ));
    }

    out.push_str("\nIMPORTANT: Page content is NOT loaded. Use LOAD_PAGE opcode to fetch pages you need.\n");
    out
}

/// Format execution trace section for the user prompt
pub fn format_trace_section(trace: &[ExecutionStep]) -> String {
    if trace.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n## Execution History\nThese steps have already been executed:\n\n");

    for step in trace {
        if let Some(err) = &step.error {
            out.push_str(&format!("{}. {} → ERROR: {}\n", step.step, step.opcode, err));
        } else {
            out.push_str(&format!("{}. {} → {}\n", step.step, step.opcode, step.result));
        }
    }

    out.push_str("\nContinue from where execution left off.\n");
    out
}

// ============================================================================
// VmSchema - Simplified, uses external templates
// ============================================================================

/// VM Schema - provides prompt generation methods
#[derive(Debug, Clone, Default)]
pub struct VmSchema;

impl VmSchema {
    pub fn new() -> Self {
        Self
    }

    /// Get the system prompt (complete VM specification)
    pub fn system_prompt(&self) -> &'static str {
        SYSTEM_PROMPT
    }

    /// Generate the user prompt for a task with dynamic content
    pub fn user_prompt<'a>(
        &self,
        task: &str,
        pages: impl Iterator<Item = (&'a String, &'a PageIndex)>,
        trace: &[ExecutionStep],
    ) -> String {
        USER_PROMPT_TEMPLATE
            .replace("{{TASK}}", task)
            .replace("{{PAGES}}", &format_pages_section(pages))
            .replace("{{TRACE}}", &format_trace_section(trace))
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// A record of what happened in a previous execution step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    /// Step number
    pub step: usize,
    /// Opcode that was executed
    pub opcode: String,
    /// Result or outcome
    pub result: String,
    /// Any error that occurred
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_loaded() {
        let schema = VmSchema::new();
        let prompt = schema.system_prompt();

        assert!(prompt.contains("LLcraft VM"));
        assert!(prompt.contains("INFER"));
        assert!(prompt.contains("STORE"));
        assert!(prompt.contains("Context Window Management"));

        println!("System prompt length: {} chars", prompt.len());
    }

    #[test]
    fn test_user_prompt_with_placeholders() {
        let schema = VmSchema::new();
        let prompt = schema.user_prompt("Analyze this file", std::iter::empty(), &[]);

        assert!(prompt.contains("Analyze this file"));
        assert!(!prompt.contains("{{TASK}}"));
        assert!(!prompt.contains("{{PAGES}}"));
        assert!(!prompt.contains("{{TRACE}}"));

        println!("{}", prompt);
    }

    #[test]
    fn test_user_prompt_with_trace() {
        let schema = VmSchema::new();
        let trace = vec![
            ExecutionStep {
                step: 1,
                opcode: "READ_FILE".to_string(),
                result: "success".to_string(),
                error: None,
            },
        ];
        let prompt = schema.user_prompt("Continue task", std::iter::empty(), &trace);

        assert!(prompt.contains("Execution History"));
        assert!(prompt.contains("READ_FILE"));

        println!("{}", prompt);
    }
}