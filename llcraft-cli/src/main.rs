//! # LLcraft CLI
//!
//! Command-line interface for running the LLcraft agent.
//!
//! Usage:
//!   llcraft <task>
//!   llcraft --session <id> <task>
//!
//! Examples:
//!   llcraft "Read Cargo.toml and list the dependencies"
//!   llcraft -s demo "Read Cargo.toml and extract the package name"
//!   llcraft -s demo "What is the version of this package?"

use llcraft_agent::{Agent, AgentConfig};
use std::collections::HashMap;

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len])
    }
}

/// Extract a human-readable answer from result and pages
fn extract_answer(
    result: &serde_json::Value,
    pages: &HashMap<String, serde_json::Value>,
) -> String {
    let mut answer_parts = Vec::new();

    // First check if result itself contains readable content
    if let Some(msg) = result.get("message").and_then(|v| v.as_str()) {
        answer_parts.push(msg.to_string());
    }
    if let Some(note) = result.get("note").and_then(|v| v.as_str()) {
        answer_parts.push(note.to_string());
    }

    // Look for page references in result and extract their content
    for (key, value) in result.as_object().into_iter().flatten() {
        if key.ends_with("_page") || key == "page" {
            if let Some(page_id) = value.as_str() {
                if let Some(page_content) = pages.get(page_id) {
                    if let Some(response) = page_content.get("response").and_then(|v| v.as_str()) {
                        answer_parts.push(format!("─── {} ───\n{}", page_id, response));
                    } else if let Some(content) = page_content.get("content").and_then(|v| v.as_str())
                    {
                        let preview: String =
                            content.lines().take(20).collect::<Vec<_>>().join("\n");
                        if content.lines().count() > 20 {
                            answer_parts.push(format!(
                                "─── {} ───\n{}...\n(truncated)",
                                page_id, preview
                            ));
                        } else {
                            answer_parts.push(format!("─── {} ───\n{}", page_id, content));
                        }
                    } else if let Some(files) = page_content.get("files").and_then(|v| v.as_array())
                    {
                        let file_list: Vec<&str> =
                            files.iter().filter_map(|f| f.as_str()).collect();
                        answer_parts.push(format!(
                            "─── {} ───\nFiles: {}",
                            page_id,
                            file_list.join(", ")
                        ));
                    } else if let Some(stdout) = page_content.get("stdout").and_then(|v| v.as_str())
                    {
                        answer_parts.push(format!("─── {} ───\n{}", page_id, stdout.trim()));
                    } else if let Some(matches) =
                        page_content.get("matches").and_then(|v| v.as_array())
                    {
                        let match_list: Vec<&str> =
                            matches.iter().filter_map(|m| m.as_str()).take(10).collect();
                        let count = page_content
                            .get("count")
                            .and_then(|c| c.as_u64())
                            .unwrap_or(0);
                        answer_parts.push(format!(
                            "─── {} ({} matches) ───\n{}",
                            page_id,
                            count,
                            match_list.join("\n")
                        ));
                    } else {
                        let content_str =
                            serde_json::to_string_pretty(page_content).unwrap_or_default();
                        answer_parts.push(format!("─── {} ───\n{}", page_id, content_str));
                    }
                }
            }
        }
    }

    if answer_parts.is_empty() {
        serde_json::to_string_pretty(result).unwrap_or_else(|_| "No result".to_string())
    } else {
        answer_parts.join("\n\n")
    }
}

async fn run_task(task: &str, session_id: Option<&str>) {
    println!();

    let config = AgentConfig {
        verbose: true,
        session_dir: ".llcraft_sessions".to_string(),
    };

    let mut agent = Agent::with_config(config);

    if let Some(sid) = session_id {
        match agent.with_session(Some(sid)) {
            Ok(a) => agent = a,
            Err(e) => {
                eprintln!("Failed to initialize session: {}", e);
                return;
            }
        }
    }

    match agent.run(task).await {
        Ok(agent_result) => {
            println!("\n╔═══════════════════════════════════════════════════════════╗");
            println!("║                      FINAL ANSWER                         ║");
            println!("╚═══════════════════════════════════════════════════════════╝\n");

            let answer = extract_answer(&agent_result.result, &agent_result.pages);
            println!("{}", answer);

            if std::env::var("VERBOSE").is_ok() {
                println!("\nRaw Result:");
                println!(
                    "{}",
                    serde_json::to_string_pretty(&agent_result.result).unwrap_or_default()
                );

                if !agent_result.pages.is_empty() {
                    println!("\nAll Pages:");
                    for (id, content) in &agent_result.pages {
                        println!("  {}:", id);
                        let content_str = serde_json::to_string_pretty(content).unwrap_or_default();
                        for line in content_str.lines().take(15) {
                            println!("    {}", line);
                        }
                        if content_str.lines().count() > 15 {
                            println!("    ... (truncated)");
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("\nError: {}", e);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("Execution Trace ({} steps):", agent.trace().len());
    println!("═══════════════════════════════════════════════════════════");
    for step in agent.trace() {
        let err = step
            .error
            .as_ref()
            .map(|e| format!(" {}", e))
            .unwrap_or_default();
        println!(
            "  {:3}. {} → {}{}",
            step.step,
            step.opcode,
            truncate(&step.result, 50),
            err
        );
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut session_id: Option<String> = None;
    let mut task_parts: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--session" || args[i] == "-s" {
            if i + 1 < args.len() {
                session_id = Some(args[i + 1].clone());
                i += 2;
            } else {
                eprintln!("Error: --session requires a session ID");
                std::process::exit(1);
            }
        } else {
            task_parts.push(args[i].clone());
            i += 1;
        }
    }

    if task_parts.is_empty() {
        eprintln!("Usage: {} [--session <id>] <task>", args[0]);
        eprintln!(
            "Example: {} \"Read Cargo.toml and list the dependencies\"",
            args[0]
        );
        eprintln!(
            "Example: {} --session my_session \"Read Cargo.toml\"",
            args[0]
        );
        eprintln!("\nWith --session, context persists between runs:");
        eprintln!(
            "  1. {} -s demo \"Read Cargo.toml and extract the package name\"",
            args[0]
        );
        eprintln!(
            "  2. {} -s demo \"What is the version of this package?\"",
            args[0]
        );
        std::process::exit(1);
    }

    let task = task_parts.join(" ");

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                    LLcraft Agent                          ║");
    println!("║      The LLM generates programs, the VM executes them     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    run_task(&task, session_id.as_deref()).await;
}
