//! # LLcraft CLI
//!
//! Command-line interface for running the LLcraft agent.
//!
//! Usage:
//!   llcraft <task>
//!   llcraft --session <id> <task>
//!   llcraft program <file.json>
//!
//! Examples:
//!   llcraft "Read Cargo.toml and list the dependencies"
//!   llcraft -s demo "Read Cargo.toml and extract the package name"
//!   llcraft -s demo "What is the version of this package?"
//!   llcraft program examples/ralph.json

use clap::{Parser, Subcommand};
use llcraft_agent::{Agent, AgentConfig};
use llcraft_vm::{
    BridgeProvider, DefaultSyscallHandler, ExecutionResult, Interpreter, LlmProvider,
    LlmRequest, LlmRequestType, Program, ChatMessage, CompletionRequest,
};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "llcraft")]
#[command(author, version, about = "LLcraft - Your AI's operating system")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Task to execute (when not using subcommands)
    #[arg(trailing_var_arg = true)]
    task: Vec<String>,

    /// Session ID for persistent context across runs
    #[arg(short, long, global = true)]
    session: Option<String>,

    /// Enable verbose output (show raw results and all pages)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode - only show final answer
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a task
    Run {
        /// The task description
        #[arg(trailing_var_arg = true, required = true)]
        task: Vec<String>,
    },
    /// Run a program from a JSON file
    Program {
        /// Path to the program JSON file
        #[arg(required = true)]
        file: String,

        /// Maximum execution steps (default: 1000)
        #[arg(short, long, default_value = "1000")]
        max_steps: usize,
    },
    /// List existing sessions
    Sessions,
    /// Show VM schema (available opcodes)
    Schema,
}

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
                        answer_parts.push(format!("--- {} ---\n{}", page_id, response));
                    } else if let Some(content) = page_content.get("content").and_then(|v| v.as_str())
                    {
                        let preview: String =
                            content.lines().take(20).collect::<Vec<_>>().join("\n");
                        if content.lines().count() > 20 {
                            answer_parts.push(format!(
                                "--- {} ---\n{}...\n(truncated)",
                                page_id, preview
                            ));
                        } else {
                            answer_parts.push(format!("--- {} ---\n{}", page_id, content));
                        }
                    } else if let Some(files) = page_content.get("files").and_then(|v| v.as_array())
                    {
                        let file_list: Vec<&str> =
                            files.iter().filter_map(|f| f.as_str()).collect();
                        answer_parts.push(format!(
                            "--- {} ---\nFiles: {}",
                            page_id,
                            file_list.join(", ")
                        ));
                    } else if let Some(stdout) = page_content.get("stdout").and_then(|v| v.as_str())
                    {
                        answer_parts.push(format!("--- {} ---\n{}", page_id, stdout.trim()));
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
                            "--- {} ({} matches) ---\n{}",
                            page_id,
                            count,
                            match_list.join("\n")
                        ));
                    } else {
                        let content_str =
                            serde_json::to_string_pretty(page_content).unwrap_or_default();
                        answer_parts.push(format!("--- {} ---\n{}", page_id, content_str));
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

async fn run_task(task: &str, session_id: Option<&str>, verbose: bool, quiet: bool) {
    if !quiet {
        println!();
    }

    let config = AgentConfig {
        verbose: !quiet,
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
            if !quiet {
                println!("\n--- FINAL ANSWER ---\n");
            }

            let answer = extract_answer(&agent_result.result, &agent_result.pages);
            println!("{}", answer);

            if verbose {
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
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    if !quiet {
        println!("\n--- Execution Trace ({} steps) ---", agent.trace().len());
        for step in agent.trace() {
            let err = step
                .error
                .as_ref()
                .map(|e| format!(" {}", e))
                .unwrap_or_default();
            println!(
                "  {:3}. {} -> {}{}",
                step.step,
                step.opcode,
                truncate(&step.result, 50),
                err
            );
        }
    }
}

fn list_sessions() {
    let session_dir = ".llcraft_sessions";
    match std::fs::read_dir(session_dir) {
        Ok(entries) => {
            println!("Sessions in {}:", session_dir);
            let mut count = 0;
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    if let Some(name) = entry.file_name().to_str() {
                        println!("  - {}", name);
                        count += 1;
                    }
                }
            }
            if count == 0 {
                println!("  (no sessions found)");
            }
        }
        Err(_) => {
            println!("No sessions directory found.");
        }
    }
}

fn show_schema() {
    let schema = llcraft_agent::schema_summary();
    println!("{}", schema);
}

async fn run_program_file(file: &str, max_steps: usize, verbose: bool, quiet: bool) {
    // Read and parse program
    let content = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {}: {}", file, e);
            std::process::exit(1);
        }
    };

    let program: Program = match serde_json::from_str(&content) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error parsing program: {}", e);
            std::process::exit(1);
        }
    };

    if !quiet {
        println!("Running program: {} ({})", program.name, program.id);
        println!("Description: {}", program.description.as_deref().unwrap_or("(no description)"));
        println!("Opcodes: {}", program.code.len());
        println!("Max steps: {}", max_steps);
        println!();
    }

    // Create interpreter
    let mut interp = Interpreter::new(program, DefaultSyscallHandler::default());

    if verbose {
        interp = interp.with_log_callback(|level, msg| {
            println!("   [{:?}] {}", level, msg);
        });
    }

    // Create LLM provider for handling INFER/PLAN/REFLECT/INJECT
    let provider = BridgeProvider::local();

    // Track steps manually
    let mut total_steps = 0;

    // Run the program
    loop {
        if total_steps >= max_steps {
            eprintln!("\n=== STEP LIMIT EXCEEDED ===");
            eprintln!("Program did not complete within {} steps", max_steps);
            std::process::exit(1);
        }

        match interp.run() {
            Ok(ExecutionResult::Complete(result)) => {
                if !quiet {
                    println!("\n=== PROGRAM COMPLETE ===\n");
                }
                println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());

                if verbose {
                    println!("\n--- Pages ---");
                    for (id, content) in interp.all_pages() {
                        println!("  {}: {}", id, truncate(&serde_json::to_string(&content).unwrap_or_default(), 80));
                    }
                }
                break;
            }
            Ok(ExecutionResult::Failed(error)) => {
                eprintln!("\n=== PROGRAM FAILED ===\n");
                eprintln!("Error: {}", error);
                std::process::exit(1);
            }
            Ok(ExecutionResult::StepLimitExceeded) => {
                eprintln!("\n=== STEP LIMIT EXCEEDED ===");
                eprintln!("Program did not complete within {} steps", max_steps);
                std::process::exit(1);
            }
            Ok(ExecutionResult::NeedsLlm(request)) => {
                if !quiet {
                    println!("   LLM Request: {:?}", request.request_type);
                    println!("      Prompt: {}", truncate(&request.prompt, 60));
                }

                // Handle the LLM request
                let response = handle_llm_request(&provider, &request, &interp, quiet).await;

                match response {
                    Ok(value) => {
                        if let LlmRequestType::Inject { .. } = &request.request_type {
                            // For INJECT, parse and inject opcodes
                            let opcodes = parse_opcodes(&value);
                            match interp.inject_opcodes(opcodes) {
                                Ok(count) => {
                                    if !quiet {
                                        println!("      Injected {} opcodes", count);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Error injecting opcodes: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        } else {
                            if let Err(e) = interp.provide_llm_response(value, &request.store_to) {
                                eprintln!("Error providing LLM response: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("LLM error: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Execution error: {}", e);
                std::process::exit(1);
            }
        }
        total_steps += 1;
    }

    if !quiet {
        println!("\n--- Execution Trace ({} steps) ---", interp.trace().len());
        for step in interp.trace().iter().take(50) {
            println!(
                "  {:3}. {} -> {}",
                step.step,
                step.opcode,
                truncate(&step.result, 50)
            );
        }
        if interp.trace().len() > 50 {
            println!("  ... ({} more steps)", interp.trace().len() - 50);
        }
    }
}

async fn handle_llm_request(
    provider: &BridgeProvider,
    request: &LlmRequest,
    interp: &Interpreter<DefaultSyscallHandler>,
    quiet: bool,
) -> Result<serde_json::Value, String> {
    // Build context from pages
    let mut context = String::new();
    for page_id in &request.context_pages {
        if let Some(content) = interp.get_page(page_id) {
            context.push_str(&format!("### Page: {}\n{}\n\n", page_id, content));
        }
    }

    let prompt = match &request.request_type {
        LlmRequestType::Infer => {
            if context.is_empty() {
                request.prompt.clone()
            } else {
                format!("{}\n\n## Context:\n{}", request.prompt, context)
            }
        }
        LlmRequestType::Plan => {
            format!(
                "# Planning Request\n\n{}\n\n## Context:\n{}\n\nGenerate a plan as JSON with steps.",
                request.prompt, context
            )
        }
        LlmRequestType::Reflect { include_trace } => {
            let trace_text = if *include_trace {
                let trace: Vec<String> = interp
                    .trace()
                    .iter()
                    .map(|s| format!("{}: {} -> {}", s.step, s.opcode, s.result))
                    .collect();
                format!("\n\n## Execution Trace:\n{}", trace.join("\n"))
            } else {
                String::new()
            };
            format!(
                "# Reflection Request\n\n{}\n\n## Context:\n{}{}",
                request.prompt, context, trace_text
            )
        }
        LlmRequestType::Inject { include_trace, include_memory } => {
            let trace_text = if *include_trace {
                let trace: Vec<String> = interp
                    .trace()
                    .iter()
                    .map(|s| format!("{}: {} -> {}", s.step, s.opcode, s.result))
                    .collect();
                format!("\n\n## Execution Trace:\n{}", trace.join("\n"))
            } else {
                String::new()
            };

            let memory_text = if *include_memory {
                let pages = interp.all_pages();
                let page_summary: Vec<String> = pages
                    .iter()
                    .map(|(id, content)| {
                        let preview = serde_json::to_string(content)
                            .map(|s| truncate(&s, 200))
                            .unwrap_or_default();
                        format!("  - {}: {}", id, preview)
                    })
                    .collect();
                format!("\n\n## Memory Pages:\n{}", page_summary.join("\n"))
            } else {
                String::new()
            };

            format!(
                r#"# JIT Code Injection Request

Generate opcodes to accomplish this goal:

{}

## Context
{}{}{}

## Opcode Syntax (use EXACT field names)
- READ_FILE: {{"op": "READ_FILE", "path": "filepath", "store_to": "page_id"}}
- WRITE_FILE: {{"op": "WRITE_FILE", "path": "filepath", "content": "text"}}
- EXEC: {{"op": "EXEC", "command": "shell command", "store_to": "page_id"}}
- LIST_DIR: {{"op": "LIST_DIR", "path": "dirpath", "store_to": "page_id"}}
- GREP: {{"op": "GREP", "pattern": "regex", "path": "dir", "store_to": "page_id"}}
- INFER: {{"op": "INFER", "prompt": "question", "context": ["page1"], "store_to": "page_id"}}
- STORE: {{"op": "STORE", "page_id": "name", "data": {{"any": "json"}}}}
- LOG: {{"op": "LOG", "level": "info", "message": "text"}}
- COMPLETE: {{"op": "COMPLETE", "result": {{"any": "json"}}}}
- FAIL: {{"op": "FAIL", "error": "error message"}}

Return ONLY a valid JSON array. No markdown, no explanation."#,
                request.prompt, context, trace_text, memory_text
            )
        }
        LlmRequestType::InferBatch { .. } => {
            // For now, handle as single infer
            request.prompt.clone()
        }
    };

    let completion_request = CompletionRequest::new(vec![ChatMessage::user(prompt)]);

    let response = provider
        .complete(completion_request)
        .await
        .map_err(|e| format!("LLM error: {:?}", e))?;

    let content = response.content.ok_or("Empty LLM response")?;

    if !quiet {
        println!("      Response: {} chars", content.len());
    }

    // For INJECT, return raw content; otherwise wrap in JSON
    if matches!(&request.request_type, LlmRequestType::Inject { .. }) {
        Ok(serde_json::json!(content))
    } else {
        Ok(serde_json::json!({
            "response": content,
            "success": true
        }))
    }
}

fn parse_opcodes(value: &serde_json::Value) -> Vec<llcraft_vm::Opcode> {
    let content = value.as_str().unwrap_or("");

    // Extract JSON from markdown fences if present
    let json_str = if content.contains("```json") {
        content
            .split("```json")
            .nth(1)
            .and_then(|s| s.split("```").next())
            .map(|s| s.trim())
            .unwrap_or(content)
    } else if content.contains("```") {
        content
            .split("```")
            .nth(1)
            .map(|s| s.trim())
            .unwrap_or(content)
    } else {
        content.trim()
    };

    // Try to parse as a complete array first
    match serde_json::from_str::<Vec<llcraft_vm::Opcode>>(json_str) {
        Ok(opcodes) => return opcodes,
        Err(e) => {
            eprintln!("Warning: Failed to parse opcodes as array: {}", e);
        }
    }
    
    // Fall back: try to parse each object individually
    if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
        let mut opcodes = Vec::new();
        for (i, val) in arr.iter().enumerate() {
            match serde_json::from_value::<llcraft_vm::Opcode>(val.clone()) {
                Ok(op) => opcodes.push(op),
                Err(e) => {
                    eprintln!("  Skipping opcode {}: {} ({})", i, e, truncate(&val.to_string(), 60));
                }
            }
        }
        if !opcodes.is_empty() {
            eprintln!("  Recovered {} valid opcodes out of {}", opcodes.len(), arr.len());
            return opcodes;
        }
    }
    
    eprintln!("Content: {}", truncate(json_str, 200));
    vec![]
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Handle subcommands
    match cli.command {
        Some(Commands::Sessions) => {
            list_sessions();
            return;
        }
        Some(Commands::Schema) => {
            show_schema();
            return;
        }
        Some(Commands::Program { file, max_steps }) => {
            if !cli.quiet {
                println!("LLcraft VM - Running program from file\n");
            }
            run_program_file(&file, max_steps, cli.verbose, cli.quiet).await;
            return;
        }
        Some(Commands::Run { task }) => {
            let task_str = task.join(" ");
            if !cli.quiet {
                println!("LLcraft Agent - Your AI's operating system\n");
            }
            run_task(&task_str, cli.session.as_deref(), cli.verbose, cli.quiet).await;
            return;
        }
        None => {
            // Default: treat remaining args as task
            if cli.task.is_empty() {
                eprintln!("Error: No task provided.");
                eprintln!("Usage: llcraft [OPTIONS] <TASK>...");
                eprintln!("       llcraft run <TASK>...");
                eprintln!("       llcraft program <FILE.json>");
                eprintln!("       llcraft sessions");
                eprintln!("       llcraft schema");
                eprintln!("\nExamples:");
                eprintln!("  llcraft \"Read Cargo.toml and list dependencies\"");
                eprintln!("  llcraft -s demo \"Read Cargo.toml\"");
                eprintln!("  llcraft program examples/ralph.json");
                eprintln!("  llcraft --help");
                std::process::exit(1);
            }
        }
    }

    // Default: run task from positional args
    let task_str = cli.task.join(" ");
    if !cli.quiet {
        println!("LLcraft Agent - Your AI's operating system\n");
    }
    run_task(&task_str, cli.session.as_deref(), cli.verbose, cli.quiet).await;
}