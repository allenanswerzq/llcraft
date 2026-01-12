//! # LLcraft Agent - True End-to-End Demo
//!
//! This demonstrates the full agent loop:
//! 1. User provides a task description
//! 2. LLM generates a program to solve it
//! 3. Interpreter runs the program
//! 4. If program needs more LLM input, we call the LLM and continue
//! 5. Multi-step until COMPLETE or FAIL
//!
//! The LLM is the brain, the VM is the body.

use llcraft_vm::{
    BridgeProvider, ChatMessage, CompletionRequest, DefaultSyscallHandler, ExecutionResult,
    Interpreter, LlmProvider, LlmRequest, LlmRequestType, Program, VmSchema, TaskRequest,
    ExecutionStep,
};
use std::io::{self, Write};

/// Result from agent execution
struct AgentResult {
    /// Final result value
    result: serde_json::Value,
    /// All pages from the final interpreter state
    pages: std::collections::HashMap<String, serde_json::Value>,
}

/// The agent orchestrator - manages the LLM <-> VM loop
struct Agent {
    provider: BridgeProvider,
    schema: VmSchema,
    /// Accumulated trace across all programs
    full_trace: Vec<ExecutionStep>,
    /// Verbose logging
    verbose: bool,
}

impl Agent {
    fn new() -> Self {
        Self {
            provider: BridgeProvider::local(),
            schema: VmSchema::new(),
            full_trace: Vec::new(),
            verbose: true,
        }
    }

    /// Run a task to completion
    async fn run(&mut self, task: &str) -> Result<AgentResult, String> {
        println!("🎯 Task: {}\n", task);

        // Step 1: Ask LLM to generate a program ONCE
        let program = self.generate_program(task).await?;

        if self.verbose {
            println!("📜 Generated Program:");
            program.pretty_print();
        }

        // Step 2: Run the program to completion
        // The LLM controls loops via LOOP/PLAN/REFLECT opcodes
        // We just execute and handle LLM requests as they come
        self.run_program(program).await
    }

    /// Generate a program from the LLM based on the task
    async fn generate_program(&mut self, task: &str) -> Result<Program, String> {
        // Build the request with execution history
        let request = TaskRequest::new(task)
            .with_trace(self.full_trace.clone());

        // Build messages
        let system = TaskRequest::system_prompt(&self.schema);
        let user = request.user_prompt();

        if self.verbose {
            println!("🤖 Asking LLM to generate program...");
            if !self.full_trace.is_empty() {
                println!("   (with {} previous execution steps as context)", self.full_trace.len());
            }
        }

        let completion_request = CompletionRequest::new(vec![
            ChatMessage::system(system),
            ChatMessage::user(user),
        ]);

        let response = self.provider.complete(completion_request).await
            .map_err(|e| format!("LLM error: {:?}", e))?;

        let content = response.content.ok_or("Empty LLM response")?;

        if self.verbose {
            println!("   Response: {} chars", content.len());
        }

        // Parse the program from JSON
        self.parse_program(&content)
    }

    /// Parse a program from LLM output (handles markdown fences)
    fn parse_program(&self, content: &str) -> Result<Program, String> {
        // Try to extract JSON from markdown code blocks
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

        serde_json::from_str::<Program>(json_str)
            .map_err(|e| format!("Failed to parse program: {}\n\nContent:\n{}", e, json_str))
    }

    /// Run a program, handling any LLM requests along the way
    async fn run_program(&mut self, program: Program) -> Result<AgentResult, String> {
        let mut interp = Interpreter::new(program, DefaultSyscallHandler::default());

        // Set up logging
        if self.verbose {
            interp = interp.with_log_callback(|level, msg| {
                println!("   [{:?}] {}", level, msg);
            });
        }

        loop {
            match interp.run().map_err(|e| e.to_string())? {
                ExecutionResult::Complete(result) => {
                    self.full_trace.extend(interp.trace().iter().cloned());
                    println!("\n✅ Task completed!");

                    // Collect all pages for the result
                    let pages = self.collect_pages(&interp);
                    return Ok(AgentResult { result, pages });
                }
                ExecutionResult::Failed(error) => {
                    self.full_trace.extend(interp.trace().iter().cloned());
                    println!("\n❌ Task failed: {}", error);
                    return Err(error);
                }
                ExecutionResult::NeedsLlm(request) => {
                    // Handle the LLM request and continue
                    let response = self.handle_llm_request(&request, &interp).await?;
                    interp.provide_llm_response(response, &request.store_to)
                        .map_err(|e| e.to_string())?;
                }
                ExecutionResult::StepLimitExceeded => {
                    self.full_trace.extend(interp.trace().iter().cloned());
                    return Err("Step limit exceeded".to_string());
                }
            }
        }
    }

    /// Collect all pages from interpreter for final result
    fn collect_pages(&self, interp: &Interpreter<DefaultSyscallHandler>) -> std::collections::HashMap<String, serde_json::Value> {
        interp.all_pages()
    }

    /// Handle an LLM request from the interpreter
    async fn handle_llm_request(
        &self,
        request: &LlmRequest,
        interp: &Interpreter<DefaultSyscallHandler>,
    ) -> Result<serde_json::Value, String> {
        if self.verbose {
            println!("\n   🤖 LLM Request ({:?})", request.request_type);
            println!("      Prompt: {}", truncate(&request.prompt, 60));
        }

        // Build context from pages
        let mut context = String::new();
        for page_id in &request.context_pages {
            if let Some(content) = interp.get_page(page_id) {
                context.push_str(&format!("### Page: {}\n{}\n\n", page_id, content));
            }
        }

        // Build the full prompt based on request type
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
                    "# Planning Request\n\n{}\n\n## Context:\n{}\n\n\
                     Generate a plan as JSON with steps.",
                    request.prompt, context
                )
            }
            LlmRequestType::Reflect { include_trace } => {
                let trace_text = if *include_trace {
                    let trace: Vec<String> = interp.trace().iter()
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
        };

        let completion_request = CompletionRequest::new(vec![
            ChatMessage::user(prompt),
        ]);

        let response = self.provider.complete(completion_request).await
            .map_err(|e| format!("LLM error: {:?}", e))?;

        let content = response.content.ok_or("Empty LLM response")?;

        if self.verbose {
            println!("      Response: {} chars", content.len());
        }

        Ok(serde_json::json!({
            "response": content,
            "success": true
        }))
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len])
    }
}

#[tokio::main]
async fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           LLcraft Agent - End-to-End Demo                 ║");
    println!("║  The LLM generates programs, the VM executes them         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Example tasks to try
    let example_tasks = vec![
        "List all Rust source files in the current directory and count them.",
        "Read the Cargo.toml file and tell me what dependencies this project has.",
        "Find all files containing 'fn main' in the current directory.",
    ];

    println!("Example tasks you can try:");
    for (i, task) in example_tasks.iter().enumerate() {
        println!("  {}. {}", i + 1, task);
    }
    println!();

    // Get task from user
    print!("Enter your task (or number 1-{}): ", example_tasks.len());
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    let task = if let Ok(num) = input.parse::<usize>() {
        if num >= 1 && num <= example_tasks.len() {
            example_tasks[num - 1]
        } else {
            input
        }
    } else {
        input
    };

    if task.is_empty() {
        println!("No task provided, using example task 1.");
        run_task(example_tasks[0]).await;
    } else {
        run_task(task).await;
    }
}

async fn run_task(task: &str) {
    println!("\n");

    let mut agent = Agent::new();

    match agent.run(task).await {
        Ok(agent_result) => {
            println!("\n🎉 Final Result:");
            println!("{}", serde_json::to_string_pretty(&agent_result.result).unwrap_or_default());

            if !agent_result.pages.is_empty() {
                println!("\n📄 Pages:");
                for (id, content) in &agent_result.pages {
                    println!("  {}:", id);
                    let content_str = serde_json::to_string_pretty(content).unwrap_or_default();
                    for line in content_str.lines().take(10) {
                        println!("    {}", line);
                    }
                    if content_str.lines().count() > 10 {
                        println!("    ... (truncated)");
                    }
                }
            }
        }
        Err(e) => {
            println!("\n💥 Error: {}", e);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("📊 Execution Trace ({} steps):", agent.full_trace.len());
    println!("═══════════════════════════════════════════════════════════");
    for step in &agent.full_trace {
        let err = step.error.as_ref().map(|e| format!(" ⚠️ {}", e)).unwrap_or_default();
        println!("  {:3}. {} → {}{}", step.step, step.opcode, truncate(&step.result, 50), err);
    }
}
