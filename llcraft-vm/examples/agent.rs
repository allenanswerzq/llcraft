//! # LLcraft Agent - True End-to-End Demo
//!
//! This demonstrates the full agent loop:
//! 1. User provides a task description
//! 2. LLM generates a program to solve it
//! 3. Interpreter runs the program
//! 4. If program needs more LLM input, we call the LLM and continue
//! 5. INJECT allows JIT code generation - LLM generates new opcodes at runtime
//! 6. Multi-step until COMPLETE or FAIL
//!
//! The LLM is the brain, the VM is the body.

use llcraft_vm::{
    BridgeProvider, ChatMessage, CompletionRequest, DefaultSyscallHandler, ExecutionResult,
    Interpreter, LlmProvider, LlmRequest, LlmRequestType, MemoryPage, Program, VmSchema, TaskRequest,
    ExecutionStep, Opcode, SessionManager, PageIndex,
};

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
    /// Session manager for persistence
    session_manager: Option<SessionManager>,
    /// Current session ID
    session_id: Option<String>,
    /// Page index from session (rich metadata - NOT content)
    /// LLM sees these summaries and uses LOAD_PAGE to fetch what it needs
    page_index: std::collections::HashMap<String, PageIndex>,
}

impl Agent {
    fn new() -> Self {
        Self {
            provider: BridgeProvider::local(),
            schema: VmSchema::new(),
            full_trace: Vec::new(),
            verbose: true,
            session_manager: None,
            session_id: None,
            page_index: std::collections::HashMap::new(),
        }
    }

    /// Enable session persistence
    fn with_session(mut self, session_dir: &str, session_id: Option<&str>) -> Result<Self, String> {
        let manager = SessionManager::new(session_dir).map_err(|e| e.to_string())?;

        // Either resume existing session or create a new one
        let (sid, new_page_index) = if let Some(id) = session_id {
            // Resume existing session
            if manager.session_exists(id) {
                let session = manager.load_session(id).map_err(|e| e.to_string())?;

                // Only load page INDEX (rich metadata) - NOT the content!
                // LLM will use LOAD_PAGE opcode to fetch what it needs
                let mut page_index = std::collections::HashMap::new();

                println!("Resuming session: {}", id);
                println!("   Previous task: {}", session.metadata.task);
                println!("   Available pages (use LOAD_PAGE to fetch content):");

                // Calculate total tokens across all pages
                let total_tokens: usize = session.page_index.values().map(|idx| idx.tokens).sum();

                for (page_id, idx) in &session.page_index {
                    // Show rich metadata like RLM's QueryMetadata
                    println!("     - {} (~{} tokens): {}", page_id, idx.tokens, idx.summary);
                    page_index.insert(page_id.clone(), idx.clone());
                }

                if !page_index.is_empty() {
                    println!("   {} pages indexed (~{} total tokens, content NOT loaded)\n",
                        page_index.len(), total_tokens);
                }

                (id.to_string(), page_index)
            } else {
                // Create new session with this specific ID
                println!("Creating new session: {}", id);
                let session = llcraft_vm::Session::new(id, "agent session");
                manager.save_session(&session).map_err(|e| e.to_string())?;
                (id.to_string(), std::collections::HashMap::new())
            }
        } else {
            // Create new session with auto-generated ID
            let session = manager.create_session("agent session").map_err(|e| e.to_string())?;
            println!("Created new session: {}", session.metadata.id);
            (session.metadata.id.clone(), std::collections::HashMap::new())
        };

        self.session_manager = Some(manager);
        self.session_id = Some(sid);
        self.page_index = new_page_index;

        Ok(self)
    }

    /// Run a task to completion
    async fn run(&mut self, task: &str) -> Result<AgentResult, String> {
        println!("Task: {}\n", task);

        // Show available pages (rich metadata - LLM must LOAD_PAGE to get content)
        if !self.page_index.is_empty() {
            let total_tokens: usize = self.page_index.values().map(|idx| idx.tokens).sum();
            println!("Available pages in session (~{} tokens total, use LOAD_PAGE to fetch):", total_tokens);
            for (page_id, idx) in &self.page_index {
                println!("   - {} (~{} tokens): {}", page_id, idx.tokens, idx.summary);
            }
            println!();
        }

        // Step 1: Ask LLM to generate a program ONCE
        let program = self.generate_program(task).await?;

        if self.verbose {
            println!("Generated Program:");
            program.pretty_print();
        }

        // Step 2: Run the program to completion
        // The LLM controls loops via LOOP/PLAN/REFLECT opcodes
        // We just execute and handle LLM requests as they come
        self.run_program(program).await
    }

    /// Generate a program from the LLM based on the task
    async fn generate_program(&mut self, task: &str) -> Result<Program, String> {
        // Build the request with execution history and session context
        let mut enhanced_task = task.to_string();

        // Show page INDEX with rich metadata (like RLM's QueryMetadata)
        // LLM must use LOAD_PAGE to get content
        if !self.page_index.is_empty() {
            let total_tokens: usize = self.page_index.values().map(|idx| idx.tokens).sum();
            enhanced_task.push_str(&format!(
                "\n\nAVAILABLE PAGES FROM PREVIOUS TASKS (~{} tokens total):\n",
                total_tokens
            ));
            for (page_id, idx) in &self.page_index {
                enhanced_task.push_str(&format!(
                    "- Page '{}' (~{} tokens): {}\n",
                    page_id, idx.tokens, idx.summary
                ));
            }
            enhanced_task.push_str("\nIMPORTANT: Page content is NOT loaded. Use LOAD_PAGE opcode to fetch pages you need.\n");
            enhanced_task.push_str("You can also use INFER_BATCH to process multiple chunks concurrently.\n");
        }

        let request = TaskRequest::new(&enhanced_task)
            .with_trace(self.full_trace.clone());

        // Build messages
        let system = TaskRequest::system_prompt(&self.schema);
        let user = request.user_prompt();

        if self.verbose {
            println!("Asking LLM to generate program...");
            if !self.full_trace.is_empty() {
                println!("   (with {} previous execution steps as context)", self.full_trace.len());
            }
            if !self.page_index.is_empty() {
                println!("   (with {} page summaries from session)", self.page_index.len());
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

        // Connect interpreter to session for LOAD_PAGE opcode to work
        // LLM uses LOAD_PAGE to fetch pages on-demand (lazy loading)
        if let (Some(ref manager), Some(ref session_id)) = (&self.session_manager, &self.session_id) {
            // Clone the manager for the interpreter
            let interp_manager = SessionManager::new(".llcraft_sessions").map_err(|e| e.to_string())?;
            interp = interp.with_session_manager(interp_manager);
            interp.resume_session(session_id).map_err(|e| e.to_string())?;

            if self.verbose {
                println!("   Session connected - LOAD_PAGE enabled for: {}", session_id);
            }
        }

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
                    println!("\nTask completed!");

                    // Collect all pages for the result
                    let pages = self.collect_pages(&interp);

                    // Save pages to session if enabled
                    self.save_to_session(&pages)?;

                    return Ok(AgentResult { result, pages });
                }
                ExecutionResult::Failed(error) => {
                    self.full_trace.extend(interp.trace().iter().cloned());
                    println!("\nTask failed: {}", error);
                    return Err(error);
                }
                ExecutionResult::NeedsLlm(request) => {
                    // Check if this is an INJECT request (returns opcodes, not response)
                    if let LlmRequestType::Inject { .. } = &request.request_type {
                        let opcodes = self.handle_inject_request(&request, &interp).await?;
                        let count = interp.inject_opcodes(opcodes).map_err(|e| e.to_string())?;
                        if self.verbose {
                            println!("   Injected {} opcodes", count);
                        }
                    } else if let LlmRequestType::InferBatch { prompts, context, store_prefix, store_combined, .. } = &request.request_type {
                        // Batched inference - run concurrently
                        let results = self.handle_infer_batch_request(
                            prompts,
                            context,
                            store_prefix,
                            store_combined,
                        ).await?;

                        // Store each result in its own page
                        for (i, result) in results.iter().enumerate() {
                            let page_id = format!("{}_{}", store_prefix, i);
                            interp.provide_llm_response(result.clone(), &page_id)
                                .map_err(|e| e.to_string())?;
                        }

                        // Optionally store combined results
                        if let Some(combined_page) = store_combined {
                            let combined = serde_json::json!({
                                "results": results,
                                "count": results.len(),
                                "success": true
                            });
                            interp.provide_llm_response(combined, combined_page)
                                .map_err(|e| e.to_string())?;
                        }
                    } else {
                        // Regular LLM request - store response in memory
                        let response = self.handle_llm_request(&request, &interp).await?;
                        interp.provide_llm_response(response, &request.store_to)
                            .map_err(|e| e.to_string())?;
                    }
                }
                ExecutionResult::StepLimitExceeded => {
                    self.full_trace.extend(interp.trace().iter().cloned());
                    return Err("Step limit exceeded".to_string());
                }
            }
        }
    }

    /// Save pages to session
    fn save_to_session(&mut self, pages: &std::collections::HashMap<String, serde_json::Value>) -> Result<(), String> {
        if let (Some(manager), Some(session_id)) = (&self.session_manager, &self.session_id) {
            // Load existing session (it should exist since we create it in with_session)
            let mut session = manager.load_session(session_id).map_err(|e| e.to_string())?;

            // Save each page
            for (page_id, content) in pages {
                let page = MemoryPage::new(page_id, content.clone());
                let summary = summarize_value(content);
                session.index_page(&page, Some(summary.clone()));
                manager.save_page(session_id, &page).map_err(|e| e.to_string())?;

                // Update our local page index with rich metadata
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let idx = PageIndex {
                    id: page_id.clone(),
                    summary,
                    tokens: page.size_tokens,
                    content_type: None,
                    created_at: now,
                    accessed_at: now,
                    loaded: false,
                };
                self.page_index.insert(page_id.clone(), idx);
            }

            // Save session metadata
            manager.save_session(&session).map_err(|e| e.to_string())?;

            if self.verbose {
                println!("   Saved {} pages to session", pages.len());
            }
        }
        Ok(())
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
            println!("\n   LLM Request ({:?})", request.request_type);
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
            LlmRequestType::Inject { .. } => {
                // Should not reach here - handled by handle_inject_request
                unreachable!("INJECT should be handled by handle_inject_request");
            }
            LlmRequestType::InferBatch { .. } => {
                // Should not reach here - handled by handle_infer_batch_request
                unreachable!("INFER_BATCH should be handled by handle_infer_batch_request");
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

    /// Handle an INJECT request - LLM generates opcodes to insert
    async fn handle_inject_request(
        &self,
        request: &LlmRequest,
        interp: &Interpreter<DefaultSyscallHandler>,
    ) -> Result<Vec<Opcode>, String> {
        if self.verbose {
            println!("\n   INJECT Request");
            println!("      Goal: {}", truncate(&request.prompt, 60));
        }

        // Build context from pages
        let mut context = String::new();
        for page_id in &request.context_pages {
            if let Some(content) = interp.get_page(page_id) {
                context.push_str(&format!("### Page: {}\n{}\n\n", page_id, content));
            }
        }

        // Build trace if requested
        let (include_trace, include_memory) = match &request.request_type {
            LlmRequestType::Inject { include_trace, include_memory } => (*include_trace, *include_memory),
            _ => (false, false),
        };

        let trace_text = if include_trace {
            let trace: Vec<String> = interp.trace().iter()
                .map(|s| format!("{}: {} -> {}", s.step, s.opcode, s.result))
                .collect();
            format!("\n\n## Execution Trace:\n{}", trace.join("\n"))
        } else {
            String::new()
        };

        let memory_text = if include_memory {
            let pages = interp.all_pages();
            let page_summary: Vec<String> = pages.iter()
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

        // Build the injection prompt
        let prompt = format!(
            r#"# JIT Code Injection Request

You are the LLM CPU of a running VM program. The program has reached an INJECT point and needs you to generate the next set of opcodes to execute.

## Goal
{}

## Current Context
{}{}{}

## Tool Opcode Reference (EXACT field names required):
- READ_FILE: {{"op": "READ_FILE", "path": "<file>", "store_to": "<page>"}}
- WRITE_FILE: {{"op": "WRITE_FILE", "path": "<file>", "content": "<text>", "store_to": "<page>"}}
- LIST_DIR: {{"op": "LIST_DIR", "path": "<dir>", "store_to": "<page>"}}
- EXEC: {{"op": "EXEC", "command": "<shell cmd>", "store_to": "<page>"}}  ← NOTE: "command" not "cmd"
- GREP: {{"op": "GREP", "pattern": "<regex>", "path": "<file>", "store_to": "<page>"}}
- INFER: {{"op": "INFER", "prompt": "<question>", "context": ["<page1>"], "store_to": "<page>"}}
- BRANCH: {{"op": "BRANCH", "condition": "<page.field>", "if_true": "<label>", "if_false": "<label>"}}
- COMPLETE: {{"op": "COMPLETE", "result": {{...}}}}
- FAIL: {{"op": "FAIL", "error": "<message>"}}

## Instructions
Generate a JSON array of opcodes. These will be inserted and executed immediately.

IMPORTANT: Return ONLY a valid JSON array. Example:
[
  {{"op": "READ_FILE", "path": "file.txt", "store_to": "content"}},
  {{"op": "INFER", "prompt": "Analyze this", "context": ["content"], "store_to": "result"}},
  {{"op": "COMPLETE", "result": {{"page": "result"}}}}
]

Generate the opcodes now:"#,
            request.prompt, context, trace_text, memory_text
        );

        let completion_request = CompletionRequest::new(vec![
            ChatMessage::user(prompt),
        ]);

        let response = self.provider.complete(completion_request).await
            .map_err(|e| format!("LLM error: {:?}", e))?;

        let content = response.content.ok_or("Empty LLM response")?;

        if self.verbose {
            println!("      Response: {} chars", content.len());
        }

        // Parse the opcodes from the response
        self.parse_opcodes(&content)
    }

    /// Parse opcodes from LLM output (handles markdown fences)
    fn parse_opcodes(&self, content: &str) -> Result<Vec<Opcode>, String> {
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

        serde_json::from_str::<Vec<Opcode>>(json_str)
            .map_err(|e| format!("Failed to parse injected opcodes: {}\n\nContent:\n{}", e, json_str))
    }

    /// Handle an INFER_BATCH request - run multiple LLM queries
    /// Note: Currently runs sequentially, but could be made truly parallel
    /// with proper provider architecture (Clone or Arc<Provider>)
    async fn handle_infer_batch_request(
        &self,
        prompts: &[String],
        context: &[serde_json::Value],
        store_prefix: &str,
        _store_combined: &Option<String>,
    ) -> Result<Vec<serde_json::Value>, String> {
        if self.verbose {
            println!("\n   INFER_BATCH Request");
            println!("      Running {} prompts...", prompts.len());
        }

        // Build context string once (shared by all prompts)
        let context_text: String = context.iter()
            .enumerate()
            .map(|(i, v)| format!("### Context {}\n{}\n", i, serde_json::to_string_pretty(v).unwrap_or_default()))
            .collect();

        // Run prompts (sequentially for now due to provider ownership)
        // TODO: Make truly parallel with Arc<Provider> or channels
        let mut results = Vec::with_capacity(prompts.len());

        for (i, prompt) in prompts.iter().enumerate() {
            let full_prompt = if context_text.is_empty() {
                prompt.clone()
            } else {
                format!("{}\n\n## Context:\n{}", prompt, context_text)
            };

            let req = CompletionRequest::new(vec![ChatMessage::user(full_prompt)]);
            let result = match self.provider.complete(req).await {
                Ok(resp) => {
                    let content = resp.content.unwrap_or_default();
                    serde_json::json!({
                        "response": content,
                        "success": true,
                        "index": i
                    })
                }
                Err(e) => {
                    serde_json::json!({
                        "error": format!("{:?}", e),
                        "success": false,
                        "index": i
                    })
                }
            };
            results.push(result);

            if self.verbose {
                println!("      [{}/{}] {} → {}", i + 1, prompts.len(), store_prefix,
                    if results.last().map(|r| r["success"].as_bool().unwrap_or(false)).unwrap_or(false) { "ok" } else { "err" });
            }
        }

        if self.verbose {
            let successes = results.iter().filter(|r| r["success"].as_bool().unwrap_or(false)).count();
            println!("      Completed: {}/{} successful", successes, results.len());
        }

        Ok(results)
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len])
    }
}

fn summarize_value(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => {
            if s.len() > 60 {
                format!("{}...", &s[..60])
            } else {
                s.clone()
            }
        }
        serde_json::Value::Object(obj) => {
            format!("Object with keys: {:?}", obj.keys().collect::<Vec<_>>())
        }
        serde_json::Value::Array(arr) => {
            format!("Array with {} items", arr.len())
        }
        _ => format!("{}", content),
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
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
        eprintln!("Example: {} \"Read Cargo.toml and list the dependencies\"", args[0]);
        eprintln!("Example: {} --session my_session \"Read Cargo.toml\"", args[0]);
        eprintln!("\nWith --session, context persists between runs:");
        eprintln!("  1. {} -s demo \"Read Cargo.toml and extract the package name\"", args[0]);
        eprintln!("  2. {} -s demo \"What is the version of this package?\"", args[0]);
        std::process::exit(1);
    }

    let task = task_parts.join(" ");

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           LLcraft Agent - End-to-End Demo                 ║");
    println!("║  The LLM generates programs, the VM executes them         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    run_task(&task, session_id.as_deref()).await;
}

async fn run_task(task: &str, session_id: Option<&str>) {
    println!();

    let mut agent = Agent::new();

    // Enable session persistence if requested
    if let Some(sid) = session_id {
        match agent.with_session(".llcraft_sessions", Some(sid)) {
            Ok(a) => agent = a,
            Err(e) => {
                eprintln!("Failed to initialize session: {}", e);
                return;
            }
        }
    }

    match agent.run(task).await {
        Ok(agent_result) => {
            // Try to extract the actual answer from result pages
            println!("\n╔═══════════════════════════════════════════════════════════╗");
            println!("║                      FINAL ANSWER                       ║");
            println!("╚═══════════════════════════════════════════════════════════╝\n");

            // Look for the main result page and extract readable content
            let answer = extract_answer(&agent_result.result, &agent_result.pages);
            println!("{}", answer);

            // Show raw result structure if verbose
            if std::env::var("VERBOSE").is_ok() {
                println!("\nRaw Result:");
                println!("{}", serde_json::to_string_pretty(&agent_result.result).unwrap_or_default());

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
    println!("Execution Trace ({} steps):", agent.full_trace.len());
    println!("═══════════════════════════════════════════════════════════");
    for step in &agent.full_trace {
        let err = step.error.as_ref().map(|e| format!(" {}", e)).unwrap_or_default();
        println!("  {:3}. {} → {}{}", step.step, step.opcode, truncate(&step.result, 50), err);
    }
}

/// Extract a human-readable answer from result and pages
fn extract_answer(
    result: &serde_json::Value,
    pages: &std::collections::HashMap<String, serde_json::Value>,
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
                    // Extract the most relevant content from the page
                    if let Some(response) = page_content.get("response").and_then(|v| v.as_str()) {
                        answer_parts.push(format!("─── {} ───\n{}", page_id, response));
                    } else if let Some(content) = page_content.get("content").and_then(|v| v.as_str()) {
                        // For file contents, show first part
                        let preview: String = content.lines().take(20).collect::<Vec<_>>().join("\n");
                        if content.lines().count() > 20 {
                            answer_parts.push(format!("─── {} ───\n{}...\n(truncated)", page_id, preview));
                        } else {
                            answer_parts.push(format!("─── {} ───\n{}", page_id, content));
                        }
                    } else if let Some(files) = page_content.get("files").and_then(|v| v.as_array()) {
                        let file_list: Vec<&str> = files.iter().filter_map(|f| f.as_str()).collect();
                        answer_parts.push(format!("─── {} ───\nFiles: {}", page_id, file_list.join(", ")));
                    } else if let Some(stdout) = page_content.get("stdout").and_then(|v| v.as_str()) {
                        answer_parts.push(format!("─── {} ───\n{}", page_id, stdout.trim()));
                    } else if let Some(matches) = page_content.get("matches").and_then(|v| v.as_array()) {
                        let match_list: Vec<&str> = matches.iter().filter_map(|m| m.as_str()).take(10).collect();
                        let count = page_content.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
                        answer_parts.push(format!("─── {} ({} matches) ───\n{}", page_id, count, match_list.join("\n")));
                    } else {
                        // Fallback: show the whole page content
                        let content_str = serde_json::to_string_pretty(page_content).unwrap_or_default();
                        answer_parts.push(format!("─── {} ───\n{}", page_id, content_str));
                    }
                }
            }
        }
    }

    if answer_parts.is_empty() {
        // Fallback: just show the result
        serde_json::to_string_pretty(result).unwrap_or_else(|_| "No result".to_string())
    } else {
        answer_parts.join("\n\n")
    }
}
