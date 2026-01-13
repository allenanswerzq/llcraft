//! Agent implementation - orchestrates LLM <-> VM loop

use llcraft_vm::{
    BridgeProvider, ChatMessage, CompletionRequest, DefaultSyscallHandler, ExecutionResult,
    Interpreter, LlmProvider, LlmRequest, LlmRequestType, MemoryPage, Opcode, PageIndex,
    Program, Session, SessionManager, VmSchema,
};
use std::collections::HashMap;

/// Configuration for the agent
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Enable verbose logging
    pub verbose: bool,
    /// Session directory for persistence
    pub session_dir: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            verbose: true,
            session_dir: ".llcraft_sessions".to_string(),
        }
    }
}

/// Result from agent execution
pub struct AgentResult {
    /// Final result value
    pub result: serde_json::Value,
    /// All pages from the final interpreter state
    pub pages: HashMap<String, serde_json::Value>,
}

/// The agent orchestrator - manages the LLM <-> VM loop
pub struct Agent {
    provider: BridgeProvider,
    schema: VmSchema,
    /// Accumulated trace across all programs
    full_trace: Vec<llcraft_vm::ExecutionStep>,
    /// Configuration
    config: AgentConfig,
    /// Session manager for persistence
    session_manager: Option<SessionManager>,
    /// Current session ID
    session_id: Option<String>,
    /// Page index from session (rich metadata - NOT content)
    page_index: HashMap<String, PageIndex>,
}

impl Agent {
    /// Create a new agent with default configuration
    pub fn new() -> Self {
        Self::with_config(AgentConfig::default())
    }

    /// Create a new agent with custom configuration
    pub fn with_config(config: AgentConfig) -> Self {
        Self {
            provider: BridgeProvider::local(),
            schema: VmSchema::new(),
            full_trace: Vec::new(),
            config,
            session_manager: None,
            session_id: None,
            page_index: HashMap::new(),
        }
    }

    /// Get the execution trace
    pub fn trace(&self) -> &[llcraft_vm::ExecutionStep] {
        &self.full_trace
    }

    /// Enable session persistence
    pub fn with_session(mut self, session_id: Option<&str>) -> Result<Self, String> {
        let manager =
            SessionManager::new(&self.config.session_dir).map_err(|e| e.to_string())?;

        let (sid, new_page_index) = if let Some(id) = session_id {
            if manager.session_exists(id) {
                let session = manager.load_session(id).map_err(|e| e.to_string())?;
                let mut page_index = HashMap::new();

                if self.config.verbose {
                    println!("Resuming session: {}", id);
                    println!("   Previous task: {}", session.metadata.task);
                    println!("   Available pages (use LOAD_PAGE to fetch content):");
                }

                let total_tokens: usize =
                    session.page_index.values().map(|idx| idx.tokens).sum();

                for (page_id, idx) in &session.page_index {
                    if self.config.verbose {
                        println!(
                            "     - {} (~{} tokens): {}",
                            page_id, idx.tokens, idx.summary
                        );
                    }
                    page_index.insert(page_id.clone(), idx.clone());
                }

                if self.config.verbose && !page_index.is_empty() {
                    println!(
                        "   {} pages indexed (~{} total tokens, content NOT loaded)\n",
                        page_index.len(),
                        total_tokens
                    );
                }

                (id.to_string(), page_index)
            } else {
                if self.config.verbose {
                    println!("Creating new session: {}", id);
                }
                let session = Session::new(id, "agent session");
                manager.save_session(&session).map_err(|e| e.to_string())?;
                (id.to_string(), HashMap::new())
            }
        } else {
            let session = manager
                .create_session("agent session")
                .map_err(|e| e.to_string())?;
            if self.config.verbose {
                println!("Created new session: {}", session.metadata.id);
            }
            (session.metadata.id.clone(), HashMap::new())
        };

        self.session_manager = Some(manager);
        self.session_id = Some(sid);
        self.page_index = new_page_index;

        Ok(self)
    }

    /// Run a task to completion
    pub async fn run(&mut self, task: &str) -> Result<AgentResult, String> {
        if self.config.verbose {
            println!("Task: {}\n", task);

            if !self.page_index.is_empty() {
                let total_tokens: usize = self.page_index.values().map(|idx| idx.tokens).sum();
                println!(
                    "Available pages in session (~{} tokens total, use LOAD_PAGE to fetch):",
                    total_tokens
                );
                for (page_id, idx) in &self.page_index {
                    println!("   - {} (~{} tokens): {}", page_id, idx.tokens, idx.summary);
                }
                println!();
            }
        }

        let program = self.generate_program(task).await?;

        if self.config.verbose {
            println!("Generated Program:");
            program.pretty_print();
        }

        self.run_program(program).await
    }

    /// Generate a program from the LLM based on the task
    async fn generate_program(&mut self, task: &str) -> Result<Program, String> {
        let system = self.schema.system_prompt().to_string();
        let user = self.schema.user_prompt(task, self.page_index.iter(), &self.full_trace);

        if self.config.verbose {
            println!("Asking LLM to generate program...");
            if !self.full_trace.is_empty() {
                println!(
                    "   (with {} previous execution steps as context)",
                    self.full_trace.len()
                );
            }
            if !self.page_index.is_empty() {
                println!(
                    "   (with {} page summaries from session)",
                    self.page_index.len()
                );
            }
        }

        let completion_request = CompletionRequest::new(vec![
            ChatMessage::system(&system),
            ChatMessage::user(&user),
        ]);

        let response = self
            .provider
            .complete(completion_request)
            .await
            .map_err(|e| format!("LLM error: {:?}", e))?;

        let content = response.content.ok_or("Empty LLM response")?;

        if self.config.verbose {
            println!("   Response: {} chars", content.len());
        }

        self.parse_program(&content)
    }

    /// Parse a program from LLM output (handles markdown fences)
    fn parse_program(&self, content: &str) -> Result<Program, String> {
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

        if let (Some(_), Some(ref session_id)) = (&self.session_manager, &self.session_id) {
            let interp_manager =
                SessionManager::new(&self.config.session_dir).map_err(|e| e.to_string())?;
            interp = interp.with_session_manager(interp_manager);
            interp
                .resume_session(session_id)
                .map_err(|e| e.to_string())?;

            if self.config.verbose {
                println!(
                    "   Session connected - LOAD_PAGE enabled for: {}",
                    session_id
                );
            }
        }

        if self.config.verbose {
            interp = interp.with_log_callback(|level, msg| {
                println!("   [{:?}] {}", level, msg);
            });
        }

        loop {
            match interp.run().map_err(|e| e.to_string())? {
                ExecutionResult::Complete(result) => {
                    self.full_trace.extend(interp.trace().iter().cloned());

                    if self.config.verbose {
                        println!("\nTask completed!");
                    }

                    let pages = self.collect_pages(&interp);
                    self.save_to_session(&pages)?;

                    return Ok(AgentResult { result, pages });
                }
                ExecutionResult::Failed(error) => {
                    self.full_trace.extend(interp.trace().iter().cloned());
                    return Err(error);
                }
                ExecutionResult::NeedsLlm(request) => {
                    if let LlmRequestType::Inject { .. } = &request.request_type {
                        let opcodes = self.handle_inject_request(&request, &interp).await?;
                        let count = interp.inject_opcodes(opcodes).map_err(|e| e.to_string())?;
                        if self.config.verbose {
                            println!("   Injected {} opcodes", count);
                        }
                    } else if let LlmRequestType::InferBatch {
                        prompts,
                        context,
                        store_prefix,
                        store_combined,
                        ..
                    } = &request.request_type
                    {
                        let results = self
                            .handle_infer_batch_request(prompts, context, store_prefix)
                            .await?;

                        for (i, result) in results.iter().enumerate() {
                            let page_id = format!("{}_{}", store_prefix, i);
                            interp
                                .provide_llm_response(result.clone(), &page_id)
                                .map_err(|e| e.to_string())?;
                        }

                        if let Some(combined_page) = store_combined {
                            let combined = serde_json::json!({
                                "results": results,
                                "count": results.len(),
                                "success": true
                            });
                            interp
                                .provide_llm_response(combined, combined_page)
                                .map_err(|e| e.to_string())?;
                        }
                    } else {
                        let response = self.handle_llm_request(&request, &interp).await?;
                        interp
                            .provide_llm_response(response, &request.store_to)
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
    fn save_to_session(
        &mut self,
        pages: &HashMap<String, serde_json::Value>,
    ) -> Result<(), String> {
        if let (Some(manager), Some(session_id)) = (&self.session_manager, &self.session_id) {
            let mut session = manager.load_session(session_id).map_err(|e| e.to_string())?;

            for (page_id, content) in pages {
                let page = MemoryPage::new(page_id, content.clone());
                let summary = summarize_value(content);
                session.index_page(&page, Some(summary.clone()));
                manager
                    .save_page(session_id, &page)
                    .map_err(|e| e.to_string())?;

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

            manager.save_session(&session).map_err(|e| e.to_string())?;

            if self.config.verbose {
                println!("   Saved {} pages to session", pages.len());
            }
        }
        Ok(())
    }

    /// Collect all pages from interpreter for final result
    fn collect_pages(
        &self,
        interp: &Interpreter<DefaultSyscallHandler>,
    ) -> HashMap<String, serde_json::Value> {
        interp.all_pages()
    }

    /// Handle an LLM request from the interpreter
    async fn handle_llm_request(
        &self,
        request: &LlmRequest,
        interp: &Interpreter<DefaultSyscallHandler>,
    ) -> Result<serde_json::Value, String> {
        if self.config.verbose {
            println!("\n   LLM Request ({:?})", request.request_type);
            println!("      Prompt: {}", truncate(&request.prompt, 60));
        }

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
                    "# Planning Request\n\n{}\n\n## Context:\n{}\n\n\
                     Generate a plan as JSON with steps.",
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
            LlmRequestType::Inject { .. } => {
                unreachable!("INJECT should be handled by handle_inject_request");
            }
            LlmRequestType::InferBatch { .. } => {
                unreachable!("INFER_BATCH should be handled by handle_infer_batch_request");
            }
        };

        let completion_request = CompletionRequest::new(vec![ChatMessage::user(prompt)]);

        let response = self
            .provider
            .complete(completion_request)
            .await
            .map_err(|e| format!("LLM error: {:?}", e))?;

        let content = response.content.ok_or("Empty LLM response")?;

        if self.config.verbose {
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
        if self.config.verbose {
            println!("\n   INJECT Request");
            println!("      Goal: {}", truncate(&request.prompt, 60));
        }

        let mut context = String::new();
        for page_id in &request.context_pages {
            if let Some(content) = interp.get_page(page_id) {
                context.push_str(&format!("### Page: {}\n{}\n\n", page_id, content));
            }
        }

        let (include_trace, include_memory) = match &request.request_type {
            LlmRequestType::Inject {
                include_trace,
                include_memory,
            } => (*include_trace, *include_memory),
            _ => (false, false),
        };

        let trace_text = if include_trace {
            let trace: Vec<String> = interp
                .trace()
                .iter()
                .map(|s| format!("{}: {} -> {}", s.step, s.opcode, s.result))
                .collect();
            format!("\n\n## Execution Trace:\n{}", trace.join("\n"))
        } else {
            String::new()
        };

        let memory_text = if include_memory {
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
- EXEC: {{"op": "EXEC", "command": "<shell cmd>", "store_to": "<page>"}}
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

        let completion_request = CompletionRequest::new(vec![ChatMessage::user(prompt)]);

        let response = self
            .provider
            .complete(completion_request)
            .await
            .map_err(|e| format!("LLM error: {:?}", e))?;

        let content = response.content.ok_or("Empty LLM response")?;

        if self.config.verbose {
            println!("      Response: {} chars", content.len());
        }

        self.parse_opcodes(&content)
    }

    /// Parse opcodes from LLM output (handles markdown fences)
    fn parse_opcodes(&self, content: &str) -> Result<Vec<Opcode>, String> {
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
    async fn handle_infer_batch_request(
        &self,
        prompts: &[String],
        context: &[serde_json::Value],
        store_prefix: &str,
    ) -> Result<Vec<serde_json::Value>, String> {
        if self.config.verbose {
            println!("\n   INFER_BATCH Request");
            println!("      Running {} prompts...", prompts.len());
        }

        let context_text: String = context
            .iter()
            .enumerate()
            .map(|(i, v)| {
                format!(
                    "### Context {}\n{}\n",
                    i,
                    serde_json::to_string_pretty(v).unwrap_or_default()
                )
            })
            .collect();

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

            if self.config.verbose {
                println!(
                    "      [{}/{}] {} → {}",
                    i + 1,
                    prompts.len(),
                    store_prefix,
                    if results
                        .last()
                        .map(|r| r["success"].as_bool().unwrap_or(false))
                        .unwrap_or(false)
                    {
                        "ok"
                    } else {
                        "err"
                    }
                );
            }
        }

        if self.config.verbose {
            let successes = results
                .iter()
                .filter(|r| r["success"].as_bool().unwrap_or(false))
                .count();
            println!("      Completed: {}/{} successful", successes, results.len());
        }

        Ok(results)
    }
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
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
