//! # VM Schema for LLM Agent Execution
//!
//! This module provides a structured description of the LLcraft VM.
//! The VM enables LLMs to act as intelligent agents by providing:
//! - Efficient context window management through page-based memory
//! - Tool integration via syscalls
//! - Multi-step execution with state persistence
//! - Self-reflection and planning capabilities
//!
//! ## Philosophy
//! The LLM is the CPU - it thinks, reasons, and decides.
//! The VM is the instruction set - it executes actions efficiently.
//! Programs are plans - sequences of operations to accomplish goals.
//!
//! Since each LLM invocation is stateless, the VM maintains state between
//! calls and provides execution history for continuity.

use serde::{Deserialize, Serialize};

/// Complete VM schema - everything an LLM needs to know to generate programs
#[derive(Debug, Clone, Serialize)]
pub struct VmSchema {
    /// Schema version
    pub version: &'static str,
    /// Brief description of the VM
    pub description: &'static str,
    /// Available opcodes grouped by category
    pub opcodes: Vec<OpcodeCategory>,
    /// Description of VM state (what the model can work with)
    pub state: VmStateSchema,
    /// Execution model explanation
    pub execution: ExecutionModel,
    /// Best practices for program generation
    pub guidelines: Vec<Guideline>,
}

impl Default for VmSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl VmSchema {
    pub fn new() -> Self {
        Self {
            version: "0.1.0",
            description: "LLcraft VM - You are the CPU, the VM is your instruction set.\n\n\
                         You solve tasks by generating programs that the interpreter executes.\n\
                         The VM handles context window management, tool execution, and state persistence.\n\
                         Since each of your invocations is stateless, the VM tracks execution history\n\
                         and provides it to you for continuity across steps.",
            opcodes: Self::define_opcodes(),
            state: VmStateSchema::default(),
            execution: ExecutionModel::default(),
            guidelines: Self::define_guidelines(),
        }
    }

    /// Render as a prompt-friendly string for the LLM
    pub fn to_prompt(&self) -> String {
        let mut out = String::new();

        out.push_str("# LLcraft VM Specification\n\n");
        out.push_str(self.description);
        out.push_str("\n\n");

        // State description
        out.push_str("## VM State\n\n");
        out.push_str(&format!("**Stack**: {} (max {} items)\n",
            self.state.stack.description, self.state.stack.max_size));
        out.push_str(&format!("**Memory**: {} (max {} pages, ~{} tokens each)\n",
            self.state.memory.description,
            self.state.memory.max_pages,
            self.state.memory.page_size_tokens));
        out.push_str(&format!("**Registers**: {}\n\n", self.state.registers.description));

        // Opcodes
        out.push_str("## Opcodes\n\n");
        for category in &self.opcodes {
            out.push_str(&format!("### {}\n", category.name));
            out.push_str(&format!("{}\n\n", category.description));
            for op in &category.opcodes {
                out.push_str(&format!("- **{}**: {}\n", op.name, op.description));
                if !op.params.is_empty() {
                    out.push_str(&format!("  - Params: {}\n", op.params.join(", ")));
                }
                if let Some(example) = &op.example {
                    out.push_str(&format!("  - Example: `{}`\n", example));
                }
            }
            out.push('\n');
        }

        // Guidelines
        out.push_str("## Guidelines\n\n");
        for g in &self.guidelines {
            out.push_str(&format!("### {}\n{}\n\n", g.title, g.content));
        }

        out
    }

    /// Render as JSON for structured consumption
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    fn define_opcodes() -> Vec<OpcodeCategory> {
        vec![
            OpcodeCategory {
                name: "Memory",
                description: "Page-based memory for context management. Each page holds JSON data.",
                opcodes: vec![
                    OpcodeSpec {
                        name: "LOAD",
                        description: "Load a page into working memory",
                        params: vec!["page_id: string", "range?: {start, end}"],
                        example: Some(r#"{"op": "LOAD", "page_id": "context"}"#),
                    },
                    OpcodeSpec {
                        name: "STORE",
                        description: "Store data to a page (creates if not exists)",
                        params: vec!["page_id: string", "data: any"],
                        example: Some(r#"{"op": "STORE", "page_id": "result", "data": {"key": "value"}}"#),
                    },
                    OpcodeSpec {
                        name: "ALLOC",
                        description: "Allocate a new empty page",
                        params: vec!["size_hint?: number", "label?: string"],
                        example: Some(r#"{"op": "ALLOC", "label": "scratch"}"#),
                    },
                    OpcodeSpec {
                        name: "FREE",
                        description: "Free a page from memory",
                        params: vec!["page_id: string"],
                        example: Some(r#"{"op": "FREE", "page_id": "temp"}"#),
                    },
                    OpcodeSpec {
                        name: "COPY",
                        description: "Copy data between pages",
                        params: vec!["src: string", "dst: string", "range?: {start, end}"],
                        example: Some(r#"{"op": "COPY", "src": "input", "dst": "backup"}"#),
                    },
                ],
            },
            OpcodeCategory {
                name: "Thinking",
                description: "LLM reasoning operations - use these when you need to think, analyze, or decide",
                opcodes: vec![
                    OpcodeSpec {
                        name: "INFER",
                        description: "General LLM inference - think about a prompt with context",
                        params: vec![
                            "prompt: string",
                            "context: string[]",
                            "store_to: string",
                            "params?: {temperature, max_tokens, model}",
                        ],
                        example: Some(r#"{"op": "INFER", "prompt": "Analyze this code for bugs", "context": ["code"], "store_to": "analysis"}"#),
                    },
                    OpcodeSpec {
                        name: "PLAN",
                        description: "Generate a plan or next steps based on current state",
                        params: vec!["goal: string", "context: string[]", "store_to: string"],
                        example: Some(r#"{"op": "PLAN", "goal": "How should I refactor this function?", "context": ["code", "requirements"], "store_to": "plan"}"#),
                    },
                    OpcodeSpec {
                        name: "REFLECT",
                        description: "Analyze what happened and decide what to do next (receives execution trace)",
                        params: vec!["question: string", "include_trace: bool", "store_to: string"],
                        example: Some(r#"{"op": "REFLECT", "question": "Did the edit succeed? What should I do next?", "include_trace": true, "store_to": "reflection"}"#),
                    },
                ],
            },
            OpcodeCategory {
                name: "Context Management",
                description: "Manage context window efficiently - compress, chunk, merge data",
                opcodes: vec![
                    OpcodeSpec {
                        name: "SUMMARIZE",
                        description: "Compress pages to fit context window",
                        params: vec!["pages: string[]", "store_to: string", "target_tokens?: number"],
                        example: Some(r#"{"op": "SUMMARIZE", "pages": ["doc1", "doc2"], "store_to": "summary"}"#),
                    },
                    OpcodeSpec {
                        name: "CHUNK",
                        description: "Split large content into smaller pages",
                        params: vec!["source: string", "chunk_size: number", "prefix?: string"],
                        example: Some(r#"{"op": "CHUNK", "source": "large_file", "chunk_size": 2000}"#),
                    },
                    OpcodeSpec {
                        name: "MERGE",
                        description: "Combine multiple pages into one",
                        params: vec!["pages: string[]", "store_to: string", "separator?: string"],
                        example: Some(r#"{"op": "MERGE", "pages": ["part1", "part2"], "store_to": "combined"}"#),
                    },
                ],
            },
            OpcodeCategory {
                name: "Control Flow",
                description: "Program execution control",
                opcodes: vec![
                    OpcodeSpec {
                        name: "LABEL",
                        description: "Define a jump target",
                        params: vec!["name: string"],
                        example: Some(r#"{"op": "LABEL", "name": "loop_start"}"#),
                    },
                    OpcodeSpec {
                        name: "JUMP",
                        description: "Unconditional jump to label",
                        params: vec!["target: string"],
                        example: Some(r#"{"op": "JUMP", "target": "loop_start"}"#),
                    },
                    OpcodeSpec {
                        name: "BRANCH",
                        description: "Conditional branch based on condition",
                        params: vec!["condition: string", "if_true: string", "if_false: string"],
                        example: Some(r#"{"op": "BRANCH", "condition": "result.is_empty", "if_true": "retry", "if_false": "done"}"#),
                    },
                    OpcodeSpec {
                        name: "CALL",
                        description: "Call a subprogram",
                        params: vec!["program_id: string", "args?: any"],
                        example: Some(r#"{"op": "CALL", "program_id": "analyze_function", "args": {"name": "main"}}"#),
                    },
                    OpcodeSpec {
                        name: "RETURN",
                        description: "Return from subprogram",
                        params: vec!["value?: any"],
                        example: Some(r#"{"op": "RETURN", "value": {"status": "ok"}}"#),
                    },
                    OpcodeSpec {
                        name: "LOOP",
                        description: "Iterate over items",
                        params: vec!["var: string", "over: string", "body: opcode[]"],
                        example: Some(r#"{"op": "LOOP", "var": "file", "over": "files", "body": [...]}"#),
                    },
                    OpcodeSpec {
                        name: "COMPLETE",
                        description: "Successfully finish execution with result",
                        params: vec!["result: any"],
                        example: Some(r#"{"op": "COMPLETE", "result": {"answer": "42"}}"#),
                    },
                    OpcodeSpec {
                        name: "FAIL",
                        description: "Fail execution with error",
                        params: vec!["error: string"],
                        example: Some(r#"{"op": "FAIL", "error": "Could not parse input"}"#),
                    },
                ],
            },
            OpcodeCategory {
                name: "Stack",
                description: "Working value stack for intermediate computations",
                opcodes: vec![
                    OpcodeSpec {
                        name: "PUSH",
                        description: "Push value onto stack",
                        params: vec!["value: any"],
                        example: Some(r#"{"op": "PUSH", "value": 42}"#),
                    },
                    OpcodeSpec {
                        name: "PUSH_PAGE",
                        description: "Push page contents onto stack",
                        params: vec!["page_id: string"],
                        example: Some(r#"{"op": "PUSH_PAGE", "page_id": "result"}"#),
                    },
                    OpcodeSpec {
                        name: "POP",
                        description: "Pop and discard top value",
                        params: vec![],
                        example: Some(r#"{"op": "POP"}"#),
                    },
                    OpcodeSpec {
                        name: "POP_TO",
                        description: "Pop top value into a page",
                        params: vec!["store_to: string"],
                        example: Some(r#"{"op": "POP_TO", "store_to": "output"}"#),
                    },
                    OpcodeSpec {
                        name: "DUP",
                        description: "Duplicate top value",
                        params: vec![],
                        example: Some(r#"{"op": "DUP"}"#),
                    },
                    OpcodeSpec {
                        name: "SWAP",
                        description: "Swap top two values",
                        params: vec![],
                        example: Some(r#"{"op": "SWAP"}"#),
                    },
                ],
            },
            OpcodeCategory {
                name: "Tools",
                description: "External tool operations - file I/O, shell commands, search",
                opcodes: vec![
                    OpcodeSpec {
                        name: "READ_FILE",
                        description: "Read a file's contents",
                        params: vec!["path: string", "store_to: string"],
                        example: Some(r#"{"op": "READ_FILE", "path": "src/main.rs", "store_to": "code"}"#),
                    },
                    OpcodeSpec {
                        name: "WRITE_FILE",
                        description: "Write content to a file",
                        params: vec!["path: string", "content: string", "store_to?: string"],
                        example: Some(r#"{"op": "WRITE_FILE", "path": "output.txt", "content": "Hello", "store_to": "result"}"#),
                    },
                    OpcodeSpec {
                        name: "LIST_DIR",
                        description: "List files in a directory",
                        params: vec!["path: string", "store_to: string"],
                        example: Some(r#"{"op": "LIST_DIR", "path": "src", "store_to": "files"}"#),
                    },
                    OpcodeSpec {
                        name: "EXEC",
                        description: "Execute a shell command",
                        params: vec!["command: string", "store_to: string"],
                        example: Some(r#"{"op": "EXEC", "command": "find . -name '*.rs'", "store_to": "result"}"#),
                    },
                    OpcodeSpec {
                        name: "GREP",
                        description: "Search for a pattern in files",
                        params: vec!["pattern: string", "path: string", "store_to: string"],
                        example: Some(r#"{"op": "GREP", "pattern": "fn main", "path": "src/", "store_to": "matches"}"#),
                    },
                ],
            },
            OpcodeCategory {
                name: "Debug",
                description: "Debugging and checkpointing",
                opcodes: vec![
                    OpcodeSpec {
                        name: "LOG",
                        description: "Log a debug message",
                        params: vec!["level: debug|info|warn|error", "message: string"],
                        example: Some(r#"{"op": "LOG", "level": "info", "message": "Processing file"}"#),
                    },
                    OpcodeSpec {
                        name: "CHECKPOINT",
                        description: "Save state for potential rollback",
                        params: vec!["name: string"],
                        example: Some(r#"{"op": "CHECKPOINT", "name": "before_edit"}"#),
                    },
                    OpcodeSpec {
                        name: "ASSERT",
                        description: "Assert condition, fail if false",
                        params: vec!["condition: string", "message: string"],
                        example: Some(r#"{"op": "ASSERT", "condition": "result.success", "message": "Expected success"}"#),
                    },
                ],
            },
        ]
    }

    fn define_guidelines() -> Vec<Guideline> {
        vec![
            Guideline {
                title: "You Are the Brain",
                content: "You are the CPU of this system. The VM executes your programs, \
                         but you make all the decisions. Use INFER/PLAN/REFLECT when you need to think. \
                         Use SYSCALL when you need to act. Use BRANCH to handle different outcomes.",
            },
            Guideline {
                title: "Stateless But Continuous",
                content: "Each of your invocations is stateless - you won't remember previous calls. \
                         The VM maintains state for you in memory pages. Use REFLECT with include_trace=true \
                         to see what happened before. Store important context in pages for future steps.",
            },
            Guideline {
                title: "Context Window Management",
                content: "Your context window is limited. Use SUMMARIZE to compress information, \
                         CHUNK to split large inputs, and FREE to release unused pages. \
                         Always estimate token usage before loading large data.",
            },
            Guideline {
                title: "Tool Usage",
                content: "Use tool opcodes for external operations: READ_FILE, WRITE_FILE, LIST_DIR, EXEC, GREP. \
                         Results are stored to pages with {success: bool, ...data}. \
                         Always check results with BRANCH on 'page.success' and handle errors.",
            },
            Guideline {
                title: "Program Structure",
                content: "Start with LABEL 'entry'. End with COMPLETE containing the final result \
                         or FAIL with a clear error. Use meaningful page names like 'file_content', \
                         'analysis', 'plan'. Log important steps for debugging.",
            },
        ]
    }
}

/// A category of opcodes
#[derive(Debug, Clone, Serialize)]
pub struct OpcodeCategory {
    pub name: &'static str,
    pub description: &'static str,
    pub opcodes: Vec<OpcodeSpec>,
}

/// Specification for a single opcode
#[derive(Debug, Clone, Serialize)]
pub struct OpcodeSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub params: Vec<&'static str>,
    pub example: Option<&'static str>,
}

/// Description of VM state
#[derive(Debug, Clone, Serialize)]
pub struct VmStateSchema {
    pub stack: StackSchema,
    pub memory: MemorySchema,
    pub registers: RegisterSchema,
}

impl Default for VmStateSchema {
    fn default() -> Self {
        Self {
            stack: StackSchema {
                description: "LIFO stack for working values (JSON)",
                max_size: 256,
            },
            memory: MemorySchema {
                description: "Named pages holding JSON data",
                max_pages: 1024,
                page_size_tokens: 4096,
            },
            registers: RegisterSchema {
                description: "Named registers: pc (program counter), goal, focus, thought, flags",
                registers: vec!["pc", "goal", "focus", "thought", "flags", "sp"],
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StackSchema {
    pub description: &'static str,
    pub max_size: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySchema {
    pub description: &'static str,
    pub max_pages: usize,
    pub page_size_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisterSchema {
    pub description: &'static str,
    pub registers: Vec<&'static str>,
}

/// Execution model description
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionModel {
    pub description: &'static str,
    pub flow: Vec<&'static str>,
}

impl Default for ExecutionModel {
    fn default() -> Self {
        Self {
            description: "The interpreter runs your program. When it needs your input (INFER/PLAN/REFLECT), \
                         it calls you with the relevant context. Your response is stored and execution continues.",
            flow: vec![
                "1. You receive a task and generate a program",
                "2. The interpreter executes your program step by step",
                "3. SYSCALL operations interact with external tools",
                "4. INFER/PLAN/REFLECT pause execution and call you with context",
                "5. Your response is stored in the specified page",
                "6. COMPLETE returns the final result, FAIL reports an error",
                "7. If the program needs continuation, REFLECT with execution trace helps you decide next steps",
            ],
        }
    }
}

/// A guideline for program generation
#[derive(Debug, Clone, Serialize)]
pub struct Guideline {
    pub title: &'static str,
    pub content: &'static str,
}

// ============================================================================
// Task Request - What the user wants solved
// ============================================================================

/// A task request that an LLM should solve by generating a program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    /// The user's task description
    pub task: String,
    /// Available context (file contents, previous results, etc.)
    pub context: Vec<ContextItem>,
    /// Constraints on the solution
    pub constraints: TaskConstraints,
    /// Expected output format
    pub output_format: OutputFormat,
    /// Execution history from previous steps (for continuation)
    #[serde(default)]
    pub execution_trace: Vec<ExecutionStep>,
}

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

impl TaskRequest {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            context: vec![],
            constraints: TaskConstraints::default(),
            output_format: OutputFormat::default(),
            execution_trace: vec![],
        }
    }

    pub fn with_context(mut self, name: impl Into<String>, content: impl Into<String>) -> Self {
        self.context.push(ContextItem {
            name: name.into(),
            content: content.into(),
            tokens: None,
        });
        self
    }

    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.constraints.max_context_tokens = Some(max);
        self
    }

    pub fn with_trace(mut self, trace: Vec<ExecutionStep>) -> Self {
        self.execution_trace = trace;
        self
    }

    /// Generate the system prompt (VM specification)
    /// This should be used as the system message
    pub fn system_prompt(schema: &VmSchema) -> String {
        let mut out = String::new();

        out.push_str("# LLcraft VM Agent\n\n");
        out.push_str("You are an intelligent agent that solves tasks by generating VM programs.\n");
        out.push_str("The interpreter will execute your program and call you when it needs your input.\n\n");
        out.push_str(&schema.to_prompt());
        out.push_str("\n## Output Format\n");
        out.push_str("Output a JSON program with fields: id, name, description, code (array of opcodes).\n");
        out.push_str("Output ONLY valid JSON, no markdown fences or explanation.\n");

        out
    }

    /// Generate the user prompt (just the task)
    /// This should be used as the user message
    pub fn user_prompt(&self) -> String {
        let mut out = String::new();

        out.push_str("# Task\n\n");
        out.push_str(&self.task);
        out.push_str("\n");

        if !self.context.is_empty() {
            out.push_str("\n## Available Pages\n");
            for ctx in &self.context {
                out.push_str(&format!("- `{}`: {} chars", ctx.name, ctx.content.len()));
                if let Some(tokens) = ctx.tokens {
                    out.push_str(&format!(" (~{} tokens)", tokens));
                }
                out.push('\n');
            }
        }

        if !self.execution_trace.is_empty() {
            out.push_str("\n## Execution History\n");
            out.push_str("These steps have already been executed:\n\n");
            for step in &self.execution_trace {
                if let Some(err) = &step.error {
                    out.push_str(&format!("{}. {} → ERROR: {}\n", step.step, step.opcode, err));
                } else {
                    out.push_str(&format!("{}. {} → {}\n", step.step, step.opcode, step.result));
                }
            }
            out.push_str("\nContinue from where execution left off.\n");
        }

        if let Some(max) = self.constraints.max_context_tokens {
            out.push_str(&format!("\n**Constraint**: Max context tokens: {}\n", max));
        }

        out
    }

    /// Render as combined prompt (legacy, for backwards compatibility)
    pub fn to_prompt(&self, schema: &VmSchema) -> String {
        let mut out = String::new();

        out.push_str("# Task: Generate an LLcraft VM Program\n\n");
        out.push_str("## User Request\n");
        out.push_str(&self.task);
        out.push_str("\n\n");

        if !self.context.is_empty() {
            out.push_str("## Available Context\n");
            for ctx in &self.context {
                out.push_str(&format!("- **{}**: {} chars", ctx.name, ctx.content.len()));
                if let Some(tokens) = ctx.tokens {
                    out.push_str(&format!(" (~{} tokens)", tokens));
                }
                out.push('\n');
            }
            out.push('\n');
        }

        if !self.execution_trace.is_empty() {
            out.push_str("## Execution History\n");
            for step in &self.execution_trace {
                if let Some(err) = &step.error {
                    out.push_str(&format!("{}. {} → ERROR: {}\n", step.step, step.opcode, err));
                } else {
                    out.push_str(&format!("{}. {} → {}\n", step.step, step.opcode, step.result));
                }
            }
            out.push('\n');
        }

        out.push_str("## Constraints\n");
        if let Some(max) = self.constraints.max_context_tokens {
            out.push_str(&format!("- Max context tokens: {}\n", max));
        }
        if let Some(max) = self.constraints.max_infer_calls {
            out.push_str(&format!("- Max inference calls: {}\n", max));
        }
        out.push('\n');

        out.push_str("## VM Specification\n\n");
        out.push_str(&schema.to_prompt());
        out.push_str("Generate a valid LLcraft VM program (JSON) that solves the user's request. ");
        out.push_str("The program should efficiently manage the context window and produce the expected output.\n\n");
        out.push_str("Output the program as a JSON object with fields: id, name, description, code (array of opcodes).\n");

        out
    }
}

/// A piece of context available to the task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub name: String,
    pub content: String,
    pub tokens: Option<usize>,
}

/// Constraints on task execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskConstraints {
    /// Maximum tokens for context window
    pub max_context_tokens: Option<usize>,
    /// Maximum number of INFER calls
    pub max_infer_calls: Option<usize>,
    /// Allowed syscalls (None = all allowed)
    pub allowed_syscalls: Option<Vec<String>>,
    /// Time limit in seconds
    pub timeout_secs: Option<u64>,
}

/// Expected output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputFormat {
    /// Description of expected output
    pub description: String,
    /// JSON schema for structured output (optional)
    pub schema: Option<serde_json::Value>,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self {
            description: "Final result stored in a page and returned via COMPLETE".to_string(),
            schema: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_generation() {
        let schema = VmSchema::new();
        let prompt = schema.to_prompt();

        assert!(prompt.contains("LLcraft VM"));
        assert!(prompt.contains("INFER"));
        assert!(prompt.contains("STORE"));
        assert!(prompt.contains("Context Window Management"));

        println!("{}", prompt);
    }

    #[test]
    fn test_task_request() {
        let schema = VmSchema::new();
        let task = TaskRequest::new("Analyze this Rust file for bugs")
            .with_context("code", "fn main() { println!(\"Hello\"); }")
            .with_max_tokens(8000);

        let prompt = task.to_prompt(&schema);

        assert!(prompt.contains("Analyze this Rust file"));
        assert!(prompt.contains("code"));
        assert!(prompt.contains("8000"));

        println!("{}", prompt);
    }

    #[test]
    fn test_schema_json() {
        let schema = VmSchema::new();
        let json = schema.to_json();

        // Should be valid JSON
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();

        println!("{}", json);
    }
}
