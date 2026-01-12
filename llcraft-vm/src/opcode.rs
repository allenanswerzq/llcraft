//! # LLM-VM Opcodes
//!
//! Opcodes for the LLM Virtual Machine - treating LLMs as compute units.
//! These opcodes define the instruction set for LLM cognition.
//!
//! ## Design Philosophy
//! - The LLM is the CPU, not the program
//! - Programs are sequences of opcodes that orchestrate LLM computation
//! - Memory is organized as pages that can be loaded/stored
//! - Syscalls provide controlled access to external tools

use serde::{Deserialize, Serialize};

/// LLM-VM Opcode - the instruction set for LLM cognition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Opcode {
    // =========================================================================
    // MEMORY OPERATIONS - Page-based memory management
    // =========================================================================

    /// Load a page from storage into the heap
    /// Analogous to loading data into RAM
    Load {
        /// Page identifier to load
        page_id: String,
        /// Optional: specific range within the page
        #[serde(default)]
        range: Option<Range>,
    },

    /// Store data to a page
    /// Marks the page as dirty for later persistence
    Store {
        /// Page identifier to write to
        page_id: String,
        /// Data to store
        data: serde_json::Value,
    },

    /// Allocate a new page
    /// Returns the page ID of the newly allocated page
    Alloc {
        /// Hint for page size (in tokens)
        #[serde(default)]
        size_hint: Option<usize>,
        /// Optional label for the page
        #[serde(default)]
        label: Option<String>,
    },

    /// Free a page from the heap
    /// The page may be persisted to storage if dirty
    Free {
        /// Page identifier to free
        page_id: String,
    },

    /// Copy data between pages
    Copy {
        /// Source page
        src: String,
        /// Destination page
        dst: String,
        /// Optional range to copy
        #[serde(default)]
        range: Option<Range>,
    },

    // =========================================================================
    // CONTROL FLOW - Process and execution management
    // =========================================================================

    /// Call a subprogram (push new frame onto stack)
    /// Like a function call - saves current state and jumps to new program
    Call {
        /// Program/task identifier to call
        program_id: String,
        /// Arguments to pass to the program
        #[serde(default)]
        args: serde_json::Value,
    },

    /// Return from a subprogram (pop frame from stack)
    /// Restores previous state and continues execution
    Return {
        /// Return value
        #[serde(default)]
        value: serde_json::Value,
    },

    /// Yield execution - give up the CPU
    /// Process remains ready but allows other processes to run
    Yield,

    /// Complete the current task successfully
    /// Terminal instruction - process exits with result
    Complete {
        /// Final result of the task
        result: serde_json::Value,
    },

    /// Fail the current task with an error
    /// Terminal instruction - process exits with error
    Fail {
        /// Error message
        error: String,
    },

    /// Conditional branch
    Branch {
        /// Condition to evaluate (references a page or value)
        condition: String,
        /// Target if condition is true
        if_true: String,
        /// Target if condition is false
        if_false: String,
    },

    /// Unconditional jump to a label
    Jump {
        /// Label to jump to
        target: String,
    },

    /// Define a label (jump target)
    Label {
        /// Label name
        name: String,
    },

    /// Loop construct
    Loop {
        /// Iterator variable name
        var: String,
        /// Iterable (page_id or inline list)
        over: String,
        /// Body opcodes (inline program)
        #[serde(default)]
        body: Vec<Opcode>,
    },

    // =========================================================================
    // SYSCALLS - External tool invocations (legacy, prefer explicit tool ops)
    // =========================================================================

    /// Invoke a syscall (external tool) - legacy, prefer specific tool opcodes
    Syscall {
        /// Syscall name (read_file, grep, exec_code, llm_query, etc.)
        call: String,
        /// Arguments to the syscall
        #[serde(default)]
        args: serde_json::Value,
        /// Page to store the result
        #[serde(default)]
        store_to: Option<String>,
    },

    // =========================================================================
    // TOOLS - Explicit external tool operations
    // =========================================================================

    /// Read a file's contents
    ReadFile {
        /// Path to the file
        path: String,
        /// Page to store result {success, content, path}
        store_to: String,
    },

    /// Write content to a file
    WriteFile {
        /// Path to the file
        path: String,
        /// Content to write
        content: String,
        /// Page to store result {success, path}
        #[serde(default)]
        store_to: Option<String>,
    },

    /// List files in a directory
    ListDir {
        /// Path to directory
        path: String,
        /// Page to store result {success, files, path}
        store_to: String,
    },

    /// Execute a shell command
    Exec {
        /// Shell command to execute
        command: String,
        /// Page to store result {success, stdout, stderr, exit_code}
        store_to: String,
    },

    /// Search for a pattern in files
    Grep {
        /// Pattern to search for
        pattern: String,
        /// Path to search in
        path: String,
        /// Page to store result {success, matches, count}
        store_to: String,
    },

    /// Wait for an async syscall to complete
    Wait {
        /// Handle returned by async syscall
        handle: String,
        /// Timeout in milliseconds
        #[serde(default)]
        timeout_ms: Option<u64>,
    },

    // =========================================================================
    // PROCESS MANAGEMENT - Multi-process operations
    // =========================================================================

    /// Fork a new process
    /// Creates a child process with copied state
    Fork {
        /// Program for the child to execute
        program_id: String,
        /// Arguments for the child
        #[serde(default)]
        args: serde_json::Value,
    },

    /// Join a child process (wait for completion)
    Join {
        /// Process ID to wait for
        pid: String,
    },

    /// Send a message to another process
    Send {
        /// Target process ID
        pid: String,
        /// Message to send
        message: serde_json::Value,
    },

    /// Receive a message from the message queue
    Recv {
        /// Timeout in milliseconds (None = block forever)
        #[serde(default)]
        timeout_ms: Option<u64>,
        /// Page to store received message
        store_to: String,
    },

    // =========================================================================
    // LLM-SPECIFIC OPERATIONS - AI compute primitives
    // =========================================================================

    /// Invoke the LLM for inference
    /// This is where the actual AI computation happens
    Infer {
        /// Prompt or instruction for the LLM
        prompt: String,
        /// Context pages to include
        #[serde(default)]
        context: Vec<String>,
        /// Page to store the response
        store_to: String,
        /// Model parameters
        #[serde(default)]
        params: InferParams,
    },

    /// Yield control back to the LLM for planning
    /// The LLM receives execution history and decides the next program segment
    /// This enables multi-turn agent execution with persistent state
    Plan {
        /// What the LLM should plan/decide
        goal: String,
        /// Pages containing relevant context for decision
        #[serde(default)]
        context: Vec<String>,
        /// Page to store the generated program segment
        store_to: String,
    },

    /// Reflect on execution and decide next steps
    /// Like PLAN but specifically for analyzing what happened
    Reflect {
        /// Question to answer about the execution
        question: String,
        /// Include execution trace in context
        #[serde(default)]
        include_trace: bool,
        /// Page to store the reflection result
        store_to: String,
    },

    /// Summarize one or more pages
    /// Compresses information for context window management
    Summarize {
        /// Pages to summarize
        pages: Vec<String>,
        /// Target size hint (in tokens)
        #[serde(default)]
        target_tokens: Option<usize>,
        /// Page to store the summary
        store_to: String,
    },

    /// Chunk a large page into smaller pages
    /// For processing large contexts incrementally
    Chunk {
        /// Source page to chunk
        source: String,
        /// Approximate chunk size (in tokens)
        chunk_size: usize,
        /// Prefix for chunk page IDs
        #[serde(default)]
        prefix: Option<String>,
    },

    /// Merge multiple pages into one
    Merge {
        /// Pages to merge
        pages: Vec<String>,
        /// Destination page
        store_to: String,
        /// Optional separator between pages
        #[serde(default)]
        separator: Option<String>,
    },

    // =========================================================================
    // DEBUGGING AND INTROSPECTION
    // =========================================================================

    /// No operation - useful for labels and debugging
    Nop,

    /// Log a message for debugging
    Log {
        /// Log level
        level: LogLevel,
        /// Message to log
        message: String,
    },

    /// Checkpoint the current state
    /// Allows rollback to this point
    Checkpoint {
        /// Checkpoint name
        name: String,
    },

    /// Rollback to a checkpoint
    Rollback {
        /// Checkpoint name to rollback to
        name: String,
    },

    /// Assert a condition (fail if false)
    Assert {
        /// Condition to check
        condition: String,
        /// Error message if assertion fails
        message: String,
    },

    // =========================================================================
    // REGISTER OPERATIONS - Working with execution state
    // =========================================================================

    /// Set a register value
    SetReg {
        /// Register name (pc, goal, focus, etc.)
        reg: Register,
        /// Value to set
        value: serde_json::Value,
    },

    /// Get a register value into a page
    GetReg {
        /// Register name
        reg: Register,
        /// Page to store the value
        store_to: String,
    },

    // =========================================================================
    // STACK OPERATIONS - LIFO data structure for working values
    // =========================================================================

    /// Push a value onto the stack
    Push {
        /// Value to push
        value: serde_json::Value,
    },

    /// Push contents of a page onto the stack
    PushPage {
        /// Page ID to push
        page_id: String,
    },

    /// Pop top value from stack
    Pop,

    /// Pop top value and store to a page
    PopTo {
        /// Page ID to store to
        store_to: String,
    },

    /// Peek at top value without removing (copies to page)
    Peek {
        /// Page ID to store the peeked value
        store_to: String,
    },

    /// Peek at value at depth N (0 = top)
    PeekAt {
        /// Depth to peek at (0-indexed from top)
        depth: usize,
        /// Page ID to store the peeked value
        store_to: String,
    },

    /// Duplicate the top value
    Dup,

    /// Duplicate the value at depth N (0 = top)
    DupN {
        /// Depth of value to duplicate (0-indexed from top)
        n: usize,
    },

    /// Swap top two values
    Swap,

    /// Swap top with value at depth N
    SwapN {
        /// Depth to swap with (1-indexed, SwapN{n:1} = swap top two)
        n: usize,
    },

    /// Rotate top N values (moves top to Nth position)
    Rot {
        /// Number of values to rotate
        n: usize,
    },

    /// Drop top N values
    Drop {
        /// Number of values to drop
        #[serde(default = "default_one")]
        n: usize,
    },

    /// Get stack depth
    Depth {
        /// Page to store the depth value
        store_to: String,
    },

    /// Clear the entire stack
    Clear,
}

/// Default value of 1 for drop
fn default_one() -> usize {
    1
}

/// Range specification for partial page operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Range {
    /// Start offset (0-indexed)
    pub start: usize,
    /// End offset (exclusive)
    pub end: usize,
}

/// Parameters for LLM inference
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InferParams {
    /// Temperature (0.0 = deterministic, 1.0+ = creative)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Model to use (if different from default)
    #[serde(default)]
    pub model: Option<String>,
}

/// Log levels for debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Named registers in the VM
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Register {
    /// Program counter - current step
    Pc,
    /// Current goal/objective
    Goal,
    /// Current focus (file, function, etc.)
    Focus,
    /// Last thought/reasoning
    Thought,
    /// Flags (waiting, blocked, etc.)
    Flags,
    /// Stack pointer
    Sp,
    /// Custom register
    Custom(String),
}

impl Opcode {
    /// Check if this opcode is a terminal instruction
    pub fn is_terminal(&self) -> bool {
        matches!(self, Opcode::Complete { .. } | Opcode::Fail { .. })
    }

    /// Check if this opcode modifies control flow
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Opcode::Call { .. }
                | Opcode::Return { .. }
                | Opcode::Branch { .. }
                | Opcode::Jump { .. }
                | Opcode::Loop { .. }
                | Opcode::Complete { .. }
                | Opcode::Fail { .. }
        )
    }

    /// Check if this opcode involves external I/O
    pub fn is_io(&self) -> bool {
        matches!(
            self,
            Opcode::Syscall { .. }
                | Opcode::Infer { .. }
                | Opcode::Send { .. }
                | Opcode::Recv { .. }
        )
    }

    /// Check if this opcode is a stack operation
    pub fn is_stack_op(&self) -> bool {
        matches!(
            self,
            Opcode::Push { .. }
                | Opcode::PushPage { .. }
                | Opcode::Pop
                | Opcode::PopTo { .. }
                | Opcode::Peek { .. }
                | Opcode::PeekAt { .. }
                | Opcode::Dup
                | Opcode::DupN { .. }
                | Opcode::Swap
                | Opcode::SwapN { .. }
                | Opcode::Rot { .. }
                | Opcode::Drop { .. }
                | Opcode::Depth { .. }
                | Opcode::Clear
        )
    }

    /// Get the page IDs this opcode reads from
    pub fn reads_pages(&self) -> Vec<&str> {
        match self {
            Opcode::Load { page_id, .. } => vec![page_id.as_str()],
            Opcode::Copy { src, .. } => vec![src.as_str()],
            Opcode::Infer { context, .. } => context.iter().map(|s| s.as_str()).collect(),
            Opcode::Summarize { pages, .. } => pages.iter().map(|s| s.as_str()).collect(),
            Opcode::Chunk { source, .. } => vec![source.as_str()],
            Opcode::Merge { pages, .. } => pages.iter().map(|s| s.as_str()).collect(),
            Opcode::PushPage { page_id } => vec![page_id.as_str()],
            _ => vec![],
        }
    }

    /// Get the page IDs this opcode writes to
    pub fn writes_pages(&self) -> Vec<&str> {
        match self {
            Opcode::Store { page_id, .. } => vec![page_id.as_str()],
            Opcode::Alloc { label, .. } => label.as_ref().map(|s| vec![s.as_str()]).unwrap_or_default(),
            Opcode::Copy { dst, .. } => vec![dst.as_str()],
            Opcode::Syscall { store_to, .. } => store_to.as_ref().map(|s| vec![s.as_str()]).unwrap_or_default(),
            Opcode::Infer { store_to, .. } => vec![store_to.as_str()],
            Opcode::Summarize { store_to, .. } => vec![store_to.as_str()],
            Opcode::Merge { store_to, .. } => vec![store_to.as_str()],
            Opcode::Recv { store_to, .. } => vec![store_to.as_str()],
            Opcode::GetReg { store_to, .. } => vec![store_to.as_str()],
            Opcode::PopTo { store_to } => vec![store_to.as_str()],
            Opcode::Peek { store_to } => vec![store_to.as_str()],
            Opcode::PeekAt { store_to, .. } => vec![store_to.as_str()],
            Opcode::Depth { store_to } => vec![store_to.as_str()],
            _ => vec![],
        }
    }
}

/// A program is a sequence of opcodes with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    /// Unique program identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this program does
    #[serde(default)]
    pub description: Option<String>,
    /// The opcodes that make up this program
    pub code: Vec<Opcode>,
    /// Entry point label (defaults to first opcode)
    #[serde(default)]
    pub entry: Option<String>,
}

impl Program {
    /// Create a new program
    pub fn new(id: impl Into<String>, name: impl Into<String>, code: Vec<Opcode>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            code,
            entry: None,
        }
    }

    /// Pretty print the program to stdout
    pub fn pretty_print(&self) {
        println!("--- {} ---", self.name);
        println!("ID: {}", self.id);
        if let Some(desc) = &self.description {
            println!("{}", desc);
        }
        if let Some(entry) = &self.entry {
            println!("Entry: {}", entry);
        }
        println!("Opcodes: {}", self.code.len());
        println!();

        for (i, op) in self.code.iter().enumerate() {
            let (name, details) = op.format_parts();

            // Labels at column 0, instructions indented
            let indent = if matches!(op, Opcode::Label { .. }) { "" } else { "    " };

            if details.is_empty() {
                println!("{:3} | {}{}", i, indent, name);
            } else {
                println!("{:3} | {}{} {}", i, indent, name, details);
            }

            // Show full prompt for INFER opcodes
            if let Opcode::Infer { prompt, context, params, .. } = op {
                println!("      |     prompt: \"{}\"", prompt);
                if !context.is_empty() {
                    println!("      |     context: [{}]", context.join(", "));
                }
                if params.temperature.is_some() || params.max_tokens.is_some() {
                    let temp = params.temperature.map(|t| format!("temp={}", t)).unwrap_or_default();
                    let max = params.max_tokens.map(|m| format!("max_tokens={}", m)).unwrap_or_default();
                    let p = [temp, max].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", ");
                    println!("      |     params: {}", p);
                }
            }
        }
        println!();
    }
}

impl Opcode {
    /// Format opcode into (name, details) for pretty printing
    fn format_parts(&self) -> (&'static str, String) {
        match self {
            Opcode::Label { name } => ("LABEL", format!(":{}", name)),
            Opcode::Log { level, message } => ("LOG", format!("[{:?}] \"{}\"", level, truncate(message, 30))),
            Opcode::Syscall { call, args, store_to } => {
                let store = store_to.as_ref().map(|s| format!(" → {}", s)).unwrap_or_default();
                ("SYSCALL", format!("{}({}){}", call, format_args_brief(args), store))
            }
            Opcode::Infer { prompt, context, store_to, .. } => {
                let ctx = if context.is_empty() { String::new() } else { format!(" [{}]", context.join(", ")) };
                ("INFER", format!("\"{}\"{}  → {}", truncate(prompt, 25), ctx, store_to))
            }
            Opcode::Branch { condition, if_true, if_false } => {
                ("BRANCH", format!("{} ? {} : {}", truncate(condition, 15), if_true, if_false))
            }
            Opcode::Jump { target } => ("JUMP", format!("→ {}", target)),
            Opcode::Push { value } => ("PUSH", format_value_brief(value)),
            Opcode::PushPage { page_id } => ("PUSH_PAGE", page_id.clone()),
            Opcode::Pop => ("POP", String::new()),
            Opcode::PopTo { store_to } => ("POP_TO", format!("→ {}", store_to)),
            Opcode::Complete { result } => ("COMPLETE", format_value_brief(result)),
            Opcode::Fail { error } => ("FAIL", format!("\"{}\"", truncate(error, 40))),
            Opcode::Call { program_id, args } => ("CALL", format!("{}({})", program_id, format_args_brief(args))),
            Opcode::Return { value } => ("RETURN", format_value_brief(value)),
            Opcode::Yield => ("YIELD", String::new()),
            Opcode::Load { page_id, .. } => ("LOAD", page_id.clone()),
            Opcode::Store { page_id, .. } => ("STORE", page_id.clone()),
            Opcode::Alloc { label, .. } => ("ALLOC", label.clone().unwrap_or_default()),
            Opcode::Free { page_id } => ("FREE", page_id.clone()),
            Opcode::Copy { src, dst, .. } => ("COPY", format!("{} → {}", src, dst)),
            Opcode::Summarize { pages, store_to, .. } => ("SUMMARIZE", format!("[{}] → {}", pages.join(", "), store_to)),
            Opcode::Chunk { source, chunk_size, .. } => ("CHUNK", format!("{} / {}", source, chunk_size)),
            Opcode::Merge { pages, store_to, .. } => ("MERGE", format!("[{}] → {}", pages.join(", "), store_to)),
            Opcode::Fork { program_id, .. } => ("FORK", program_id.clone()),
            Opcode::Join { pid } => ("JOIN", pid.clone()),
            Opcode::Send { pid, .. } => ("SEND", format!("→ {}", pid)),
            Opcode::Recv { store_to, .. } => ("RECV", format!("→ {}", store_to)),
            Opcode::Wait { handle, .. } => ("WAIT", handle.clone()),
            Opcode::Nop => ("NOP", String::new()),
            Opcode::Checkpoint { name } => ("CHECKPOINT", name.clone()),
            Opcode::Rollback { name } => ("ROLLBACK", name.clone()),
            Opcode::Assert { condition, .. } => ("ASSERT", truncate(condition, 40)),
            Opcode::SetReg { reg, .. } => ("SET_REG", format!("{:?}", reg)),
            Opcode::GetReg { reg, store_to } => ("GET_REG", format!("{:?} → {}", reg, store_to)),
            Opcode::Dup => ("DUP", String::new()),
            Opcode::DupN { n } => ("DUP_N", format!("{}", n)),
            Opcode::Swap => ("SWAP", String::new()),
            Opcode::SwapN { n } => ("SWAP_N", format!("{}", n)),
            Opcode::Rot { n } => ("ROT", format!("{}", n)),
            Opcode::Drop { n } => ("DROP", format!("{}", n)),
            Opcode::Peek { store_to } => ("PEEK", format!("→ {}", store_to)),
            Opcode::PeekAt { depth, store_to } => ("PEEK_AT", format!("[{}] → {}", depth, store_to)),
            Opcode::Loop { var, over, .. } => ("LOOP", format!("{} in {}", var, over)),
            Opcode::Depth { store_to } => ("DEPTH", format!("→ {}", store_to)),
            Opcode::Clear => ("CLEAR", String::new()),
            Opcode::Plan { goal, context, store_to } => {
                let ctx = if context.is_empty() { String::new() } else { format!(" [{}]", context.join(", ")) };
                ("PLAN", format!("\"{}\"{}  → {}", truncate(goal, 25), ctx, store_to))
            }
            Opcode::Reflect { question, include_trace, store_to } => {
                let trace = if *include_trace { " +trace" } else { "" };
                ("REFLECT", format!("\"{}\"{}  → {}", truncate(question, 25), trace, store_to))
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

fn format_value_brief(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => format!("\"{}\"", truncate(s, 20)),
        serde_json::Value::Array(a) => format!("[{} items]", a.len()),
        serde_json::Value::Object(o) => {
            let keys: Vec<_> = o.keys().take(3).cloned().collect();
            if keys.len() < o.len() {
                format!("{{{}, …}}", keys.join(", "))
            } else {
                format!("{{{}}}", keys.join(", "))
            }
        }
    }
}

fn format_args_brief(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(o) if o.len() <= 2 => {
            o.iter()
                .map(|(k, v)| format!("{}={}", k, format_value_brief(v)))
                .collect::<Vec<_>>()
                .join(", ")
        }
        serde_json::Value::Object(o) => format!("{} args", o.len()),
        other => format_value_brief(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_serialization() {
        let op = Opcode::Syscall {
            call: "read_file".to_string(),
            args: serde_json::json!({"path": "src/main.rs"}),
            store_to: Some("file_content".to_string()),
        };

        let json = serde_json::to_string_pretty(&op).unwrap();
        println!("{}", json);

        let parsed: Opcode = serde_json::from_str(&json).unwrap();
        assert_eq!(op, parsed);
    }

    #[test]
    fn test_program_serialization() {
        let program = Program::new(
            "analyze_file",
            "Analyze File",
            vec![
                Opcode::Syscall {
                    call: "read_file".to_string(),
                    args: serde_json::json!({"path": "target.rs"}),
                    store_to: Some("content".to_string()),
                },
                Opcode::Infer {
                    prompt: "Analyze this code for bugs".to_string(),
                    context: vec!["content".to_string()],
                    store_to: "analysis".to_string(),
                    params: InferParams::default(),
                },
                Opcode::Complete {
                    result: serde_json::json!({"page": "analysis"}),
                },
            ],
        );

        let json = serde_json::to_string_pretty(&program).unwrap();
        println!("{}", json);

        let parsed: Program = serde_json::from_str(&json).unwrap();
        assert_eq!(program.id, parsed.id);
        assert_eq!(program.code.len(), parsed.code.len());
    }

    #[test]
    fn test_is_terminal() {
        assert!(Opcode::Complete { result: serde_json::json!({}) }.is_terminal());
        assert!(Opcode::Fail { error: "oops".to_string() }.is_terminal());
        assert!(!Opcode::Nop.is_terminal());
    }

    #[test]
    fn test_reads_writes_pages() {
        let op = Opcode::Infer {
            prompt: "test".to_string(),
            context: vec!["page1".to_string(), "page2".to_string()],
            store_to: "output".to_string(),
            params: InferParams::default(),
        };

        assert_eq!(op.reads_pages(), vec!["page1", "page2"]);
        assert_eq!(op.writes_pages(), vec!["output"]);
    }
}
