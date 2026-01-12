//! # LLcraft VM
//!
//! A virtual machine where LLMs are the compute unit.
//!
//! ## Core Concepts
//! - **Pages**: Named memory regions holding JSON data (context window management)
//! - **Stack**: Working values for intermediate computations
//! - **Opcodes**: Instruction set for orchestrating LLM inference
//! - **Syscalls**: Controlled access to external tools
//! - **Provider**: Trait-based LLM communication (OpenAI, Anthropic, local)

pub mod opcode;
pub mod error;
pub mod stack;
pub mod memory;
pub mod storage;
pub mod schema;
pub mod provider;
pub mod interpreter;

pub use opcode::{Opcode, Program, Range, InferParams, LogLevel, Register};
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use stack::Stack;
pub use memory::{Memory, MemoryPage};
pub use storage::{Storage, StorageBackend, MemoryStorage, FileStorage};
pub use schema::{VmSchema, TaskRequest, ExecutionStep};
pub use provider::{
    LlmProvider, ProviderConfig, ProviderType, ProviderError,
    ChatMessage, Role, CompletionRequest, CompletionResponse,
    ToolDefinition, ToolCall, ToolChoice,
    StreamChunk, StreamReceiver, FinishReason, Usage, UsageTracker,
    OpenAIProvider, AnthropicProvider, BridgeProvider,
};
pub use interpreter::{
    Interpreter, ExecutionResult, ExecutionState,
    LlmRequest, LlmRequestType,
    SyscallHandler, DefaultSyscallHandler,
};

