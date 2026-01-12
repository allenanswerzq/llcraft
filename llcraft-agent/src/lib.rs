//! # LLcraft Agent
//!
//! The agent orchestrates the LLM <-> VM loop:
//! 1. User provides a task description
//! 2. LLM generates a program to solve it
//! 3. Interpreter runs the program
//! 4. If program needs more LLM input, we call the LLM and continue
//! 5. INJECT allows JIT code generation - LLM generates new opcodes at runtime
//! 6. Multi-step until COMPLETE or FAIL
//!
//! The LLM is the brain, the VM is the body.

mod agent;

pub use agent::{Agent, AgentResult, AgentConfig};
