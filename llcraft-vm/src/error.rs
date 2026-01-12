//! LLM-VM Error types
//!
//! Re-exports llcraft-error and provides VM-specific conveniences.

// Re-export the core error types
pub use llcraft_error::{Error, ErrorKind, ErrorStatus, Result};

/// Legacy VmError alias - use Error instead in new code
#[deprecated(since = "0.2.0", note = "Use llcraft_error::Error instead")]
pub type VmError = Error;

// =============================================================================
// VM-specific error constructors
// =============================================================================

/// Create a PageNotFound error
pub fn page_not_found(page_id: impl Into<String>) -> Error {
    Error::page_not_found(page_id)
}

/// Create a StackOverflow error
pub fn stack_overflow() -> Error {
    Error::stack_overflow()
}

/// Create a StackUnderflow error
pub fn stack_underflow() -> Error {
    Error::stack_underflow()
}

/// Create a PageOverflow error
pub fn page_overflow() -> Error {
    Error::new(ErrorKind::PageOverflow, "exceeds context window limit")
}

/// Create an InvalidRange error
pub fn invalid_range(start: usize, end: usize) -> Error {
    Error::new(ErrorKind::InvalidRange, format!("invalid range: {}..{}", start, end))
        .with_context("start", start.to_string())
        .with_context("end", end.to_string())
}

/// Create a ProgramNotFound error
pub fn program_not_found(program_id: impl Into<String>) -> Error {
    Error::program_not_found(program_id)
}

/// Create an InvalidLabel error
pub fn invalid_label(label: impl Into<String>) -> Error {
    Error::invalid_label(label)
}

/// Create a CallDepthExceeded error
pub fn call_depth_exceeded(max: usize) -> Error {
    Error::new(ErrorKind::CallDepthExceeded, format!("call depth exceeded max {}", max))
        .with_context("max_depth", max.to_string())
}

/// Create a NoReturnAddress error
pub fn no_return_address() -> Error {
    Error::new(ErrorKind::NoReturnAddress, "no return address on call stack")
}

/// Create a SyscallFailed error
pub fn syscall_failed(name: impl Into<String>, reason: impl Into<String>) -> Error {
    Error::syscall_failed(name, reason)
}

/// Create a SyscallTimeout error
pub fn syscall_timeout(name: impl Into<String>) -> Error {
    let name = name.into();
    Error::new(ErrorKind::SyscallTimeout, format!("syscall '{}' timed out", name))
        .with_context("syscall", name)
        .temporary()
}

/// Create an UnknownSyscall error
pub fn unknown_syscall(name: impl Into<String>) -> Error {
    let name = name.into();
    Error::new(ErrorKind::SyscallUnknown, format!("unknown syscall: {}", name))
        .with_context("syscall", name)
}

/// Create a ProcessNotFound error
pub fn process_not_found(pid: impl Into<String>) -> Error {
    let pid = pid.into();
    Error::new(ErrorKind::ProcessNotFound, format!("process '{}' not found", pid))
        .with_context("pid", pid)
}

/// Create a ChannelClosed error
pub fn channel_closed(name: impl Into<String>) -> Error {
    let name = name.into();
    Error::new(ErrorKind::ChannelClosed, format!("channel '{}' closed", name))
        .with_context("channel", name)
}

/// Create a ForkFailed error
pub fn fork_failed(reason: impl Into<String>) -> Error {
    Error::new(ErrorKind::ForkFailed, reason)
}

/// Create an InferenceFailed error
pub fn inference_failed(reason: impl Into<String>) -> Error {
    Error::inference_failed(reason)
}

/// Create a ContextTooLarge error
pub fn context_too_large(size: usize, max: usize) -> Error {
    Error::new(ErrorKind::ContextTooLarge, format!("{} tokens exceeds max {}", size, max))
        .with_context("size", size.to_string())
        .with_context("max", max.to_string())
}

/// Create a ParseError error
pub fn parse_error(message: impl Into<String>) -> Error {
    Error::parse_failed(message)
}

/// Create an AssertionFailed error
pub fn assertion_failed(message: impl Into<String>) -> Error {
    Error::assertion_failed(message)
}

/// Create an InvalidOpcode error
pub fn invalid_opcode(position: usize) -> Error {
    Error::new(ErrorKind::InvalidOpcode, format!("invalid opcode at position {}", position))
        .with_context("position", position.to_string())
}

/// Create an IoError error
pub fn io_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::IoFailed, message)
}

/// Create a SerializationError error
pub fn serialization_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::SerializationFailed, message)
}

/// Create a StorageNotFound error
pub fn storage_not_found(key: impl Into<String>) -> Error {
    let key = key.into();
    Error::new(ErrorKind::StorageNotFound, format!("storage key '{}' not found", key))
        .with_context("key", key)
}

/// Create a StorageFailed error
pub fn storage_failed(reason: impl Into<String>) -> Error {
    Error::new(ErrorKind::StorageFailed, reason)
}

/// Create an InvalidArgument error
pub fn invalid_argument(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidArgument, message)
}

/// Create a LabelNotFound error
pub fn label_not_found(label: impl Into<String>) -> Error {
    let label = label.into();
    Error::new(ErrorKind::InvalidLabel, format!("label '{}' not found", label))
        .with_context("label", label)
}

/// Create a NotImplemented error
pub fn not_implemented(feature: impl Into<String>) -> Error {
    let feature = feature.into();
    Error::new(ErrorKind::NotImplemented, format!("'{}' not yet implemented", feature))
        .with_context("feature", feature)
}
