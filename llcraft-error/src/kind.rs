//! Error kinds for llcraft operations

use std::fmt;

/// The kind of error that occurred.
///
/// This enum categorizes errors to help users write clear error handling logic.
/// Users can match on ErrorKind to decide how to handle specific error cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    // =========================================================================
    // General errors
    // =========================================================================
    /// An unexpected error occurred - catch-all for unhandled cases
    Unexpected,

    /// The requested feature or operation is not supported
    Unsupported,

    /// Invalid configuration or parameters
    ConfigInvalid,

    // =========================================================================
    // Memory/Page errors
    // =========================================================================
    /// The requested page was not found in memory
    PageNotFound,

    /// Page overflow - exceeds context window or memory limit
    PageOverflow,

    /// Invalid memory range specified
    InvalidRange,

    // =========================================================================
    // Stack errors
    // =========================================================================
    /// Stack overflow - too many values pushed
    StackOverflow,

    /// Stack underflow - tried to pop from empty stack
    StackUnderflow,

    // =========================================================================
    // Storage errors
    // =========================================================================
    /// Storage key not found
    StorageNotFound,

    /// Storage operation failed
    StorageFailed,

    /// Serialization/deserialization failed
    SerializationFailed,

    // =========================================================================
    // Program/Control flow errors
    // =========================================================================
    /// Program not found
    ProgramNotFound,

    /// Invalid label for jump/branch
    InvalidLabel,

    /// Call depth exceeded maximum
    CallDepthExceeded,

    /// No return address available
    NoReturnAddress,

    /// Invalid opcode or instruction
    InvalidOpcode,

    // =========================================================================
    // Syscall errors
    // =========================================================================
    /// Syscall failed to execute
    SyscallFailed,

    /// Syscall timed out
    SyscallTimeout,

    /// Unknown syscall name
    SyscallUnknown,

    // =========================================================================
    // Process errors
    // =========================================================================
    /// Process not found
    ProcessNotFound,

    /// Channel closed unexpectedly
    ChannelClosed,

    /// Fork operation failed
    ForkFailed,

    // =========================================================================
    // Inference/LLM errors
    // =========================================================================
    /// LLM inference failed
    InferenceFailed,

    /// Context too large for model
    ContextTooLarge,

    /// Provider not available
    ProviderUnavailable,

    /// Rate limit exceeded
    RateLimited,

    // =========================================================================
    // IO errors
    // =========================================================================
    /// File not found
    FileNotFound,

    /// Permission denied
    PermissionDenied,

    /// IO operation failed
    IoFailed,

    /// Network error
    NetworkFailed,

    // =========================================================================
    // Parse errors
    // =========================================================================
    /// Failed to parse input
    ParseFailed,

    /// Assertion failed
    AssertionFailed,

    /// Invalid argument passed to function
    InvalidArgument,

    /// Feature or operation not yet implemented
    NotImplemented,
}

impl ErrorKind {
    /// Returns the error kind as a static string
    pub fn as_str(&self) -> &'static str {
        match self {
            // General
            ErrorKind::Unexpected => "Unexpected",
            ErrorKind::Unsupported => "Unsupported",
            ErrorKind::ConfigInvalid => "ConfigInvalid",

            // Memory/Page
            ErrorKind::PageNotFound => "PageNotFound",
            ErrorKind::PageOverflow => "PageOverflow",
            ErrorKind::InvalidRange => "InvalidRange",

            // Stack
            ErrorKind::StackOverflow => "StackOverflow",
            ErrorKind::StackUnderflow => "StackUnderflow",

            // Storage
            ErrorKind::StorageNotFound => "StorageNotFound",
            ErrorKind::StorageFailed => "StorageFailed",
            ErrorKind::SerializationFailed => "SerializationFailed",

            // Program/Control
            ErrorKind::ProgramNotFound => "ProgramNotFound",
            ErrorKind::InvalidLabel => "InvalidLabel",
            ErrorKind::CallDepthExceeded => "CallDepthExceeded",
            ErrorKind::NoReturnAddress => "NoReturnAddress",
            ErrorKind::InvalidOpcode => "InvalidOpcode",

            // Syscall
            ErrorKind::SyscallFailed => "SyscallFailed",
            ErrorKind::SyscallTimeout => "SyscallTimeout",
            ErrorKind::SyscallUnknown => "SyscallUnknown",

            // Process
            ErrorKind::ProcessNotFound => "ProcessNotFound",
            ErrorKind::ChannelClosed => "ChannelClosed",
            ErrorKind::ForkFailed => "ForkFailed",

            // Inference
            ErrorKind::InferenceFailed => "InferenceFailed",
            ErrorKind::ContextTooLarge => "ContextTooLarge",
            ErrorKind::ProviderUnavailable => "ProviderUnavailable",
            ErrorKind::RateLimited => "RateLimited",

            // IO
            ErrorKind::FileNotFound => "FileNotFound",
            ErrorKind::PermissionDenied => "PermissionDenied",
            ErrorKind::IoFailed => "IoFailed",
            ErrorKind::NetworkFailed => "NetworkFailed",

            // Parse
            ErrorKind::ParseFailed => "ParseFailed",
            ErrorKind::AssertionFailed => "AssertionFailed",
            ErrorKind::InvalidArgument => "InvalidArgument",
            ErrorKind::NotImplemented => "NotImplemented",
        }
    }

    /// Check if this error kind is retryable by default
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ErrorKind::InferenceFailed
                | ErrorKind::NetworkFailed
                | ErrorKind::RateLimited
                | ErrorKind::SyscallTimeout
                | ErrorKind::ProviderUnavailable
        )
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kind_display() {
        assert_eq!(ErrorKind::PageNotFound.to_string(), "PageNotFound");
        assert_eq!(ErrorKind::InferenceFailed.to_string(), "InferenceFailed");
    }

    #[test]
    fn test_is_retryable() {
        assert!(ErrorKind::NetworkFailed.is_retryable());
        assert!(ErrorKind::RateLimited.is_retryable());
        assert!(!ErrorKind::PageNotFound.is_retryable());
        assert!(!ErrorKind::StackUnderflow.is_retryable());
    }
}
