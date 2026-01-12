//! The main Error type for llcraft

use crate::{ErrorKind, ErrorStatus};
use std::fmt;

/// The unified error type for all llcraft operations.
///
/// This error type provides:
/// - `kind`: What type of error occurred
/// - `message`: Human-readable description
/// - `status`: Whether the error is retryable
/// - `operation`: What operation caused the error
/// - `context`: Key-value pairs for debugging
/// - `source`: The underlying error (if any)
///
/// # Example
///
/// ```rust
/// use llcraft_error::{Error, ErrorKind, ErrorStatus};
///
/// let err = Error::new(ErrorKind::InferenceFailed, "model returned empty response")
///     .with_operation("interpreter::infer")
///     .with_status(ErrorStatus::Temporary)
///     .with_context("model", "gpt-4")
///     .with_context("prompt_tokens", "1500");
///
/// assert_eq!(err.kind(), ErrorKind::InferenceFailed);
/// assert!(err.status().is_retryable());
/// ```
pub struct Error {
    kind: ErrorKind,
    message: String,
    status: ErrorStatus,
    operation: &'static str,
    context: Vec<(&'static str, String)>,
    source: Option<anyhow::Error>,
}

impl Error {
    /// Create a new error with the given kind and message
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        let status = if kind.is_retryable() {
            ErrorStatus::Temporary
        } else {
            ErrorStatus::Permanent
        };

        Self {
            kind,
            message: message.into(),
            status,
            operation: "",
            context: Vec::new(),
            source: None,
        }
    }

    // =========================================================================
    // Getters
    // =========================================================================

    /// Get the error kind
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Get the error message
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Get the error status
    pub fn status(&self) -> ErrorStatus {
        self.status
    }

    /// Get the operation that caused this error
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Get the context key-value pairs
    pub fn context(&self) -> &[(&'static str, String)] {
        &self.context
    }

    /// Get the source error (if any)
    pub fn source_ref(&self) -> Option<&anyhow::Error> {
        self.source.as_ref()
    }

    // =========================================================================
    // Builders (chainable)
    // =========================================================================

    /// Set the error status
    pub fn with_status(mut self, status: ErrorStatus) -> Self {
        self.status = status;
        self
    }

    /// Mark as temporary (retryable)
    pub fn temporary(mut self) -> Self {
        self.status = ErrorStatus::Temporary;
        self
    }

    /// Mark as permanent (not retryable)
    pub fn permanent(mut self) -> Self {
        self.status = ErrorStatus::Permanent;
        self
    }

    /// Set the operation that caused this error.
    ///
    /// If an operation was already set, the previous one is moved to context
    /// as "called" to preserve the call chain.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        if !self.operation.is_empty() {
            self.context.push(("called", self.operation.to_string()));
        }
        self.operation = operation;
        self
    }

    /// Add context to the error
    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    /// Set the source error.
    ///
    /// # Panics (debug only)
    /// Panics in debug mode if source was already set.
    pub fn set_source(mut self, source: impl Into<anyhow::Error>) -> Self {
        debug_assert!(self.source.is_none(), "source error already set");
        self.source = Some(source.into());
        self
    }

    // =========================================================================
    // Status mutations
    // =========================================================================

    /// Mark as persistent after failed retries
    pub fn persist(mut self) -> Self {
        self.status = self.status.persist();
        self
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        self.status.is_retryable()
    }
}

// =============================================================================
// Display - compact, single-line format for logs
// =============================================================================

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;

        if !self.context.is_empty() {
            write!(f, ", context {{ ")?;
            for (i, (key, value)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", key, value)?;
            }
            write!(f, " }}")?;
        }

        if !self.message.is_empty() {
            write!(f, " => {}", self.message)?;
        }

        Ok(())
    }
}

// =============================================================================
// Debug - verbose, multi-line format for debugging
// =============================================================================

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;

        if !self.message.is_empty() {
            writeln!(f)?;
            writeln!(f, "    Message: {}", self.message)?;
        }

        if !self.context.is_empty() {
            writeln!(f)?;
            writeln!(f, "    Context:")?;
            for (key, value) in &self.context {
                writeln!(f, "        {}: {}", key, value)?;
            }
        }

        if let Some(source) = &self.source {
            writeln!(f)?;
            writeln!(f, "    Source: {:?}", source)?;
        }

        Ok(())
    }
}

// =============================================================================
// std::error::Error implementation
// =============================================================================

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

// =============================================================================
// Convenient From implementations (be careful not to leak raw errors!)
// =============================================================================

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        let kind = match err.kind() {
            std::io::ErrorKind::NotFound => ErrorKind::FileNotFound,
            std::io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
            _ => ErrorKind::IoFailed,
        };
        Error::new(kind, err.to_string())
            .with_operation("io")
            .set_source(err)
    }
}

// =============================================================================
// Convenience constructors
// =============================================================================

impl Error {
    /// Create an Unexpected error
    pub fn unexpected(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unexpected, message)
    }

    /// Create an Unsupported error
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported, message)
    }

    /// Create a PageNotFound error
    pub fn page_not_found(page_id: impl Into<String>) -> Self {
        let page_id = page_id.into();
        Self::new(ErrorKind::PageNotFound, format!("page '{}' not found", page_id))
            .with_context("page_id", page_id)
    }

    /// Create a StackOverflow error
    pub fn stack_overflow() -> Self {
        Self::new(ErrorKind::StackOverflow, "stack depth exceeded maximum")
    }

    /// Create a StackUnderflow error
    pub fn stack_underflow() -> Self {
        Self::new(ErrorKind::StackUnderflow, "cannot pop from empty stack")
    }

    /// Create an InferenceFailed error
    pub fn inference_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InferenceFailed, message)
    }

    /// Create a SyscallFailed error
    pub fn syscall_failed(name: impl Into<String>, reason: impl Into<String>) -> Self {
        let name = name.into();
        Self::new(ErrorKind::SyscallFailed, reason)
            .with_context("syscall", name)
    }

    /// Create an InvalidLabel error
    pub fn invalid_label(label: impl Into<String>) -> Self {
        let label = label.into();
        Self::new(ErrorKind::InvalidLabel, format!("invalid label '{}'", label))
            .with_context("label", label)
    }

    /// Create a ProgramNotFound error
    pub fn program_not_found(program_id: impl Into<String>) -> Self {
        let program_id = program_id.into();
        Self::new(ErrorKind::ProgramNotFound, format!("program '{}' not found", program_id))
            .with_context("program_id", program_id)
    }

    /// Create a ParseFailed error
    pub fn parse_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ParseFailed, message)
    }

    /// Create an AssertionFailed error
    pub fn assertion_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::AssertionFailed, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::new(ErrorKind::PageNotFound, "page 'context' not found");
        assert_eq!(err.kind(), ErrorKind::PageNotFound);
        assert_eq!(err.message(), "page 'context' not found");
        assert_eq!(err.status(), ErrorStatus::Permanent);
    }

    #[test]
    fn test_error_with_context() {
        let err = Error::new(ErrorKind::InferenceFailed, "timeout")
            .with_operation("interpreter::infer")
            .with_context("model", "gpt-4")
            .with_context("tokens", "1500");

        assert_eq!(err.operation(), "interpreter::infer");
        assert_eq!(err.context().len(), 2);
        assert_eq!(err.context()[0], ("model", "gpt-4".to_string()));
    }

    #[test]
    fn test_operation_chaining() {
        let err = Error::new(ErrorKind::IoFailed, "write failed")
            .with_operation("storage::save")
            .with_operation("interpreter::checkpoint");

        assert_eq!(err.operation(), "interpreter::checkpoint");
        assert_eq!(err.context().len(), 1);
        assert_eq!(err.context()[0], ("called", "storage::save".to_string()));
    }

    #[test]
    fn test_temporary_status() {
        let err = Error::new(ErrorKind::InferenceFailed, "rate limited");
        assert!(err.is_retryable()); // InferenceFailed defaults to temporary

        let err = Error::new(ErrorKind::PageNotFound, "not found");
        assert!(!err.is_retryable()); // PageNotFound defaults to permanent
    }

    #[test]
    fn test_persist() {
        let err = Error::new(ErrorKind::NetworkFailed, "connection refused")
            .temporary();
        assert!(err.is_retryable());

        let err = err.persist();
        assert!(!err.is_retryable());
        assert_eq!(err.status(), ErrorStatus::Persistent);
    }

    #[test]
    fn test_display() {
        let err = Error::new(ErrorKind::InferenceFailed, "model unavailable")
            .with_operation("provider::infer")
            .with_context("model", "claude-3")
            .with_context("attempt", "3");

        let display = format!("{}", err);
        assert!(display.contains("InferenceFailed"));
        assert!(display.contains("temporary"));
        assert!(display.contains("provider::infer"));
        assert!(display.contains("model: claude-3"));
    }

    #[test]
    fn test_convenience_constructors() {
        let err = Error::page_not_found("context");
        assert_eq!(err.kind(), ErrorKind::PageNotFound);
        assert!(err.message().contains("context"));

        let err = Error::stack_overflow();
        assert_eq!(err.kind(), ErrorKind::StackOverflow);

        let err = Error::syscall_failed("read_file", "permission denied");
        assert_eq!(err.kind(), ErrorKind::SyscallFailed);
    }

    #[test]
    fn test_set_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = Error::new(ErrorKind::FileNotFound, "config.json not found")
            .set_source(io_err);

        assert!(err.source_ref().is_some());
    }
}
