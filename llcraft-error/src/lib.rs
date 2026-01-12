//! # llcraft-error
//!
//! Unified error handling for llcraft - following OpenDAL's error handling practices.
//!
//! ## Design Philosophy
//!
//! - **ErrorKind**: Know what error occurred (e.g., PageNotFound, InferenceFailed)
//! - **ErrorStatus**: Decide how to handle it (Permanent, Temporary, Persistent)
//! - **Error Context**: Assist in locating the cause with rich context
//! - **Error Source**: Wrap underlying errors without leaking raw types
//!
//! ## Usage
//!
//! ```rust
//! use llcraft_error::{Error, ErrorKind};
//!
//! fn example() -> Result<(), Error> {
//!     Err(Error::new(ErrorKind::PageNotFound, "page 'context' not loaded")
//!         .with_operation("interpreter::execute")
//!         .with_context("page_id", "context")
//!         .with_context("program", "analyze_code"))
//! }
//! ```
//!
//! ## Principles
//!
//! - All functions return `Result<T, llcraft_error::Error>`
//! - External errors are wrapped with `set_source(err)`
//! - Same error handled once, subsequent ops only append context
//! - Don't abuse `From<OtherError>` to prevent raw error leakage

mod error;
mod kind;
mod status;

pub use error::Error;
pub use kind::ErrorKind;
pub use status::ErrorStatus;

/// Result type alias using llcraft Error
pub type Result<T> = std::result::Result<T, Error>;
