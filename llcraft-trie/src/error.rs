//! # Error types for MPT

use thiserror::Error;

/// MPT error types
#[derive(Debug, Error, Clone)]
pub enum TrieError {
    #[error("Key not found")]
    KeyNotFound,

    #[error("Invalid node encoding")]
    InvalidEncoding,

    #[error("Invalid proof")]
    InvalidProof,

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("RLP decode error: {0}")]
    RlpDecode(String),

    #[error("Hash mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        expected: String,
        actual: String,
    },
}

/// Result type for trie operations
pub type Result<T> = std::result::Result<T, TrieError>;
