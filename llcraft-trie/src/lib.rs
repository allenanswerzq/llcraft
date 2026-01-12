//! # Merkle Patricia Trie
//!
//! Implementation of Ethereum's Modified Merkle Patricia Trie.
//!
//! This data structure is used for:
//! - State storage (accounts -> account data)
//! - Transaction tries
//! - Receipt tries
//!
//! Key features:
//! - Efficient proofs of inclusion/exclusion
//! - Cryptographic commitment to entire state
//! - Efficient updates

pub mod nibbles;
pub mod node;
pub mod trie;
pub mod proof;
pub mod error;

pub use nibbles::Nibbles;
pub use node::Node;
pub use trie::{PatriciaTrie, TrieDB};
pub use proof::Proof;
pub use error::TrieError;
