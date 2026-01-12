//! # MPT Node Types
//!
//! The MPT has three node types:
//! 1. Leaf - stores a value at a key
//! 2. Extension - shares a common prefix path
//! 3. Branch - 16-way branch point + optional value

use eth_primitives::{H256, keccak256};
use crate::nibbles::Nibbles;

/// Node hash - either inline data or a hash reference
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeRef {
    /// Empty node
    Empty,
    /// Inline data (< 32 bytes when RLP encoded)
    Inline(Vec<u8>),
    /// Hash reference to node in database
    Hash(H256),
}

impl NodeRef {
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        matches!(self, NodeRef::Empty)
    }

    /// Get hash if this is a hash reference
    pub fn as_hash(&self) -> Option<H256> {
        match self {
            NodeRef::Hash(h) => Some(*h),
            _ => None,
        }
    }
}

/// MPT node types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// Empty node (null)
    Empty,

    /// Leaf node: [encoded_path, value]
    /// The path is the remaining key nibbles
    Leaf {
        key: Nibbles,
        value: Vec<u8>,
    },

    /// Extension node: [encoded_path, child]
    /// Shares a common prefix to save space
    Extension {
        key: Nibbles,
        child: NodeRef,
    },

    /// Branch node: [child0, child1, ..., child15, value]
    /// 16 children (one per nibble) + optional value
    Branch {
        children: Box<[NodeRef; 16]>,
        value: Option<Vec<u8>>,
    },
}

impl Default for Node {
    fn default() -> Self {
        Node::Empty
    }
}

impl Node {
    /// Create empty branch node
    pub fn empty_branch() -> Self {
        Node::Branch {
            children: Box::new([
                NodeRef::Empty, NodeRef::Empty, NodeRef::Empty, NodeRef::Empty,
                NodeRef::Empty, NodeRef::Empty, NodeRef::Empty, NodeRef::Empty,
                NodeRef::Empty, NodeRef::Empty, NodeRef::Empty, NodeRef::Empty,
                NodeRef::Empty, NodeRef::Empty, NodeRef::Empty, NodeRef::Empty,
            ]),
            value: None,
        }
    }

    /// Create leaf node
    pub fn leaf(key: Nibbles, value: Vec<u8>) -> Self {
        Node::Leaf { key, value }
    }

    /// Create extension node
    pub fn extension(key: Nibbles, child: NodeRef) -> Self {
        Node::Extension { key, child }
    }

    /// Check if node is empty
    pub fn is_empty(&self) -> bool {
        matches!(self, Node::Empty)
    }

    /// RLP encode this node
    pub fn rlp_encode(&self) -> Vec<u8> {
        match self {
            Node::Empty => vec![0x80], // RLP null

            Node::Leaf { key, value } => {
                let path = key.to_hex_prefix(true);
                rlp_encode_list(&[&path[..], value])
            }

            Node::Extension { key, child } => {
                let path = key.to_hex_prefix(false);
                let child_data = match child {
                    NodeRef::Empty => vec![0x80],
                    NodeRef::Inline(data) => data.clone(),
                    NodeRef::Hash(h) => rlp_encode_bytes(h.as_bytes()),
                };
                rlp_encode_two_items(&path, &child_data)
            }

            Node::Branch { children, value } => {
                let mut items: Vec<Vec<u8>> = Vec::with_capacity(17);

                for child in children.iter() {
                    let encoded = match child {
                        NodeRef::Empty => vec![0x80],
                        NodeRef::Inline(data) => data.clone(),
                        NodeRef::Hash(h) => rlp_encode_bytes(h.as_bytes()),
                    };
                    items.push(encoded);
                }

                // Add value (or empty)
                match value {
                    Some(v) => items.push(rlp_encode_bytes(v)),
                    None => items.push(vec![0x80]),
                };

                rlp_encode_list_raw(&items)
            }
        }
    }

    /// Get hash of this node
    /// If RLP encoding is < 32 bytes, returns Inline reference
    /// Otherwise returns Hash reference
    pub fn hash(&self) -> NodeRef {
        if self.is_empty() {
            return NodeRef::Empty;
        }

        let encoded = self.rlp_encode();

        if encoded.len() < 32 {
            NodeRef::Inline(encoded)
        } else {
            NodeRef::Hash(keccak256(&encoded))
        }
    }

    /// Get root hash (always returns H256)
    pub fn root_hash(&self) -> H256 {
        if self.is_empty() {
            // Empty trie root = keccak256(RLP(""))
            keccak256(&[0x80])
        } else {
            let encoded = self.rlp_encode();
            keccak256(&encoded)
        }
    }
}

// =========================================
// RLP Encoding Helpers
// =========================================

/// Encode bytes with RLP string encoding
fn rlp_encode_bytes(data: &[u8]) -> Vec<u8> {
    if data.len() == 1 && data[0] < 0x80 {
        // Single byte < 0x80 is encoded as itself
        data.to_vec()
    } else if data.len() < 56 {
        let mut result = vec![0x80 + data.len() as u8];
        result.extend_from_slice(data);
        result
    } else {
        let len_bytes = data.len().to_be_bytes();
        let len_bytes: Vec<u8> = len_bytes.iter().skip_while(|&&b| b == 0).copied().collect();
        let mut result = vec![0xb7 + len_bytes.len() as u8];
        result.extend_from_slice(&len_bytes);
        result.extend_from_slice(data);
        result
    }
}

/// Encode list with RLP
fn rlp_encode_list(items: &[&[u8]]) -> Vec<u8> {
    let mut payload = Vec::new();
    for item in items {
        payload.extend_from_slice(&rlp_encode_bytes(item));
    }

    rlp_encode_list_payload(&payload)
}

/// Encode two items as a list (for extension/leaf)
fn rlp_encode_two_items(path: &[u8], child: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&rlp_encode_bytes(path));
    // Child is already RLP encoded
    payload.extend_from_slice(child);

    rlp_encode_list_payload(&payload)
}

/// Encode raw items (already RLP encoded) as a list
fn rlp_encode_list_raw(items: &[Vec<u8>]) -> Vec<u8> {
    let mut payload = Vec::new();
    for item in items {
        payload.extend_from_slice(item);
    }

    rlp_encode_list_payload(&payload)
}

/// Add list prefix to payload
fn rlp_encode_list_payload(payload: &[u8]) -> Vec<u8> {
    if payload.len() < 56 {
        let mut result = vec![0xc0 + payload.len() as u8];
        result.extend_from_slice(payload);
        result
    } else {
        let len_bytes = payload.len().to_be_bytes();
        let len_bytes: Vec<u8> = len_bytes.iter().skip_while(|&&b| b == 0).copied().collect();
        let mut result = vec![0xf7 + len_bytes.len() as u8];
        result.extend_from_slice(&len_bytes);
        result.extend_from_slice(payload);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_node() {
        let node = Node::Empty;
        assert!(node.is_empty());

        let encoded = node.rlp_encode();
        assert_eq!(encoded, vec![0x80]);
    }

    #[test]
    fn test_leaf_node() {
        let key = Nibbles::from_raw(vec![1, 2, 3]);
        let value = b"hello".to_vec();
        let node = Node::leaf(key, value);

        let encoded = node.rlp_encode();
        assert!(!encoded.is_empty());

        // Verify it produces a hash
        let hash = node.hash();
        assert!(!matches!(hash, NodeRef::Empty));
    }

    #[test]
    fn test_branch_node() {
        let mut node = Node::empty_branch();

        if let Node::Branch { ref mut children, ref mut value } = node {
            children[0] = NodeRef::Hash(keccak256(b"test"));
            *value = Some(b"value".to_vec());
        }

        let encoded = node.rlp_encode();
        assert!(!encoded.is_empty());

        // Branch nodes are typically > 32 bytes
        let hash = node.hash();
        assert!(matches!(hash, NodeRef::Hash(_)));
    }

    #[test]
    fn test_extension_node() {
        let key = Nibbles::from_raw(vec![1, 2, 3, 4]);
        let child_hash = keccak256(b"child");
        let node = Node::extension(key, NodeRef::Hash(child_hash));

        let encoded = node.rlp_encode();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_node_ref() {
        let empty = NodeRef::Empty;
        assert!(empty.is_empty());

        let hash = NodeRef::Hash(keccak256(b"test"));
        assert!(!hash.is_empty());
        assert!(hash.as_hash().is_some());

        let inline = NodeRef::Inline(vec![1, 2, 3]);
        assert!(inline.as_hash().is_none());
    }
}
