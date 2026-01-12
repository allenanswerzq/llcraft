//! # Patricia Trie
//!
//! The main trie data structure with insert, get, and delete operations.

use std::collections::HashMap;
use eth_primitives::{H256, keccak256};
use crate::nibbles::Nibbles;
use crate::node::{Node, NodeRef};
use crate::error::{Result, TrieError};

/// Empty trie root hash (keccak256(RLP("")))
pub const EMPTY_ROOT: [u8; 32] = [
    0x56, 0xe8, 0x1f, 0x17, 0x1b, 0xcc, 0x55, 0xa6,
    0xff, 0x83, 0x45, 0xe6, 0x92, 0xc0, 0xf8, 0x6e,
    0x5b, 0x48, 0xe0, 0x1b, 0x99, 0x6c, 0xad, 0xc0,
    0x01, 0x62, 0x2f, 0xb5, 0xe3, 0x63, 0xb4, 0x21,
];

/// Database interface for storing nodes
pub trait TrieDB {
    /// Get node by hash
    fn get(&self, hash: &H256) -> Option<Vec<u8>>;

    /// Store node, returns hash
    fn insert(&mut self, data: Vec<u8>) -> H256;

    /// Remove node by hash
    fn remove(&mut self, hash: &H256);
}

/// In-memory trie database
#[derive(Debug, Clone, Default)]
pub struct MemoryDB {
    nodes: HashMap<H256, Vec<u8>>,
}

impl MemoryDB {
    pub fn new() -> Self {
        MemoryDB {
            nodes: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl TrieDB for MemoryDB {
    fn get(&self, hash: &H256) -> Option<Vec<u8>> {
        self.nodes.get(hash).cloned()
    }

    fn insert(&mut self, data: Vec<u8>) -> H256 {
        let hash = keccak256(&data);
        self.nodes.insert(hash, data);
        hash
    }

    fn remove(&mut self, hash: &H256) {
        self.nodes.remove(hash);
    }
}

/// Merkle Patricia Trie
#[derive(Debug)]
pub struct PatriciaTrie<DB: TrieDB> {
    /// Root node
    root: Node,
    /// Node database
    db: DB,
}

impl<DB: TrieDB> PatriciaTrie<DB> {
    /// Create new empty trie
    pub fn new(db: DB) -> Self {
        PatriciaTrie {
            root: Node::Empty,
            db,
        }
    }

    /// Get root hash
    pub fn root_hash(&self) -> H256 {
        self.root.root_hash()
    }

    /// Check if trie is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// Get value for key
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let nibbles = Nibbles::from_bytes(key);
        self.get_node(&self.root, &nibbles)
    }

    /// Collect proof nodes along the path for a key
    /// Returns (value, proof_nodes) where proof_nodes are RLP-encoded nodes
    pub fn get_with_proof(&self, key: &[u8]) -> (Option<Vec<u8>>, Vec<Vec<u8>>) {
        let nibbles = Nibbles::from_bytes(key);
        let mut proof_nodes = Vec::new();
        let value = self.collect_proof_nodes(&self.root, &nibbles, &mut proof_nodes);
        (value, proof_nodes)
    }

    /// Internal recursive proof collection
    fn collect_proof_nodes(&self, node: &Node, key: &Nibbles, proof: &mut Vec<Vec<u8>>) -> Option<Vec<u8>> {
        match node {
            Node::Empty => None,

            Node::Leaf { key: leaf_key, value } => {
                // Add this node to proof
                proof.push(node.rlp_encode());
                if leaf_key == key {
                    Some(value.clone())
                } else {
                    None
                }
            }

            Node::Extension { key: ext_key, child } => {
                // Add this node to proof
                proof.push(node.rlp_encode());

                if key.len() < ext_key.len() {
                    return None;
                }

                let prefix_len = key.common_prefix_len(ext_key);
                if prefix_len != ext_key.len() {
                    return None;
                }

                let remaining = key.slice(ext_key.len());
                let child_node = self.resolve_ref(child)?;
                self.collect_proof_nodes(&child_node, &remaining, proof)
            }

            Node::Branch { children, value } => {
                // Add this node to proof
                proof.push(node.rlp_encode());

                if key.is_empty() {
                    return value.clone();
                }

                let idx = key.first()? as usize;
                let child = &children[idx];

                if child.is_empty() {
                    return None;
                }

                let child_node = self.resolve_ref(child)?;
                let remaining = key.slice(1);
                self.collect_proof_nodes(&child_node, &remaining, proof)
            }
        }
    }

    /// Internal recursive get
    fn get_node(&self, node: &Node, key: &Nibbles) -> Option<Vec<u8>> {
        match node {
            Node::Empty => None,

            Node::Leaf { key: leaf_key, value } => {
                if leaf_key == key {
                    Some(value.clone())
                } else {
                    None
                }
            }

            Node::Extension { key: ext_key, child } => {
                if key.len() < ext_key.len() {
                    return None;
                }

                let prefix_len = key.common_prefix_len(ext_key);
                if prefix_len != ext_key.len() {
                    return None;
                }

                let remaining = key.slice(ext_key.len());
                let child_node = self.resolve_ref(child)?;
                self.get_node(&child_node, &remaining)
            }

            Node::Branch { children, value } => {
                if key.is_empty() {
                    return value.clone();
                }

                let idx = key.first()? as usize;
                let child = &children[idx];

                if child.is_empty() {
                    return None;
                }

                let child_node = self.resolve_ref(child)?;
                let remaining = key.slice(1);
                self.get_node(&child_node, &remaining)
            }
        }
    }

    /// Resolve a node reference
    fn resolve_ref(&self, node_ref: &NodeRef) -> Option<Node> {
        match node_ref {
            NodeRef::Empty => Some(Node::Empty),
            NodeRef::Inline(data) => self.decode_node(data),
            NodeRef::Hash(hash) => {
                let data: Vec<u8> = self.db.get(hash)?;
                self.decode_node(&data)
            }
        }
    }

    /// Decode RLP-encoded node (simplified)
    fn decode_node(&self, data: &[u8]) -> Option<Node> {
        if data.is_empty() || data == [0x80] {
            return Some(Node::Empty);
        }

        // Parse RLP list
        let (items, _) = decode_rlp_list(data)?;

        if items.len() == 2 {
            // Leaf or Extension
            let path_bytes = &items[0];
            let mut is_leaf = false;
            let key = Nibbles::from_hex_prefix(path_bytes, &mut is_leaf);

            if is_leaf {
                Some(Node::Leaf {
                    key,
                    value: items[1].clone(),
                })
            } else {
                let child_ref = self.bytes_to_ref(&items[1]);
                Some(Node::Extension { key, child: child_ref })
            }
        } else if items.len() == 17 {
            // Branch
            let mut children: [NodeRef; 16] = Default::default();
            for (i, item) in items[..16].iter().enumerate() {
                children[i] = self.bytes_to_ref(item);
            }

            let value = if items[16].is_empty() || items[16] == [0x80] {
                None
            } else {
                Some(items[16].clone())
            };

            Some(Node::Branch {
                children: Box::new(children),
                value,
            })
        } else {
            None
        }
    }

    /// Convert bytes to NodeRef
    fn bytes_to_ref(&self, data: &[u8]) -> NodeRef {
        if data.is_empty() || data == [0x80] {
            NodeRef::Empty
        } else if data.len() == 32 {
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(data);
            NodeRef::Hash(H256::new(bytes))
        } else {
            NodeRef::Inline(data.to_vec())
        }
    }

    /// Insert key-value pair
    pub fn insert(&mut self, key: &[u8], value: Vec<u8>) {
        let nibbles = Nibbles::from_bytes(key);
        let new_root = self.insert_node(self.root.clone(), nibbles, value);
        self.root = new_root;
    }

    /// Internal recursive insert
    fn insert_node(&mut self, node: Node, key: Nibbles, value: Vec<u8>) -> Node {
        match node {
            Node::Empty => {
                // Create new leaf
                Node::Leaf { key, value }
            }

            Node::Leaf { key: leaf_key, value: leaf_value } => {
                if leaf_key == key {
                    // Update existing leaf
                    Node::Leaf { key, value }
                } else {
                    // Split into branch
                    let common_len = key.common_prefix_len(&leaf_key);

                    if common_len == 0 {
                        // No common prefix - create branch
                        let mut branch = Node::empty_branch();

                        if let Node::Branch { ref mut children, value: ref mut branch_value } = branch {
                            // Insert existing leaf
                            if leaf_key.is_empty() {
                                *branch_value = Some(leaf_value);
                            } else {
                                let idx = leaf_key.first().unwrap() as usize;
                                let child = Node::Leaf {
                                    key: leaf_key.slice(1),
                                    value: leaf_value,
                                };
                                children[idx] = self.store_node(child);
                            }

                            // Insert new value
                            if key.is_empty() {
                                *branch_value = Some(value);
                            } else {
                                let idx = key.first().unwrap() as usize;
                                let child = Node::Leaf {
                                    key: key.slice(1),
                                    value,
                                };
                                children[idx] = self.store_node(child);
                            }
                        }

                        branch
                    } else {
                        // Common prefix - create extension
                        let prefix = key.slice_range(0, common_len);
                        let remaining_key = key.slice(common_len);
                        let remaining_leaf = leaf_key.slice(common_len);

                        // Create branch for the split point
                        let mut branch = Node::empty_branch();

                        if let Node::Branch { ref mut children, value: ref mut branch_value } = branch {
                            // Existing leaf
                            if remaining_leaf.is_empty() {
                                *branch_value = Some(leaf_value);
                            } else {
                                let idx = remaining_leaf.first().unwrap() as usize;
                                let child = Node::Leaf {
                                    key: remaining_leaf.slice(1),
                                    value: leaf_value,
                                };
                                children[idx] = self.store_node(child);
                            }

                            // New value
                            if remaining_key.is_empty() {
                                *branch_value = Some(value);
                            } else {
                                let idx = remaining_key.first().unwrap() as usize;
                                let child = Node::Leaf {
                                    key: remaining_key.slice(1),
                                    value,
                                };
                                children[idx] = self.store_node(child);
                            }
                        }

                        // Wrap with extension if prefix exists
                        Node::Extension {
                            key: prefix,
                            child: self.store_node(branch),
                        }
                    }
                }
            }

            Node::Extension { key: ext_key, child } => {
                let common_len = key.common_prefix_len(&ext_key);

                if common_len == ext_key.len() {
                    // Full match - descend into child
                    let child_node = self.resolve_ref(&child).unwrap_or(Node::Empty);
                    let remaining = key.slice(ext_key.len());
                    let new_child = self.insert_node(child_node, remaining, value);

                    Node::Extension {
                        key: ext_key,
                        child: self.store_node(new_child),
                    }
                } else if common_len == 0 {
                    // No match - create branch
                    let mut branch = Node::empty_branch();

                    if let Node::Branch { ref mut children, value: ref mut branch_value } = branch {
                        // Existing extension
                        let ext_idx = ext_key.first().unwrap() as usize;
                        if ext_key.len() == 1 {
                            children[ext_idx] = child;
                        } else {
                            let new_ext = Node::Extension {
                                key: ext_key.slice(1),
                                child,
                            };
                            children[ext_idx] = self.store_node(new_ext);
                        }

                        // New value
                        if key.is_empty() {
                            *branch_value = Some(value);
                        } else {
                            let idx = key.first().unwrap() as usize;
                            let child = Node::Leaf {
                                key: key.slice(1),
                                value,
                            };
                            children[idx] = self.store_node(child);
                        }
                    }

                    branch
                } else {
                    // Partial match - split extension
                    let prefix = ext_key.slice_range(0, common_len);
                    let ext_remaining = ext_key.slice(common_len);
                    let key_remaining = key.slice(common_len);

                    let mut branch = Node::empty_branch();

                    if let Node::Branch { ref mut children, value: ref mut branch_value } = branch {
                        // Remaining extension
                        let ext_idx = ext_remaining.first().unwrap() as usize;
                        if ext_remaining.len() == 1 {
                            children[ext_idx] = child;
                        } else {
                            let new_ext = Node::Extension {
                                key: ext_remaining.slice(1),
                                child,
                            };
                            children[ext_idx] = self.store_node(new_ext);
                        }

                        // New value
                        if key_remaining.is_empty() {
                            *branch_value = Some(value);
                        } else {
                            let idx = key_remaining.first().unwrap() as usize;
                            let child = Node::Leaf {
                                key: key_remaining.slice(1),
                                value,
                            };
                            children[idx] = self.store_node(child);
                        }
                    }

                    Node::Extension {
                        key: prefix,
                        child: self.store_node(branch),
                    }
                }
            }

            Node::Branch { mut children, value: branch_value } => {
                if key.is_empty() {
                    // Set value at branch
                    Node::Branch {
                        children,
                        value: Some(value),
                    }
                } else {
                    // Descend into child
                    let idx = key.first().unwrap() as usize;
                    let child = std::mem::replace(&mut children[idx], NodeRef::Empty);
                    let child_node = self.resolve_ref(&child).unwrap_or(Node::Empty);
                    let remaining = key.slice(1);
                    let new_child = self.insert_node(child_node, remaining, value);
                    children[idx] = self.store_node(new_child);

                    Node::Branch {
                        children,
                        value: branch_value,
                    }
                }
            }
        }
    }

    /// Store node in database, return reference
    fn store_node(&mut self, node: Node) -> NodeRef {
        if node.is_empty() {
            return NodeRef::Empty;
        }

        let encoded = node.rlp_encode();

        if encoded.len() < 32 {
            NodeRef::Inline(encoded)
        } else {
            let hash = self.db.insert(encoded);
            NodeRef::Hash(hash)
        }
    }

    /// Delete key from trie
    pub fn delete(&mut self, key: &[u8]) -> bool {
        let nibbles = Nibbles::from_bytes(key);
        if let Some(new_root) = self.delete_node(self.root.clone(), nibbles) {
            self.root = new_root;
            true
        } else {
            false
        }
    }

    /// Internal recursive delete
    fn delete_node(&mut self, node: Node, key: Nibbles) -> Option<Node> {
        match node {
            Node::Empty => None,

            Node::Leaf { key: leaf_key, .. } => {
                if leaf_key == key {
                    Some(Node::Empty)
                } else {
                    None
                }
            }

            Node::Extension { key: ext_key, child } => {
                if key.len() < ext_key.len() {
                    return None;
                }

                let prefix_len = key.common_prefix_len(&ext_key);
                if prefix_len != ext_key.len() {
                    return None;
                }

                let remaining = key.slice(ext_key.len());
                let child_node = self.resolve_ref(&child)?;
                let new_child = self.delete_node(child_node, remaining)?;

                // Collapse if possible
                Some(self.collapse_extension(ext_key, new_child))
            }

            Node::Branch { mut children, value } => {
                if key.is_empty() {
                    if value.is_none() {
                        return None;
                    }

                    // Remove value, collapse if possible
                    return Some(self.collapse_branch(children, None));
                }

                let idx = key.first()? as usize;
                let child = std::mem::replace(&mut children[idx], NodeRef::Empty);

                if child.is_empty() {
                    return None;
                }

                let child_node = self.resolve_ref(&child)?;
                let remaining = key.slice(1);
                let new_child = self.delete_node(child_node, remaining)?;

                children[idx] = self.store_node(new_child);

                Some(self.collapse_branch(children, value))
            }
        }
    }

    /// Collapse extension after child modification
    fn collapse_extension(&mut self, key: Nibbles, child: Node) -> Node {
        match child {
            Node::Empty => Node::Empty,

            Node::Leaf { key: child_key, value } => {
                // Merge keys
                let mut new_key = key;
                new_key.extend(&child_key);
                Node::Leaf { key: new_key, value }
            }

            Node::Extension { key: child_key, child: grandchild } => {
                // Merge extensions
                let mut new_key = key;
                new_key.extend(&child_key);
                Node::Extension { key: new_key, child: grandchild }
            }

            _ => {
                Node::Extension {
                    key,
                    child: self.store_node(child),
                }
            }
        }
    }

    /// Collapse branch after child deletion
    fn collapse_branch(&mut self, children: Box<[NodeRef; 16]>, value: Option<Vec<u8>>) -> Node {
        // Count non-empty children
        let mut non_empty: Vec<(usize, &NodeRef)> = children.iter()
            .enumerate()
            .filter(|(_, c)| !c.is_empty())
            .collect();

        let child_count = non_empty.len();
        let has_value = value.is_some();

        if child_count == 0 && !has_value {
            Node::Empty
        } else if child_count == 0 && has_value {
            // Only value - convert to leaf
            Node::Leaf {
                key: Nibbles::new(),
                value: value.unwrap(),
            }
        } else if child_count == 1 && !has_value {
            // Single child - try to collapse
            let (idx, child_ref) = non_empty.remove(0);
            let child = self.resolve_ref(child_ref).unwrap_or(Node::Empty);

            let mut prefix = Nibbles::new();
            prefix.push(idx as u8);

            match child {
                Node::Leaf { key, value } => {
                    prefix.extend(&key);
                    Node::Leaf { key: prefix, value }
                }
                Node::Extension { key, child } => {
                    prefix.extend(&key);
                    Node::Extension { key: prefix, child }
                }
                _ => {
                    Node::Extension {
                        key: prefix,
                        child: self.store_node(child),
                    }
                }
            }
        } else {
            // Keep as branch
            Node::Branch { children, value }
        }
    }
}

impl PatriciaTrie<MemoryDB> {
    /// Create new trie with in-memory database
    pub fn new_memory() -> Self {
        PatriciaTrie::new(MemoryDB::new())
    }
}

/// Decode RLP list (simplified)
fn decode_rlp_list(data: &[u8]) -> Option<(Vec<Vec<u8>>, usize)> {
    if data.is_empty() {
        return None;
    }

    let first = data[0];

    if first <= 0xbf {
        // Not a list - single item
        return None;
    }

    let (payload, payload_start) = if first <= 0xf7 {
        // Short list
        let len = (first - 0xc0) as usize;
        (&data[1..1+len], 1)
    } else {
        // Long list
        let len_len = (first - 0xf7) as usize;
        let mut len = 0usize;
        for i in 0..len_len {
            len = (len << 8) | data[1 + i] as usize;
        }
        let start = 1 + len_len;
        (&data[start..start+len], start)
    };

    // Parse items from payload
    let mut items = Vec::new();
    let mut pos = 0;

    while pos < payload.len() {
        let (item, item_len) = decode_rlp_item(&payload[pos..])?;
        items.push(item);
        pos += item_len;
    }

    Some((items, payload_start + payload.len()))
}

/// Decode single RLP item
fn decode_rlp_item(data: &[u8]) -> Option<(Vec<u8>, usize)> {
    if data.is_empty() {
        return None;
    }

    let first = data[0];

    if first < 0x80 {
        // Single byte
        Some((vec![first], 1))
    } else if first <= 0xb7 {
        // Short string
        let len = (first - 0x80) as usize;
        if len == 0 {
            Some((vec![], 1))
        } else {
            Some((data[1..1+len].to_vec(), 1 + len))
        }
    } else if first <= 0xbf {
        // Long string
        let len_len = (first - 0xb7) as usize;
        let mut len = 0usize;
        for i in 0..len_len {
            len = (len << 8) | data[1 + i] as usize;
        }
        let start = 1 + len_len;
        Some((data[start..start+len].to_vec(), start + len))
    } else if first <= 0xf7 {
        // Short list - return as raw bytes
        let len = (first - 0xc0) as usize;
        Some((data[..1+len].to_vec(), 1 + len))
    } else {
        // Long list - return as raw bytes
        let len_len = (first - 0xf7) as usize;
        let mut len = 0usize;
        for i in 0..len_len {
            len = (len << 8) | data[1 + i] as usize;
        }
        let total_len = 1 + len_len + len;
        Some((data[..total_len].to_vec(), total_len))
    }
}

impl Default for NodeRef {
    fn default() -> Self {
        NodeRef::Empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_trie() {
        let trie = PatriciaTrie::new_memory();
        assert!(trie.is_empty());

        let root = trie.root_hash();
        assert_eq!(root.as_bytes(), &EMPTY_ROOT);
    }

    #[test]
    fn test_single_insert() {
        let mut trie = PatriciaTrie::new_memory();

        trie.insert(b"hello", b"world".to_vec());

        assert!(!trie.is_empty());
        assert_eq!(trie.get(b"hello"), Some(b"world".to_vec()));
        assert_eq!(trie.get(b"other"), None);
    }

    #[test]
    fn test_multiple_insert() {
        let mut trie = PatriciaTrie::new_memory();

        trie.insert(b"do", b"verb".to_vec());
        trie.insert(b"dog", b"puppy".to_vec());
        trie.insert(b"doge", b"coin".to_vec());
        trie.insert(b"horse", b"stallion".to_vec());

        assert_eq!(trie.get(b"do"), Some(b"verb".to_vec()));
        assert_eq!(trie.get(b"dog"), Some(b"puppy".to_vec()));
        assert_eq!(trie.get(b"doge"), Some(b"coin".to_vec()));
        assert_eq!(trie.get(b"horse"), Some(b"stallion".to_vec()));
        assert_eq!(trie.get(b"cat"), None);
    }

    #[test]
    fn test_update() {
        let mut trie = PatriciaTrie::new_memory();

        trie.insert(b"key", b"value1".to_vec());
        assert_eq!(trie.get(b"key"), Some(b"value1".to_vec()));

        trie.insert(b"key", b"value2".to_vec());
        assert_eq!(trie.get(b"key"), Some(b"value2".to_vec()));
    }

    #[test]
    fn test_delete() {
        let mut trie = PatriciaTrie::new_memory();

        trie.insert(b"do", b"verb".to_vec());
        trie.insert(b"dog", b"puppy".to_vec());

        assert!(trie.delete(b"do"));
        assert_eq!(trie.get(b"do"), None);
        assert_eq!(trie.get(b"dog"), Some(b"puppy".to_vec()));
    }

    #[test]
    fn test_root_changes() {
        let mut trie = PatriciaTrie::new_memory();

        let empty_root = trie.root_hash();

        trie.insert(b"key", b"value".to_vec());
        let root1 = trie.root_hash();
        assert_ne!(root1, empty_root);

        trie.insert(b"key2", b"value2".to_vec());
        let root2 = trie.root_hash();
        assert_ne!(root2, root1);

        // Same key-values should produce same root
        let mut trie2 = PatriciaTrie::new_memory();
        trie2.insert(b"key", b"value".to_vec());
        trie2.insert(b"key2", b"value2".to_vec());
        assert_eq!(trie2.root_hash(), root2);
    }

    #[test]
    fn test_many_keys() {
        let mut trie = PatriciaTrie::new_memory();

        for i in 0u32..100 {
            let key = format!("key{}", i);
            let value = format!("value{}", i);
            trie.insert(key.as_bytes(), value.as_bytes().to_vec());
        }

        for i in 0u32..100 {
            let key = format!("key{}", i);
            let expected = format!("value{}", i);
            assert_eq!(trie.get(key.as_bytes()), Some(expected.as_bytes().to_vec()));
        }
    }
}
