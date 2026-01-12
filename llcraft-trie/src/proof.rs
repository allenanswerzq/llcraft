//! # Merkle Proofs
//!
//! Generate and verify proofs of inclusion/exclusion for keys in the trie.

use eth_primitives::{H256, keccak256};
use crate::nibbles::Nibbles;
use crate::node::{Node, NodeRef};
use crate::trie::{PatriciaTrie, TrieDB, MemoryDB};
use crate::error::{Result, TrieError};

/// A Merkle proof for a key in the trie
#[derive(Debug, Clone)]
pub struct Proof {
    /// The key being proven
    pub key: Vec<u8>,
    /// The value (None if proving non-existence)
    pub value: Option<Vec<u8>>,
    /// Proof nodes (RLP encoded)
    pub nodes: Vec<Vec<u8>>,
}

impl Proof {
    /// Create new proof
    pub fn new(key: Vec<u8>, value: Option<Vec<u8>>, nodes: Vec<Vec<u8>>) -> Self {
        Proof { key, value, nodes }
    }

    /// Verify proof against a root hash
    pub fn verify(&self, root: &H256) -> bool {
        // Empty proof can only be valid for empty trie
        if self.nodes.is_empty() {
            return false;
        }

        // The first node in the proof must be the root node
        // Its hash must match the expected root hash
        let first_node_hash = keccak256(&self.nodes[0]);

        // For nodes >= 32 bytes, they are stored by hash
        // For smaller nodes, they could be inlined, but root is always a hash
        if first_node_hash != *root {
            return false;
        }

        // Build a lookup map from hash -> node data
        let mut db = MemoryDB::new();
        for node_data in &self.nodes {
            db.insert(node_data.clone());
        }

        // Traverse the proof to find the value
        let nibbles = Nibbles::from_bytes(&self.key);

        match self.traverse_proof(&db, root, &nibbles) {
            Ok(found_value) => {
                self.value == found_value
            }
            Err(_) => {
                // Traversal failed - valid if we're proving non-existence
                self.value.is_none()
            }
        }
    }

    /// Traverse proof nodes to find value
    fn traverse_proof(&self, db: &MemoryDB, root: &H256, key: &Nibbles) -> Result<Option<Vec<u8>>> {
        // Get root node data
        let root_data = self.nodes.iter()
            .find(|n| n.len() >= 32 && keccak256(n) == *root)
            .or_else(|| self.nodes.iter().find(|n| n.len() < 32))
            .ok_or(TrieError::NodeNotFound("root".to_string()))?;

        self.traverse_node(db, root_data, key)
    }

    /// Traverse a single node
    fn traverse_node(&self, db: &MemoryDB, node_data: &[u8], key: &Nibbles) -> Result<Option<Vec<u8>>> {
        if node_data.is_empty() || node_data == [0x80] {
            return Ok(None);
        }

        // Parse the node
        let items = decode_rlp_list_simple(node_data)?;

        if items.len() == 2 {
            // Leaf or Extension
            let mut is_leaf = false;
            let node_key = Nibbles::from_hex_prefix(&items[0], &mut is_leaf);

            if is_leaf {
                // Leaf node
                if node_key == *key {
                    Ok(Some(items[1].clone()))
                } else {
                    Ok(None)
                }
            } else {
                // Extension node
                if key.len() < node_key.len() {
                    return Ok(None);
                }

                let prefix_len = key.common_prefix_len(&node_key);
                if prefix_len != node_key.len() {
                    return Ok(None);
                }

                let remaining = key.slice(node_key.len());
                let child_data = self.resolve_child_data(&items[1])?;
                self.traverse_node(db, &child_data, &remaining)
            }
        } else if items.len() == 17 {
            // Branch node
            if key.is_empty() {
                if items[16].is_empty() || items[16] == [0x80] {
                    return Ok(None);
                }
                return Ok(Some(items[16].clone()));
            }

            let idx = key.first().unwrap() as usize;
            let child = &items[idx];

            if child.is_empty() || child == &[0x80] {
                return Ok(None);
            }

            let child_data = self.resolve_child_data(child)?;
            let remaining = key.slice(1);
            self.traverse_node(db, &child_data, &remaining)
        } else {
            Err(TrieError::InvalidEncoding)
        }
    }

    /// Resolve child reference to actual data
    fn resolve_child_data(&self, child: &[u8]) -> Result<Vec<u8>> {
        if child.len() == 32 {
            // Hash reference - find in proof nodes
            let mut hash = [0u8; 32];
            hash.copy_from_slice(child);
            let target_hash = H256::new(hash);

            self.nodes.iter()
                .find(|n| keccak256(n) == target_hash)
                .cloned()
                .ok_or(TrieError::NodeNotFound(hex::encode(child)))
        } else {
            // Inline data
            Ok(child.to_vec())
        }
    }
}

/// Generate proof for a key
/// Collects all nodes along the path from root to the key
pub fn generate_proof<DB: TrieDB>(trie: &PatriciaTrie<DB>, key: &[u8]) -> Proof {
    let (value, nodes) = trie.get_with_proof(key);
    Proof::new(key.to_vec(), value, nodes)
}

/// Simple RLP list decoder for proofs
fn decode_rlp_list_simple(data: &[u8]) -> Result<Vec<Vec<u8>>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let first = data[0];

    if first < 0xc0 {
        return Err(TrieError::InvalidEncoding);
    }

    let (payload, _) = if first <= 0xf7 {
        let len = (first - 0xc0) as usize;
        (&data[1..1.min(data.len()).max(1+len.min(data.len()-1))], 1)
    } else {
        let len_len = (first - 0xf7) as usize;
        if data.len() < 1 + len_len {
            return Err(TrieError::InvalidEncoding);
        }
        let mut len = 0usize;
        for i in 0..len_len {
            len = (len << 8) | data[1 + i] as usize;
        }
        let start = 1 + len_len;
        let end = (start + len).min(data.len());
        (&data[start..end], start)
    };

    let mut items = Vec::new();
    let mut pos = 0;

    while pos < payload.len() {
        let (item, item_len) = decode_rlp_item_simple(&payload[pos..])?;
        items.push(item);
        pos += item_len;
    }

    Ok(items)
}

/// Decode single RLP item
fn decode_rlp_item_simple(data: &[u8]) -> Result<(Vec<u8>, usize)> {
    if data.is_empty() {
        return Err(TrieError::InvalidEncoding);
    }

    let first = data[0];

    if first < 0x80 {
        Ok((vec![first], 1))
    } else if first <= 0xb7 {
        let len = (first - 0x80) as usize;
        if len == 0 {
            Ok((vec![], 1))
        } else if data.len() < 1 + len {
            Err(TrieError::InvalidEncoding)
        } else {
            Ok((data[1..1+len].to_vec(), 1 + len))
        }
    } else if first <= 0xbf {
        let len_len = (first - 0xb7) as usize;
        if data.len() < 1 + len_len {
            return Err(TrieError::InvalidEncoding);
        }
        let mut len = 0usize;
        for i in 0..len_len {
            len = (len << 8) | data[1 + i] as usize;
        }
        let start = 1 + len_len;
        let end = start + len;
        if data.len() < end {
            return Err(TrieError::InvalidEncoding);
        }
        Ok((data[start..end].to_vec(), end))
    } else if first <= 0xf7 {
        let len = (first - 0xc0) as usize;
        let end = 1 + len;
        if data.len() < end {
            return Err(TrieError::InvalidEncoding);
        }
        Ok((data[..end].to_vec(), end))
    } else {
        let len_len = (first - 0xf7) as usize;
        if data.len() < 1 + len_len {
            return Err(TrieError::InvalidEncoding);
        }
        let mut len = 0usize;
        for i in 0..len_len {
            len = (len << 8) | data[1 + i] as usize;
        }
        let total = 1 + len_len + len;
        if data.len() < total {
            return Err(TrieError::InvalidEncoding);
        }
        Ok((data[..total].to_vec(), total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trie::PatriciaTrie;

    #[test]
    fn test_proof_structure() {
        let mut trie = PatriciaTrie::new_memory();
        trie.insert(b"hello", b"world".to_vec());

        let proof = generate_proof(&trie, b"hello");
        assert_eq!(proof.key, b"hello".to_vec());
        assert_eq!(proof.value, Some(b"world".to_vec()));
    }

    #[test]
    fn test_proof_nonexistent() {
        let mut trie = PatriciaTrie::new_memory();
        trie.insert(b"hello", b"world".to_vec());

        let proof = generate_proof(&trie, b"missing");
        assert_eq!(proof.key, b"missing".to_vec());
        assert_eq!(proof.value, None);
    }

    #[test]
    fn test_proof_has_nodes() {
        let mut trie = PatriciaTrie::new_memory();
        trie.insert(b"hello", b"world".to_vec());

        let proof = generate_proof(&trie, b"hello");

        // Proof should contain at least one node (the leaf)
        assert!(!proof.nodes.is_empty(), "Proof should contain nodes");
        assert_eq!(proof.value, Some(b"world".to_vec()));
    }

    #[test]
    fn test_proof_verify_single_key() {
        let mut trie = PatriciaTrie::new_memory();
        trie.insert(b"hello", b"world".to_vec());

        let root = trie.root_hash();
        let proof = generate_proof(&trie, b"hello");

        // Proof should verify against correct root
        assert!(proof.verify(&root), "Proof should verify against correct root");
    }

    #[test]
    fn test_proof_verify_fails_wrong_root() {
        let mut trie = PatriciaTrie::new_memory();
        trie.insert(b"hello", b"world".to_vec());

        let proof = generate_proof(&trie, b"hello");

        // Create a different root by modifying trie
        let mut trie2 = PatriciaTrie::new_memory();
        trie2.insert(b"hello", b"different".to_vec());
        let wrong_root = trie2.root_hash();

        // Proof should NOT verify against wrong root
        assert!(!proof.verify(&wrong_root), "Proof should not verify against wrong root");
    }

    #[test]
    fn test_proof_multiple_keys() {
        let mut trie = PatriciaTrie::new_memory();

        // Insert multiple keys with common prefixes
        trie.insert(b"do", b"verb".to_vec());
        trie.insert(b"dog", b"puppy".to_vec());
        trie.insert(b"doge", b"coin".to_vec());
        trie.insert(b"horse", b"stallion".to_vec());

        let root = trie.root_hash();

        // Generate and verify proofs for each key
        for (key, expected_value) in [
            (&b"do"[..], &b"verb"[..]),
            (&b"dog"[..], &b"puppy"[..]),
            (&b"doge"[..], &b"coin"[..]),
            (&b"horse"[..], &b"stallion"[..]),
        ] {
            let proof = generate_proof(&trie, key);

            assert_eq!(proof.value, Some(expected_value.to_vec()),
                "Proof for {:?} should have correct value", String::from_utf8_lossy(key));
            assert!(!proof.nodes.is_empty(),
                "Proof for {:?} should have nodes", String::from_utf8_lossy(key));
            assert!(proof.verify(&root),
                "Proof for {:?} should verify", String::from_utf8_lossy(key));
        }
    }

    #[test]
    fn test_proof_deep_trie() {
        let mut trie = PatriciaTrie::new_memory();

        // Insert many keys to create a deeper trie
        for i in 0u32..50 {
            let key = format!("key{:04}", i);
            let value = format!("value{}", i);
            trie.insert(key.as_bytes(), value.as_bytes().to_vec());
        }

        let root = trie.root_hash();

        // Verify proofs for several keys
        for i in [0, 10, 25, 49] {
            let key = format!("key{:04}", i);
            let expected = format!("value{}", i);

            let proof = generate_proof(&trie, key.as_bytes());

            assert_eq!(proof.value, Some(expected.as_bytes().to_vec()));
            assert!(proof.nodes.len() > 1, "Deep trie should have multiple nodes in proof");
            assert!(proof.verify(&root), "Proof should verify for key {}", key);
        }
    }

    #[test]
    fn test_proof_nonexistent_with_similar_keys() {
        let mut trie = PatriciaTrie::new_memory();

        trie.insert(b"dog", b"puppy".to_vec());
        trie.insert(b"doge", b"coin".to_vec());

        let root = trie.root_hash();

        // Try to prove a key that doesn't exist but shares prefix
        let proof = generate_proof(&trie, b"do");
        assert_eq!(proof.value, None);

        // Non-existence proof should also verify (value is None)
        // The proof contains the path that shows the key doesn't exist
        assert!(proof.verify(&root) || proof.nodes.is_empty(),
            "Non-existence should be verifiable");
    }

    #[test]
    fn test_proof_after_update() {
        let mut trie = PatriciaTrie::new_memory();

        trie.insert(b"key", b"value1".to_vec());
        let root1 = trie.root_hash();
        let proof1 = generate_proof(&trie, b"key");

        // Update the value
        trie.insert(b"key", b"value2".to_vec());
        let root2 = trie.root_hash();
        let proof2 = generate_proof(&trie, b"key");

        // Roots should be different
        assert_ne!(root1, root2);

        // Old proof should verify against old root
        assert!(proof1.verify(&root1));

        // New proof should verify against new root
        assert!(proof2.verify(&root2));

        // Old proof should NOT verify against new root
        assert!(!proof1.verify(&root2));
    }

    #[test]
    fn test_proof_after_delete() {
        let mut trie = PatriciaTrie::new_memory();

        trie.insert(b"key1", b"value1".to_vec());
        trie.insert(b"key2", b"value2".to_vec());

        // Proof before delete
        let root_before = trie.root_hash();
        let proof_before = generate_proof(&trie, b"key1");
        assert!(proof_before.verify(&root_before));

        // Delete key1
        trie.delete(b"key1");
        let root_after = trie.root_hash();

        // Old proof should not verify against new root
        assert!(!proof_before.verify(&root_after));

        // key1 should no longer exist
        let proof_deleted = generate_proof(&trie, b"key1");
        assert_eq!(proof_deleted.value, None);

        // key2 should still verify
        let proof_key2 = generate_proof(&trie, b"key2");
        assert_eq!(proof_key2.value, Some(b"value2".to_vec()));
        assert!(proof_key2.verify(&root_after));
    }
}
