//! # Merkle Patricia Trie Demo
//!
//! Demonstration of the MPT implementation

use mpt_trie::{PatriciaTrie, Proof};
use eth_primitives::keccak256;

fn main() {
    println!("🌳 Merkle Patricia Trie Demo\n");

    // =========================================
    // Test 1: Empty Trie
    // =========================================
    println!("=== Test 1: Empty Trie ===");
    let trie = PatriciaTrie::new_memory();

    println!("Is empty: {}", trie.is_empty());
    println!("Root hash: 0x{}", hex::encode(trie.root_hash().as_bytes()));
    println!("(This is the well-known empty trie root)");
    println!();

    // =========================================
    // Test 2: Single Insert
    // =========================================
    println!("=== Test 2: Single Insert ===");
    let mut trie = PatriciaTrie::new_memory();

    trie.insert(b"hello", b"world".to_vec());

    println!("Inserted: 'hello' -> 'world'");
    println!("Root hash: 0x{}", hex::encode(trie.root_hash().as_bytes()));

    let value = trie.get(b"hello");
    println!("Get 'hello': {:?}", value.map(|v| String::from_utf8_lossy(&v).to_string()));

    let missing = trie.get(b"missing");
    println!("Get 'missing': {:?}", missing);
    println!();

    // =========================================
    // Test 3: Multiple Keys with Shared Prefix
    // =========================================
    println!("=== Test 3: Shared Prefix Keys ===");
    let mut trie = PatriciaTrie::new_memory();

    trie.insert(b"do", b"verb".to_vec());
    trie.insert(b"dog", b"puppy".to_vec());
    trie.insert(b"doge", b"coin".to_vec());
    trie.insert(b"horse", b"stallion".to_vec());

    println!("Inserted:");
    println!("  'do' -> 'verb'");
    println!("  'dog' -> 'puppy'");
    println!("  'doge' -> 'coin'");
    println!("  'horse' -> 'stallion'");
    println!();

    println!("Root hash: 0x{}", hex::encode(trie.root_hash().as_bytes()));
    println!();

    println!("Retrieving values:");
    for key in &["do", "dog", "doge", "horse", "cat"] {
        let value = trie.get(key.as_bytes());
        match value {
            Some(v) => println!("  '{}' -> '{}'", key, String::from_utf8_lossy(&v)),
            None => println!("  '{}' -> NOT FOUND", key),
        }
    }
    println!();

    // =========================================
    // Test 4: Updates
    // =========================================
    println!("=== Test 4: Updates ===");
    let mut trie = PatriciaTrie::new_memory();

    trie.insert(b"key", b"value1".to_vec());
    let root1 = trie.root_hash();
    println!("After insert 'key' -> 'value1':");
    println!("  Root: 0x{}...", &hex::encode(root1.as_bytes())[..16]);

    trie.insert(b"key", b"value2".to_vec());
    let root2 = trie.root_hash();
    println!("After update 'key' -> 'value2':");
    println!("  Root: 0x{}...", &hex::encode(root2.as_bytes())[..16]);

    println!("  Roots different: {}", root1 != root2);
    println!("  Current value: {:?}", trie.get(b"key").map(|v| String::from_utf8_lossy(&v).to_string()));
    println!();

    // =========================================
    // Test 5: Deletions
    // =========================================
    println!("=== Test 5: Deletions ===");
    let mut trie = PatriciaTrie::new_memory();

    trie.insert(b"key1", b"value1".to_vec());
    trie.insert(b"key2", b"value2".to_vec());
    trie.insert(b"key3", b"value3".to_vec());

    println!("Inserted 3 keys");
    println!("Root: 0x{}...", &hex::encode(trie.root_hash().as_bytes())[..16]);

    let deleted = trie.delete(b"key2");
    println!("Deleted 'key2': {}", deleted);
    println!("Root: 0x{}...", &hex::encode(trie.root_hash().as_bytes())[..16]);

    println!("Remaining values:");
    for key in &["key1", "key2", "key3"] {
        let value = trie.get(key.as_bytes());
        match value {
            Some(v) => println!("  '{}' -> '{}'", key, String::from_utf8_lossy(&v)),
            None => println!("  '{}' -> DELETED", key),
        }
    }
    println!();

    // =========================================
    // Test 6: Deterministic Roots
    // =========================================
    println!("=== Test 6: Deterministic Roots ===");

    // Insert in order A
    let mut trie_a = PatriciaTrie::new_memory();
    trie_a.insert(b"apple", b"red".to_vec());
    trie_a.insert(b"banana", b"yellow".to_vec());
    trie_a.insert(b"cherry", b"red".to_vec());

    // Insert in order B
    let mut trie_b = PatriciaTrie::new_memory();
    trie_b.insert(b"cherry", b"red".to_vec());
    trie_b.insert(b"apple", b"red".to_vec());
    trie_b.insert(b"banana", b"yellow".to_vec());

    println!("Trie A (inserted: apple, banana, cherry):");
    println!("  Root: 0x{}", hex::encode(trie_a.root_hash().as_bytes()));

    println!("Trie B (inserted: cherry, apple, banana):");
    println!("  Root: 0x{}", hex::encode(trie_b.root_hash().as_bytes()));

    println!("Same content = same root: {}", trie_a.root_hash() == trie_b.root_hash());
    println!();

    // =========================================
    // Test 7: Many Keys
    // =========================================
    println!("=== Test 7: Performance with Many Keys ===");
    let mut trie = PatriciaTrie::new_memory();

    let start = std::time::Instant::now();
    for i in 0u32..1000 {
        let key = format!("key_{:04}", i);
        let value = format!("value_{}", i);
        trie.insert(key.as_bytes(), value.as_bytes().to_vec());
    }
    let insert_time = start.elapsed();

    println!("Inserted 1000 keys in {:?}", insert_time);
    println!("Root: 0x{}...", &hex::encode(trie.root_hash().as_bytes())[..16]);

    let start = std::time::Instant::now();
    let mut found = 0;
    for i in 0u32..1000 {
        let key = format!("key_{:04}", i);
        if trie.get(key.as_bytes()).is_some() {
            found += 1;
        }
    }
    let lookup_time = start.elapsed();

    println!("Looked up 1000 keys in {:?}", lookup_time);
    println!("Found: {}/1000", found);
    println!();

    // =========================================
    // Test 8: Ethereum Account State (Conceptual)
    // =========================================
    println!("=== Test 8: Ethereum State Trie (Conceptual) ===");

    let mut state_trie = PatriciaTrie::new_memory();

    // In Ethereum, keys are keccak256(address)
    // Values are RLP-encoded account state

    let alice = keccak256(b"alice_address");
    let bob = keccak256(b"bob_address");

    // Simplified account data
    let alice_account = b"nonce:1,balance:1000000000000000000".to_vec();
    let bob_account = b"nonce:0,balance:500000000000000000".to_vec();

    state_trie.insert(alice.as_bytes(), alice_account);
    state_trie.insert(bob.as_bytes(), bob_account);

    println!("State root (with 2 accounts): 0x{}...",
             &hex::encode(state_trie.root_hash().as_bytes())[..16]);

    // Update Alice's balance
    let alice_new = b"nonce:2,balance:800000000000000000".to_vec();
    state_trie.insert(alice.as_bytes(), alice_new);

    println!("State root (after Alice transfer): 0x{}...",
             &hex::encode(state_trie.root_hash().as_bytes())[..16]);

    println!();
    println!("✅ All MPT demos completed!");
}
