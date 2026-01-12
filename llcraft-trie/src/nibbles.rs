//! # Nibbles
//!
//! Keys in the MPT are represented as nibbles (half-bytes / 4 bits).
//! This allows branching on 16 possible values at each node.

use std::fmt;

/// A sequence of nibbles (4-bit values)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Nibbles {
    /// The nibble data
    data: Vec<u8>,
}

impl Nibbles {
    /// Create empty nibbles
    pub fn new() -> Self {
        Nibbles { data: Vec::new() }
    }

    /// Create from bytes (each byte becomes 2 nibbles)
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut data = Vec::with_capacity(bytes.len() * 2);
        for byte in bytes {
            data.push(byte >> 4);      // High nibble
            data.push(byte & 0x0f);    // Low nibble
        }
        Nibbles { data }
    }

    /// Create from raw nibbles
    pub fn from_raw(nibbles: Vec<u8>) -> Self {
        debug_assert!(nibbles.iter().all(|n| *n < 16));
        Nibbles { data: nibbles }
    }

    /// Create from hex prefix encoded data
    /// Encoding: first nibble is flags, then key nibbles
    /// Flags: bit 0 = odd length, bit 1 = leaf node
    pub fn from_hex_prefix(encoded: &[u8], is_leaf: &mut bool) -> Self {
        if encoded.is_empty() {
            *is_leaf = false;
            return Nibbles::new();
        }

        let first = encoded[0];
        let prefix = first >> 4;
        *is_leaf = prefix >= 2;
        let odd = prefix & 1 == 1;

        let mut nibbles = Vec::new();

        if odd {
            // First byte contains a nibble
            nibbles.push(first & 0x0f);
        }

        // Rest of the bytes
        for byte in &encoded[1..] {
            nibbles.push(byte >> 4);
            nibbles.push(byte & 0x0f);
        }

        Nibbles { data: nibbles }
    }

    /// Encode to hex prefix format
    pub fn to_hex_prefix(&self, is_leaf: bool) -> Vec<u8> {
        let prefix = if is_leaf { 2 } else { 0 };
        let odd = self.len() % 2 == 1;

        let mut encoded = Vec::new();

        if odd {
            // First byte: prefix | first nibble
            encoded.push((prefix + 1) << 4 | self.data[0]);
            // Pair remaining nibbles
            for chunk in self.data[1..].chunks(2) {
                if chunk.len() == 2 {
                    encoded.push(chunk[0] << 4 | chunk[1]);
                }
            }
        } else {
            // First byte: prefix << 4
            encoded.push(prefix << 4);
            // Pair all nibbles
            for chunk in self.data.chunks(2) {
                if chunk.len() == 2 {
                    encoded.push(chunk[0] << 4 | chunk[1]);
                }
            }
        }

        encoded
    }

    /// Convert back to bytes (each 2 nibbles -> 1 byte)
    pub fn to_bytes(&self) -> Vec<u8> {
        if self.data.len() % 2 != 0 {
            panic!("Cannot convert odd number of nibbles to bytes");
        }

        self.data.chunks(2)
            .map(|chunk| chunk[0] << 4 | chunk[1])
            .collect()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get nibble at index
    pub fn get(&self, index: usize) -> Option<u8> {
        self.data.get(index).copied()
    }

    /// Get first nibble
    pub fn first(&self) -> Option<u8> {
        self.data.first().copied()
    }

    /// Get slice from index
    pub fn slice(&self, start: usize) -> Self {
        Nibbles {
            data: self.data[start..].to_vec()
        }
    }

    /// Get slice range
    pub fn slice_range(&self, start: usize, end: usize) -> Self {
        Nibbles {
            data: self.data[start..end].to_vec()
        }
    }

    /// Find common prefix length with another nibble sequence
    pub fn common_prefix_len(&self, other: &Nibbles) -> usize {
        self.data.iter()
            .zip(other.data.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Get common prefix
    pub fn common_prefix(&self, other: &Nibbles) -> Self {
        let len = self.common_prefix_len(other);
        self.slice_range(0, len)
    }

    /// Append another nibble sequence
    pub fn extend(&mut self, other: &Nibbles) {
        self.data.extend_from_slice(&other.data);
    }

    /// Push a single nibble
    pub fn push(&mut self, nibble: u8) {
        debug_assert!(nibble < 16);
        self.data.push(nibble);
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl Default for Nibbles {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Nibbles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Nibbles(")?;
        for n in &self.data {
            write!(f, "{:x}", n)?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for Nibbles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for n in &self.data {
            write!(f, "{:x}", n)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_bytes() {
        let nibbles = Nibbles::from_bytes(&[0xab, 0xcd]);
        assert_eq!(nibbles.len(), 4);
        assert_eq!(nibbles.get(0), Some(0xa));
        assert_eq!(nibbles.get(1), Some(0xb));
        assert_eq!(nibbles.get(2), Some(0xc));
        assert_eq!(nibbles.get(3), Some(0xd));
    }

    #[test]
    fn test_to_bytes() {
        let nibbles = Nibbles::from_raw(vec![0xa, 0xb, 0xc, 0xd]);
        let bytes = nibbles.to_bytes();
        assert_eq!(bytes, vec![0xab, 0xcd]);
    }

    #[test]
    fn test_hex_prefix_leaf_odd() {
        let nibbles = Nibbles::from_raw(vec![1, 2, 3]);
        let encoded = nibbles.to_hex_prefix(true);
        // Odd leaf: prefix = 3, first byte = 0x31
        assert_eq!(encoded, vec![0x31, 0x23]);

        let mut is_leaf = false;
        let decoded = Nibbles::from_hex_prefix(&encoded, &mut is_leaf);
        assert!(is_leaf);
        assert_eq!(decoded, nibbles);
    }

    #[test]
    fn test_hex_prefix_leaf_even() {
        let nibbles = Nibbles::from_raw(vec![1, 2, 3, 4]);
        let encoded = nibbles.to_hex_prefix(true);
        // Even leaf: prefix = 2, first byte = 0x20
        assert_eq!(encoded, vec![0x20, 0x12, 0x34]);

        let mut is_leaf = false;
        let decoded = Nibbles::from_hex_prefix(&encoded, &mut is_leaf);
        assert!(is_leaf);
        assert_eq!(decoded, nibbles);
    }

    #[test]
    fn test_hex_prefix_extension_odd() {
        let nibbles = Nibbles::from_raw(vec![1, 2, 3]);
        let encoded = nibbles.to_hex_prefix(false);
        // Odd extension: prefix = 1, first byte = 0x11
        assert_eq!(encoded, vec![0x11, 0x23]);

        let mut is_leaf = false;
        let decoded = Nibbles::from_hex_prefix(&encoded, &mut is_leaf);
        assert!(!is_leaf);
        assert_eq!(decoded, nibbles);
    }

    #[test]
    fn test_hex_prefix_extension_even() {
        let nibbles = Nibbles::from_raw(vec![1, 2, 3, 4]);
        let encoded = nibbles.to_hex_prefix(false);
        // Even extension: prefix = 0, first byte = 0x00
        assert_eq!(encoded, vec![0x00, 0x12, 0x34]);

        let mut is_leaf = false;
        let decoded = Nibbles::from_hex_prefix(&encoded, &mut is_leaf);
        assert!(!is_leaf);
        assert_eq!(decoded, nibbles);
    }

    #[test]
    fn test_common_prefix() {
        let a = Nibbles::from_raw(vec![1, 2, 3, 4, 5]);
        let b = Nibbles::from_raw(vec![1, 2, 3, 6, 7]);

        assert_eq!(a.common_prefix_len(&b), 3);
        assert_eq!(a.common_prefix(&b), Nibbles::from_raw(vec![1, 2, 3]));
    }

    #[test]
    fn test_slice() {
        let nibbles = Nibbles::from_raw(vec![1, 2, 3, 4, 5]);

        let sliced = nibbles.slice(2);
        assert_eq!(sliced, Nibbles::from_raw(vec![3, 4, 5]));

        let range = nibbles.slice_range(1, 4);
        assert_eq!(range, Nibbles::from_raw(vec![2, 3, 4]));
    }
}
