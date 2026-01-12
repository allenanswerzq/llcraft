//! # LLM-VM Stack
//!
//! A LIFO stack for working values during LLM-VM execution.
//! Unlike the EVM's U256 stack, this holds arbitrary JSON values.

use crate::error::{self, Result};
use serde::{Deserialize, Serialize};

/// Maximum stack depth (prevents runaway recursion)
pub const MAX_STACK_SIZE: usize = 256;

/// LLM-VM Stack - holds JSON values
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stack {
    data: Vec<serde_json::Value>,
}

impl Stack {
    /// Create a new empty stack
    pub fn new() -> Self {
        Stack {
            data: Vec::with_capacity(32),
        }
    }

    /// Current stack size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if stack is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Push a value onto the stack
    pub fn push(&mut self, value: serde_json::Value) -> Result<()> {
        if self.data.len() >= MAX_STACK_SIZE {
            return Err(error::stack_overflow());
        }
        self.data.push(value);
        Ok(())
    }

    /// Pop a value from the stack
    pub fn pop(&mut self) -> Result<serde_json::Value> {
        self.data.pop().ok_or_else(error::stack_underflow)
    }

    /// Peek at the top value without removing it
    pub fn peek(&self) -> Result<&serde_json::Value> {
        self.data.last().ok_or_else(error::stack_underflow)
    }

    /// Peek at a value at a specific depth (0 = top)
    pub fn peek_at(&self, depth: usize) -> Result<&serde_json::Value> {
        if depth >= self.data.len() {
            return Err(error::stack_underflow());
        }
        Ok(&self.data[self.data.len() - 1 - depth])
    }

    /// Set a value at a specific depth (0 = top)
    pub fn set_at(&mut self, depth: usize, value: serde_json::Value) -> Result<()> {
        if depth >= self.data.len() {
            return Err(error::stack_underflow());
        }
        let idx = self.data.len() - 1 - depth;
        self.data[idx] = value;
        Ok(())
    }

    /// Duplicate the top value
    pub fn dup(&mut self) -> Result<()> {
        let value = self.peek()?.clone();
        self.push(value)
    }

    /// Duplicate the value at depth N (0 = top)
    pub fn dup_n(&mut self, n: usize) -> Result<()> {
        let value = self.peek_at(n)?.clone();
        self.push(value)
    }

    /// Swap top two values
    pub fn swap(&mut self) -> Result<()> {
        if self.data.len() < 2 {
            return Err(error::stack_underflow());
        }
        let len = self.data.len();
        self.data.swap(len - 1, len - 2);
        Ok(())
    }

    /// Swap top with value at depth N (1-indexed)
    pub fn swap_n(&mut self, n: usize) -> Result<()> {
        if n == 0 || self.data.len() <= n {
            return Err(error::stack_underflow());
        }
        let top_idx = self.data.len() - 1;
        let swap_idx = self.data.len() - 1 - n;
        self.data.swap(top_idx, swap_idx);
        Ok(())
    }

    /// Rotate top N values (top moves to Nth position)
    /// e.g., rot(3) on [a, b, c] -> [c, a, b]
    pub fn rot(&mut self, n: usize) -> Result<()> {
        if n > self.data.len() || n == 0 {
            return Err(error::stack_underflow());
        }
        let start = self.data.len() - n;
        self.data[start..].rotate_right(1);
        Ok(())
    }

    /// Drop top N values
    pub fn drop_n(&mut self, n: usize) -> Result<()> {
        if n > self.data.len() {
            return Err(error::stack_underflow());
        }
        self.data.truncate(self.data.len() - n);
        Ok(())
    }

    /// Clear the entire stack
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get iterator over values (bottom to top)
    pub fn iter(&self) -> impl Iterator<Item = &serde_json::Value> {
        self.data.iter()
    }

    /// Get iterator over values (top to bottom)
    pub fn iter_top_down(&self) -> impl Iterator<Item = &serde_json::Value> {
        self.data.iter().rev()
    }

    /// Get all values as a slice (bottom to top)
    pub fn as_slice(&self) -> &[serde_json::Value] {
        &self.data
    }

    /// Convert entire stack to JSON array
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::Value::Array(self.data.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_push_pop() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!("hello")).unwrap();
        stack.push(json!({"key": "value"})).unwrap();

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.pop().unwrap(), json!({"key": "value"}));
        assert_eq!(stack.pop().unwrap(), json!("hello"));
        assert_eq!(stack.pop().unwrap(), json!(1));
        assert!(stack.is_empty());
    }

    #[test]
    fn test_peek() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!(2)).unwrap();
        stack.push(json!(3)).unwrap();

        assert_eq!(stack.peek().unwrap(), &json!(3));
        assert_eq!(stack.peek_at(0).unwrap(), &json!(3));
        assert_eq!(stack.peek_at(1).unwrap(), &json!(2));
        assert_eq!(stack.peek_at(2).unwrap(), &json!(1));
    }

    #[test]
    fn test_dup() {
        let mut stack = Stack::new();
        stack.push(json!(42)).unwrap();
        stack.dup().unwrap();

        assert_eq!(stack.len(), 2);
        assert_eq!(stack.pop().unwrap(), json!(42));
        assert_eq!(stack.pop().unwrap(), json!(42));
    }

    #[test]
    fn test_dup_n() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!(2)).unwrap();
        stack.push(json!(3)).unwrap();
        stack.dup_n(2).unwrap(); // Dup the value at depth 2 (which is 1)

        assert_eq!(stack.len(), 4);
        assert_eq!(stack.pop().unwrap(), json!(1));
    }

    #[test]
    fn test_swap() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!(2)).unwrap();
        stack.swap().unwrap();

        assert_eq!(stack.pop().unwrap(), json!(1));
        assert_eq!(stack.pop().unwrap(), json!(2));
    }

    #[test]
    fn test_swap_n() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!(2)).unwrap();
        stack.push(json!(3)).unwrap();
        stack.swap_n(2).unwrap(); // Swap top (3) with value at depth 2 (1)

        assert_eq!(stack.pop().unwrap(), json!(1));
        assert_eq!(stack.pop().unwrap(), json!(2));
        assert_eq!(stack.pop().unwrap(), json!(3));
    }

    #[test]
    fn test_rot() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!(2)).unwrap();
        stack.push(json!(3)).unwrap();
        stack.rot(3).unwrap(); // [1,2,3] -> [3,1,2]

        assert_eq!(stack.pop().unwrap(), json!(2));
        assert_eq!(stack.pop().unwrap(), json!(1));
        assert_eq!(stack.pop().unwrap(), json!(3));
    }

    #[test]
    fn test_drop_n() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!(2)).unwrap();
        stack.push(json!(3)).unwrap();
        stack.drop_n(2).unwrap();

        assert_eq!(stack.len(), 1);
        assert_eq!(stack.pop().unwrap(), json!(1));
    }

    #[test]
    fn test_clear() {
        let mut stack = Stack::new();
        stack.push(json!(1)).unwrap();
        stack.push(json!(2)).unwrap();
        stack.clear();

        assert!(stack.is_empty());
    }

    #[test]
    fn test_underflow() {
        use crate::error::ErrorKind;
        let mut stack = Stack::new();
        assert!(stack.pop().is_err_and(|e| e.kind() == ErrorKind::StackUnderflow));
        assert!(stack.peek().is_err_and(|e| e.kind() == ErrorKind::StackUnderflow));
        assert!(stack.swap().is_err_and(|e| e.kind() == ErrorKind::StackUnderflow));
    }

    #[test]
    fn test_overflow() {
        use crate::error::ErrorKind;
        let mut stack = Stack::new();
        for i in 0..MAX_STACK_SIZE {
            stack.push(json!(i)).unwrap();
        }
        assert!(stack.push(json!(999)).is_err_and(|e| e.kind() == ErrorKind::StackOverflow));
    }
}
