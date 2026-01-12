//! # LLM-VM Memory
//!
//! Page-based memory system for the LLM Virtual Machine.
//! Memory is organized as named pages that can hold any JSON data.
//! This is the working memory during execution.

use crate::error::{self, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum number of pages (prevents unbounded memory growth)
pub const MAX_PAGES: usize = 1024;

/// Approximate max tokens per page (for context window management)
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// A single memory page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPage {
    /// Page identifier
    pub id: String,
    /// Page content (any JSON value)
    pub content: serde_json::Value,
    /// Approximate size in tokens
    pub size_tokens: usize,
    /// Whether the page has been modified
    pub dirty: bool,
    /// Optional label/type for the page
    pub label: Option<String>,
    /// Creation timestamp (for LRU eviction)
    pub created_at: u64,
    /// Last access timestamp
    pub accessed_at: u64,
}

impl MemoryPage {
    /// Create a new page with content
    pub fn new(id: impl Into<String>, content: serde_json::Value) -> Self {
        let id = id.into();
        let size_tokens = estimate_tokens(&content);
        let now = current_timestamp();
        Self {
            id,
            content,
            size_tokens,
            dirty: true,
            label: None,
            created_at: now,
            accessed_at: now,
        }
    }

    /// Create an empty page
    pub fn empty(id: impl Into<String>) -> Self {
        Self::new(id, serde_json::Value::Null)
    }

    /// Update content and mark as dirty
    pub fn set_content(&mut self, content: serde_json::Value) {
        self.content = content;
        self.size_tokens = estimate_tokens(&self.content);
        self.dirty = true;
        self.accessed_at = current_timestamp();
    }

    /// Mark page as accessed
    pub fn touch(&mut self) {
        self.accessed_at = current_timestamp();
    }

    /// Mark page as clean (synced to storage)
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }
}

/// LLM-VM Memory - collection of named pages
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Memory {
    pages: HashMap<String, MemoryPage>,
    /// Total approximate tokens across all pages
    total_tokens: usize,
    /// Maximum tokens allowed
    max_tokens: usize,
}

impl Memory {
    /// Create new empty memory
    pub fn new() -> Self {
        Self {
            pages: HashMap::new(),
            total_tokens: 0,
            max_tokens: 128_000, // Default context window
        }
    }

    /// Create memory with custom max tokens
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self {
            pages: HashMap::new(),
            total_tokens: 0,
            max_tokens,
        }
    }

    /// Number of pages
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Total tokens in memory
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Check if a page exists
    pub fn has_page(&self, id: &str) -> bool {
        self.pages.contains_key(id)
    }

    /// Get a page by ID (immutable)
    pub fn get(&self, id: &str) -> Option<&MemoryPage> {
        self.pages.get(id)
    }

    /// Get a page by ID (mutable)
    pub fn get_mut(&mut self, id: &str) -> Option<&mut MemoryPage> {
        let page = self.pages.get_mut(id)?;
        page.touch();
        Some(page)
    }

    /// Load page content
    pub fn load(&mut self, id: &str) -> Result<&serde_json::Value> {
        let page = self.pages.get_mut(id).ok_or_else(|| error::page_not_found(id))?;
        page.touch();
        Ok(&page.content)
    }

    /// Store content to a page (creates if not exists)
    pub fn store(&mut self, id: impl Into<String>, content: serde_json::Value) -> Result<()> {
        let id = id.into();

        if let Some(page) = self.pages.get_mut(&id) {
            let old_tokens = page.size_tokens;
            page.set_content(content);
            self.total_tokens = self.total_tokens - old_tokens + page.size_tokens;
        } else {
            if self.pages.len() >= MAX_PAGES {
                return Err(error::page_overflow());
            }
            let page = MemoryPage::new(&id, content);
            self.total_tokens += page.size_tokens;
            self.pages.insert(id, page);
        }

        Ok(())
    }

    /// Store a pre-built page directly (used when loading from session)
    pub fn store_page(&mut self, page: MemoryPage) -> Result<()> {
        if !self.pages.contains_key(&page.id) && self.pages.len() >= MAX_PAGES {
            return Err(error::page_overflow());
        }

        let old_tokens = self.pages.get(&page.id).map(|p| p.size_tokens).unwrap_or(0);
        self.total_tokens = self.total_tokens - old_tokens + page.size_tokens;
        self.pages.insert(page.id.clone(), page);

        Ok(())
    }

    /// Allocate a new empty page
    pub fn alloc(&mut self, label: Option<String>) -> Result<String> {
        if self.pages.len() >= MAX_PAGES {
            return Err(error::page_overflow());
        }

        let id = format!("page_{}", self.pages.len());
        let mut page = MemoryPage::empty(&id);
        page.label = label;
        self.total_tokens += page.size_tokens;
        self.pages.insert(id.clone(), page);

        Ok(id)
    }

    /// Free a page
    pub fn free(&mut self, id: &str) -> Result<()> {
        let page = self.pages.remove(id).ok_or_else(|| error::page_not_found(id))?;
        self.total_tokens = self.total_tokens.saturating_sub(page.size_tokens);
        Ok(())
    }

    /// Copy content from one page to another
    pub fn copy(&mut self, src: &str, dst: &str) -> Result<()> {
        let content = self.pages.get(src)
            .ok_or_else(|| error::page_not_found(src))?
            .content
            .clone();

        self.store(dst, content)
    }

    /// Get all page IDs
    pub fn page_ids(&self) -> impl Iterator<Item = &str> {
        self.pages.keys().map(|s| s.as_str())
    }

    /// Get all dirty pages
    pub fn dirty_pages(&self) -> impl Iterator<Item = &MemoryPage> {
        self.pages.values().filter(|p| p.dirty)
    }

    /// Clear all pages
    pub fn clear(&mut self) {
        self.pages.clear();
        self.total_tokens = 0;
    }

    /// Get pages sorted by access time (least recently used first)
    pub fn pages_by_lru(&self) -> Vec<&MemoryPage> {
        let mut pages: Vec<_> = self.pages.values().collect();
        pages.sort_by_key(|p| p.accessed_at);
        pages
    }

    /// Evict least recently used pages until under token limit
    pub fn evict_to_limit(&mut self, target_tokens: usize) -> Vec<String> {
        let mut evicted = Vec::new();

        while self.total_tokens > target_tokens && !self.pages.is_empty() {
            // Find LRU page
            let lru_id = self.pages
                .values()
                .min_by_key(|p| p.accessed_at)
                .map(|p| p.id.clone());

            if let Some(id) = lru_id {
                if let Ok(()) = self.free(&id) {
                    evicted.push(id);
                }
            } else {
                break;
            }
        }

        evicted
    }
}

/// Estimate token count for a JSON value (rough approximation)
fn estimate_tokens(value: &serde_json::Value) -> usize {
    let s = value.to_string();
    // Rough estimate: 4 chars per token
    s.len() / 4 + 1
}

/// Get current timestamp (mock for now)
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_store_load() {
        let mut mem = Memory::new();

        mem.store("test", json!({"hello": "world"})).unwrap();

        let content = mem.load("test").unwrap();
        assert_eq!(content, &json!({"hello": "world"}));
    }

    #[test]
    fn test_alloc_free() {
        let mut mem = Memory::new();

        let id = mem.alloc(Some("scratch".to_string())).unwrap();
        assert!(mem.has_page(&id));

        mem.free(&id).unwrap();
        assert!(!mem.has_page(&id));
    }

    #[test]
    fn test_copy() {
        let mut mem = Memory::new();

        mem.store("src", json!([1, 2, 3])).unwrap();
        mem.copy("src", "dst").unwrap();

        let content = mem.load("dst").unwrap();
        assert_eq!(content, &json!([1, 2, 3]));
    }

    #[test]
    fn test_page_not_found() {
        use crate::error::ErrorKind;
        let mut mem = Memory::new();

        let result = mem.load("nonexistent");
        assert!(result.is_err_and(|e| e.kind() == ErrorKind::PageNotFound));
    }

    #[test]
    fn test_dirty_tracking() {
        let mut mem = Memory::new();

        mem.store("page1", json!("initial")).unwrap();
        assert!(mem.get("page1").unwrap().dirty);

        mem.get_mut("page1").unwrap().mark_clean();
        assert!(!mem.get("page1").unwrap().dirty);

        mem.store("page1", json!("updated")).unwrap();
        assert!(mem.get("page1").unwrap().dirty);
    }

    #[test]
    fn test_total_tokens() {
        let mut mem = Memory::new();

        assert_eq!(mem.total_tokens(), 0);

        mem.store("page1", json!("hello world")).unwrap();
        assert!(mem.total_tokens() > 0);

        let tokens_before = mem.total_tokens();
        mem.free("page1").unwrap();
        assert!(mem.total_tokens() < tokens_before);
    }
}
