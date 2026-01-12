//! # Session Management for LLcraft VM
//!
//! This module provides persistent session state that allows the VM to:
//! - Store and retrieve memory pages across invocations
//! - Maintain a lightweight page index for efficient context management
//! - Summarize and compress execution history
//! - Enable on-demand page loading to minimize context window usage
//!
//! ## Design Philosophy
//!
//! Instead of loading all history into every LLM prompt, sessions maintain:
//! - A **page index**: lightweight metadata (id, summary, tokens, timestamps)
//! - **Stored pages**: full content persisted to disk, loaded on-demand
//! - **Trace summary**: compressed execution history
//!
//! The LLM sees the page index and can request specific pages via LOAD_PAGE.

use crate::error::{self, Result};
use crate::memory::{Memory, MemoryPage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ═══════════════════════════════════════════════════════════════════════════════
// Progress Log (for learnings across iterations)
// ═══════════════════════════════════════════════════════════════════════════════

/// A single progress entry (append-only log)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEntry {
    /// When this entry was added
    pub timestamp: u64,
    /// Which program this relates to (if any)
    pub program_id: Option<String>,
    /// What was done
    pub summary: String,
    /// Learnings for future iterations
    pub learnings: Vec<String>,
    /// Files that were changed
    pub files_changed: Vec<String>,
}

/// Append-only progress log for tracking learnings and patterns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProgressLog {
    /// Progress entries (oldest first)
    pub entries: Vec<ProgressEntry>,
    /// Reusable patterns discovered (consolidated from learnings)
    pub patterns: Vec<String>,
}

impl ProgressLog {
    /// Add a progress entry
    pub fn add_entry(&mut self, program_id: Option<&str>, summary: &str, learnings: Vec<String>, files: Vec<String>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.entries.push(ProgressEntry {
            timestamp: now,
            program_id: program_id.map(|s| s.to_string()),
            summary: summary.to_string(),
            learnings,
            files_changed: files,
        });
    }

    /// Add a reusable pattern
    pub fn add_pattern(&mut self, pattern: &str) {
        if !self.patterns.contains(&pattern.to_string()) {
            self.patterns.push(pattern.to_string());
        }
    }

    /// Get all patterns as a formatted string
    pub fn patterns_summary(&self) -> String {
        if self.patterns.is_empty() {
            return String::from("No patterns discovered yet.");
        }
        self.patterns.iter()
            .map(|p| format!("- {}", p))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get recent learnings
    pub fn recent_learnings(&self, max_entries: usize) -> Vec<&str> {
        self.entries.iter()
            .rev()
            .take(max_entries)
            .flat_map(|e| e.learnings.iter().map(|s| s.as_str()))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Page Index and Session Structures
// ═══════════════════════════════════════════════════════════════════════════════

/// Lightweight page metadata for the index (doesn't include content)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageIndex {
    /// Page identifier
    pub id: String,
    /// Brief summary of the page content (for LLM context)
    pub summary: String,
    /// Approximate token count
    pub tokens: usize,
    /// Content type hint (e.g., "file", "analysis", "result")
    pub content_type: Option<String>,
    /// When the page was created
    pub created_at: u64,
    /// When the page was last accessed
    pub accessed_at: u64,
    /// Whether the page is currently loaded in active memory
    pub loaded: bool,
}

/// Compressed execution trace entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    /// Step number
    pub step: usize,
    /// Opcode name
    pub opcode: String,
    /// Brief result description
    pub result: String,
    /// Whether this step had an error
    pub had_error: bool,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session identifier
    pub id: String,
    /// Original task description
    pub task: String,
    /// When the session was created
    pub created_at: u64,
    /// When the session was last updated
    pub updated_at: u64,
    /// Total steps executed across all invocations
    pub total_steps: usize,
    /// Number of LLM calls made
    pub llm_calls: usize,
    /// Current status
    pub status: SessionStatus,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// Session is active and can be continued
    Active,
    /// Session completed successfully
    Completed,
    /// Session failed with an error
    Failed,
    /// Session was abandoned/cancelled
    Abandoned,
}

/// A persistent session containing state across invocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session metadata
    pub metadata: SessionMetadata,
    /// Page index (lightweight - no content)
    pub page_index: HashMap<String, PageIndex>,
    /// Compressed execution trace
    pub trace_summary: Vec<TraceSummary>,
    /// Pages currently loaded in active memory
    #[serde(skip)]
    pub active_memory: Memory,
}

impl Session {
    /// Create a new session
    pub fn new(id: impl Into<String>, task: impl Into<String>) -> Self {
        let now = current_timestamp();
        Self {
            metadata: SessionMetadata {
                id: id.into(),
                task: task.into(),
                created_at: now,
                updated_at: now,
                total_steps: 0,
                llm_calls: 0,
                status: SessionStatus::Active,
            },
            page_index: HashMap::new(),
            trace_summary: Vec::new(),
            active_memory: Memory::new(),
        }
    }

    /// Generate a unique session ID
    pub fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        format!("session_{:x}", ts)
    }

    /// Add or update a page in the index
    pub fn index_page(&mut self, page: &MemoryPage, summary: Option<String>) {
        let summary = summary.unwrap_or_else(|| self.auto_summarize(page));

        self.page_index.insert(page.id.clone(), PageIndex {
            id: page.id.clone(),
            summary,
            tokens: page.size_tokens,
            content_type: page.label.clone(),
            created_at: page.created_at,
            accessed_at: page.accessed_at,
            loaded: true,
        });
    }

    /// Auto-generate a summary for a page (first ~100 chars or structure hint)
    fn auto_summarize(&self, page: &MemoryPage) -> String {
        match &page.content {
            serde_json::Value::String(s) => {
                let preview: String = s.chars().take(100).collect();
                if s.len() > 100 {
                    format!("{}...", preview)
                } else {
                    preview
                }
            }
            serde_json::Value::Object(obj) => {
                let keys: Vec<_> = obj.keys().take(5).cloned().collect();
                format!("Object with keys: {}", keys.join(", "))
            }
            serde_json::Value::Array(arr) => {
                format!("Array with {} items", arr.len())
            }
            other => format!("{:?}", other),
        }
    }

    /// Mark a page as loaded/unloaded in the index
    pub fn set_page_loaded(&mut self, page_id: &str, loaded: bool) {
        if let Some(idx) = self.page_index.get_mut(page_id) {
            idx.loaded = loaded;
            idx.accessed_at = current_timestamp();
        }
    }

    /// Get the page index as JSON for LLM context
    pub fn get_index_json(&self) -> serde_json::Value {
        let entries: Vec<_> = self.page_index.values()
            .map(|idx| serde_json::json!({
                "id": idx.id,
                "summary": idx.summary,
                "tokens": idx.tokens,
                "type": idx.content_type,
                "loaded": idx.loaded,
            }))
            .collect();
        serde_json::json!(entries)
    }

    /// Get list of loaded page IDs
    pub fn loaded_page_ids(&self) -> Vec<String> {
        self.page_index.values()
            .filter(|idx| idx.loaded)
            .map(|idx| idx.id.clone())
            .collect()
    }

    /// Add a trace entry (compressed)
    pub fn add_trace(&mut self, step: usize, opcode: &str, result: &str, had_error: bool) {
        // Keep only last N entries to avoid unbounded growth
        const MAX_TRACE_ENTRIES: usize = 50;

        self.trace_summary.push(TraceSummary {
            step,
            opcode: opcode.to_string(),
            result: result.chars().take(100).collect(),
            had_error,
        });

        // Trim old entries
        if self.trace_summary.len() > MAX_TRACE_ENTRIES {
            let to_remove = self.trace_summary.len() - MAX_TRACE_ENTRIES;
            self.trace_summary.drain(0..to_remove);
        }
    }

    /// Get trace summary as formatted string for LLM
    pub fn get_trace_summary(&self) -> String {
        self.trace_summary.iter()
            .map(|t| {
                let error_marker = if t.had_error { " ⚠️" } else { "" };
                format!("{}. {} → {}{}", t.step, t.opcode, t.result, error_marker)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Update metadata timestamps
    pub fn touch(&mut self) {
        self.metadata.updated_at = current_timestamp();
    }

    /// Increment step counter
    pub fn increment_steps(&mut self) {
        self.metadata.total_steps += 1;
        self.touch();
    }

    /// Increment LLM call counter
    pub fn increment_llm_calls(&mut self) {
        self.metadata.llm_calls += 1;
        self.touch();
    }
}

// =============================================================================
// Session Backend Trait
// =============================================================================

/// Trait for session storage backends
///
/// Implement this trait to add new storage backends (filesystem, AgentFS, SQLite, etc.)
pub trait SessionBackend: Send + Sync {
    /// Create a new session and persist it
    fn create_session(&self, task: &str) -> Result<Session>;

    /// Save session metadata and index (not page contents)
    fn save_session(&self, session: &Session) -> Result<()>;

    /// Load a session by ID
    fn load_session(&self, session_id: &str) -> Result<Session>;

    /// Save a specific page
    fn save_page(&self, session_id: &str, page: &MemoryPage) -> Result<()>;

    /// Load a specific page
    fn load_page(&self, session_id: &str, page_id: &str) -> Result<MemoryPage>;

    /// List all session IDs
    fn list_sessions(&self) -> Result<Vec<String>>;

    /// Delete a session and all its pages
    fn delete_session(&self, session_id: &str) -> Result<()>;

    /// Get session metadata without loading full session
    fn get_session_info(&self, session_id: &str) -> Result<SessionMetadata> {
        let session = self.load_session(session_id)?;
        Ok(session.metadata)
    }

    /// Check if a session exists
    fn session_exists(&self, session_id: &str) -> bool {
        self.load_session(session_id).is_ok()
    }

    /// Get backend name for debugging
    fn backend_name(&self) -> &'static str;
}

// =============================================================================
// File-based Backend (JSON files)
// =============================================================================

/// File-based session storage using JSON files
///
/// Structure:
/// ```text
/// {base_path}/
///   {session_id}/
///     session.json     # Session metadata, page index, trace
///     pages/
///       {page_id}.json # Individual page content
/// ```
pub struct FileBackend {
    base_path: PathBuf,
}

impl FileBackend {
    /// Create a new file backend
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)
            .map_err(|e| error::io_error(format!("Failed to create session directory: {}", e)))?;
        Ok(Self { base_path })
    }

    fn session_dir(&self, session_id: &str) -> PathBuf {
        self.base_path.join(session_id)
    }

    fn metadata_path(&self, session_id: &str) -> PathBuf {
        self.session_dir(session_id).join("session.json")
    }

    fn page_path(&self, session_id: &str, page_id: &str) -> PathBuf {
        let safe_id = page_id.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        self.session_dir(session_id).join("pages").join(format!("{}.json", safe_id))
    }
}

impl SessionBackend for FileBackend {
    fn create_session(&self, task: &str) -> Result<Session> {
        let session = Session::new(Session::generate_id(), task);
        self.save_session(&session)?;
        Ok(session)
    }

    fn save_session(&self, session: &Session) -> Result<()> {
        let session_dir = self.session_dir(&session.metadata.id);
        let pages_dir = session_dir.join("pages");

        std::fs::create_dir_all(&pages_dir)
            .map_err(|e| error::io_error(format!("Failed to create session dir: {}", e)))?;

        let metadata_path = self.metadata_path(&session.metadata.id);
        let json = serde_json::to_string_pretty(session)
            .map_err(|e| error::serialization_error(e.to_string()))?;
        std::fs::write(&metadata_path, json)
            .map_err(|e| error::io_error(format!("Failed to write session: {}", e)))?;

        Ok(())
    }

    fn load_session(&self, session_id: &str) -> Result<Session> {
        let metadata_path = self.metadata_path(session_id);

        let json = std::fs::read_to_string(&metadata_path)
            .map_err(|e| error::storage_not_found(format!("Session {}: {}", session_id, e)))?;

        let session: Session = serde_json::from_str(&json)
            .map_err(|e| error::parse_error(format!("Failed to parse session: {}", e)))?;

        Ok(session)
    }

    fn save_page(&self, session_id: &str, page: &MemoryPage) -> Result<()> {
        let page_path = self.page_path(session_id, &page.id);

        if let Some(parent) = page_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| error::io_error(format!("Failed to create pages dir: {}", e)))?;
        }

        let json = serde_json::to_string_pretty(page)
            .map_err(|e| error::serialization_error(e.to_string()))?;
        std::fs::write(&page_path, json)
            .map_err(|e| error::io_error(format!("Failed to write page {}: {}", page.id, e)))?;

        Ok(())
    }

    fn load_page(&self, session_id: &str, page_id: &str) -> Result<MemoryPage> {
        let page_path = self.page_path(session_id, page_id);

        let json = std::fs::read_to_string(&page_path)
            .map_err(|e| error::page_not_found(format!("{}: {}", page_id, e)))?;

        let page: MemoryPage = serde_json::from_str(&json)
            .map_err(|e| error::parse_error(format!("Failed to parse page {}: {}", page_id, e)))?;

        Ok(page)
    }

    fn list_sessions(&self) -> Result<Vec<String>> {
        let mut sessions = Vec::new();

        let entries = std::fs::read_dir(&self.base_path)
            .map_err(|e| error::io_error(format!("Failed to read sessions dir: {}", e)))?;

        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("session_") {
                        sessions.push(name.to_string());
                    }
                }
            }
        }

        Ok(sessions)
    }

    fn delete_session(&self, session_id: &str) -> Result<()> {
        let session_dir = self.session_dir(session_id);
        std::fs::remove_dir_all(&session_dir)
            .map_err(|e| error::io_error(format!("Failed to delete session {}: {}", session_id, e)))?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "file"
    }
}

// =============================================================================
// In-Memory Backend (for testing)
// =============================================================================

/// In-memory session storage (useful for testing)
pub struct MemoryBackend {
    sessions: std::sync::RwLock<HashMap<String, Session>>,
    pages: std::sync::RwLock<HashMap<(String, String), MemoryPage>>,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            sessions: std::sync::RwLock::new(HashMap::new()),
            pages: std::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBackend for MemoryBackend {
    fn create_session(&self, task: &str) -> Result<Session> {
        let session = Session::new(Session::generate_id(), task);
        self.save_session(&session)?;
        Ok(session)
    }

    fn save_session(&self, session: &Session) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session.metadata.id.clone(), session.clone());
        Ok(())
    }

    fn load_session(&self, session_id: &str) -> Result<Session> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(session_id)
            .cloned()
            .ok_or_else(|| error::storage_not_found(format!("Session {}", session_id)))
    }

    fn save_page(&self, session_id: &str, page: &MemoryPage) -> Result<()> {
        let mut pages = self.pages.write().unwrap();
        pages.insert((session_id.to_string(), page.id.clone()), page.clone());
        Ok(())
    }

    fn load_page(&self, session_id: &str, page_id: &str) -> Result<MemoryPage> {
        let pages = self.pages.read().unwrap();
        pages.get(&(session_id.to_string(), page_id.to_string()))
            .cloned()
            .ok_or_else(|| error::page_not_found(page_id))
    }

    fn list_sessions(&self) -> Result<Vec<String>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.keys().cloned().collect())
    }

    fn delete_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let mut pages = self.pages.write().unwrap();

        sessions.remove(session_id);
        pages.retain(|(sid, _), _| sid != session_id);

        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "memory"
    }
}

// =============================================================================
// SQLite Backend (for future use)
// =============================================================================

// TODO: Implement SQLite backend for better querying and single-file storage
// This could use rusqlite or turso (like AgentFS)
//
// Schema could be:
// - sessions: id, task, created_at, updated_at, status, total_steps, llm_calls
// - page_index: session_id, page_id, summary, tokens, content_type, timestamps
// - pages: session_id, page_id, content (JSON)
// - trace: session_id, step, opcode, result, had_error

// =============================================================================
// AgentFS Backend (for future integration)
// =============================================================================

// TODO: Implement AgentFS backend using third_party/agentfs
//
// AgentFS provides:
// - kv store: Use for session metadata and page index
// - fs: Use for page content storage
// - tools: Use for execution trace with timing
//
// Benefits:
// - Single SQLite file per agent
// - SQL queryable history
// - Tool call performance tracking
// - FUSE/NFS mounting support
//
// Example integration:
// ```rust
// pub struct AgentFSBackend {
//     agent: agentfs_sdk::AgentFS,
// }
//
// impl SessionBackend for AgentFSBackend {
//     fn save_session(&self, session: &Session) -> Result<()> {
//         // Use kv store for session metadata
//         self.agent.kv.set(&format!("session:{}", session.metadata.id), &session)?;
//         Ok(())
//     }
//
//     fn save_page(&self, session_id: &str, page: &MemoryPage) -> Result<()> {
//         // Use filesystem for page content
//         let path = format!("/sessions/{}/pages/{}.json", session_id, page.id);
//         let content = serde_json::to_vec(page)?;
//         self.agent.fs.write_file(&path, &content)?;
//         Ok(())
//     }
// }
// ```

// =============================================================================
// SessionManager (wrapper with backend)
// =============================================================================

/// Manages session persistence with pluggable backends
pub struct SessionManager {
    backend: Box<dyn SessionBackend>,
}

impl SessionManager {
    /// Create a new session manager with the given backend
    pub fn with_backend(backend: impl SessionBackend + 'static) -> Self {
        Self {
            backend: Box::new(backend),
        }
    }

    /// Create a session manager with file backend (default)
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let backend = FileBackend::new(base_path)?;
        Ok(Self::with_backend(backend))
    }

    /// Create a session manager with in-memory backend
    pub fn in_memory() -> Self {
        Self::with_backend(MemoryBackend::new())
    }

    /// Get the backend name
    pub fn backend_name(&self) -> &'static str {
        self.backend.backend_name()
    }

    /// Create a new session
    pub fn create_session(&self, task: impl Into<String>) -> Result<Session> {
        self.backend.create_session(&task.into())
    }

    /// Save a session
    pub fn save_session(&self, session: &Session) -> Result<()> {
        self.backend.save_session(session)
    }

    /// Load a session
    pub fn load_session(&self, session_id: &str) -> Result<Session> {
        self.backend.load_session(session_id)
    }

    /// Save a page
    pub fn save_page(&self, session_id: &str, page: &MemoryPage) -> Result<()> {
        self.backend.save_page(session_id, page)
    }

    /// Load a page
    pub fn load_page(&self, session_id: &str, page_id: &str) -> Result<MemoryPage> {
        self.backend.load_page(session_id, page_id)
    }

    /// List sessions
    pub fn list_sessions(&self) -> Result<Vec<String>> {
        self.backend.list_sessions()
    }

    /// Delete a session
    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        self.backend.delete_session(session_id)
    }

    /// Get session info
    pub fn get_session_info(&self, session_id: &str) -> Result<SessionMetadata> {
        self.backend.get_session_info(session_id)
    }

    /// Check if session exists
    pub fn session_exists(&self, session_id: &str) -> bool {
        self.backend.session_exists(session_id)
    }
}

fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_session_creation() {
        let session = Session::new("test_session", "Test task");
        assert_eq!(session.metadata.id, "test_session");
        assert_eq!(session.metadata.task, "Test task");
        assert_eq!(session.metadata.status, SessionStatus::Active);
    }

    #[test]
    fn test_page_indexing() {
        let mut session = Session::new("test", "task");
        let page = MemoryPage::new("test_page", serde_json::json!({"key": "value"}));

        session.index_page(&page, Some("Test page summary".to_string()));

        assert!(session.page_index.contains_key("test_page"));
        assert_eq!(session.page_index["test_page"].summary, "Test page summary");
    }

    #[test]
    fn test_session_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SessionManager::new(temp_dir.path()).unwrap();

        // Create and save session
        let mut session = manager.create_session("Test task").unwrap();
        let page = MemoryPage::new("data", serde_json::json!({"content": "hello"}));
        session.index_page(&page, None);
        manager.save_session(&session).unwrap();
        manager.save_page(&session.metadata.id, &page).unwrap();

        // Load session
        let loaded = manager.load_session(&session.metadata.id).unwrap();
        assert_eq!(loaded.metadata.task, "Test task");
        assert!(loaded.page_index.contains_key("data"));

        // Load page
        let loaded_page = manager.load_page(&session.metadata.id, "data").unwrap();
        assert_eq!(loaded_page.content["content"], "hello");
    }

    #[test]
    fn test_trace_summary() {
        let mut session = Session::new("test", "task");

        session.add_trace(0, "READ_FILE", "success", false);
        session.add_trace(1, "INFER", "generated response", false);
        session.add_trace(2, "FAIL", "error occurred", true);

        let summary = session.get_trace_summary();
        assert!(summary.contains("READ_FILE"));
        assert!(summary.contains("⚠️"));
    }

    #[test]
    fn test_memory_backend() {
        let manager = SessionManager::in_memory();
        assert_eq!(manager.backend_name(), "memory");

        // Create and save session
        let mut session = manager.create_session("Memory test").unwrap();
        let page = MemoryPage::new("mem_page", serde_json::json!({"data": 42}));
        session.index_page(&page, Some("Test data".to_string()));
        manager.save_session(&session).unwrap();
        manager.save_page(&session.metadata.id, &page).unwrap();

        // Load and verify
        let loaded = manager.load_session(&session.metadata.id).unwrap();
        assert_eq!(loaded.metadata.task, "Memory test");

        let loaded_page = manager.load_page(&session.metadata.id, "mem_page").unwrap();
        assert_eq!(loaded_page.content["data"], 42);

        // List sessions
        let sessions = manager.list_sessions().unwrap();
        assert!(sessions.contains(&session.metadata.id));

        // Delete session
        manager.delete_session(&session.metadata.id).unwrap();
        assert!(manager.load_session(&session.metadata.id).is_err());
    }

    #[test]
    fn test_file_backend_name() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SessionManager::new(temp_dir.path()).unwrap();
        assert_eq!(manager.backend_name(), "file");
    }
}
