//! # LLM-VM Storage
//!
//! Persistent key-value storage for the LLM Virtual Machine.
//! Unlike Memory (volatile), Storage persists across executions.
//! Used for caching, checkpoints, and long-term state.

use crate::error::{self, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Storage backend trait
pub trait StorageBackend: Send + Sync {
    fn get(&self, key: &str) -> Option<serde_json::Value>;
    fn set(&mut self, key: &str, value: serde_json::Value) -> Result<()>;
    fn delete(&mut self, key: &str) -> Result<()>;
    fn exists(&self, key: &str) -> bool;
    fn keys(&self) -> Vec<String>;
    fn clear(&mut self) -> Result<()>;
}

/// In-memory storage (volatile, but useful for testing)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStorage {
    data: HashMap<String, serde_json::Value>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl StorageBackend for MemoryStorage {
    fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.data.get(key).cloned()
    }

    fn set(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        self.data.insert(key.to_string(), value);
        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        self.data.remove(key);
        Ok(())
    }

    fn exists(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    fn clear(&mut self) -> Result<()> {
        self.data.clear();
        Ok(())
    }
}

/// File-based storage (persistent)
pub struct FileStorage {
    base_path: PathBuf,
}

impl FileStorage {
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)
            .map_err(|e| error::io_error(format!("Failed to create storage dir: {}", e)))?;
        Ok(Self { base_path })
    }

    fn key_to_path(&self, key: &str) -> PathBuf {
        // Sanitize key for use as filename
        let safe_key = key
            .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        self.base_path.join(format!("{}.json", safe_key))
    }
}

impl StorageBackend for FileStorage {
    fn get(&self, key: &str) -> Option<serde_json::Value> {
        let path = self.key_to_path(key);
        let content = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    fn set(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        let path = self.key_to_path(key);
        let content = serde_json::to_string_pretty(&value)
            .map_err(|e| error::serialization_error(e.to_string()))?;
        std::fs::write(&path, content)
            .map_err(|e| error::io_error(format!("Failed to write {}: {}", path.display(), e)))?;
        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        let path = self.key_to_path(key);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| error::io_error(format!("Failed to delete {}: {}", path.display(), e)))?;
        }
        Ok(())
    }

    fn exists(&self, key: &str) -> bool {
        self.key_to_path(key).exists()
    }

    fn keys(&self) -> Vec<String> {
        std::fs::read_dir(&self.base_path)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let path = e.path();
                        if path.extension().map(|ext| ext == "json").unwrap_or(false) {
                            path.file_stem()
                                .and_then(|s| s.to_str())
                                .map(|s| s.to_string())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn clear(&mut self) -> Result<()> {
        for key in self.keys() {
            self.delete(&key)?;
        }
        Ok(())
    }
}

/// LLM-VM Storage - high-level interface
pub struct Storage {
    backend: Box<dyn StorageBackend>,
    /// Namespace prefix for keys
    namespace: Option<String>,
}

impl Storage {
    /// Create storage with in-memory backend
    pub fn memory() -> Self {
        Self {
            backend: Box::new(MemoryStorage::new()),
            namespace: None,
        }
    }

    /// Create storage with file backend
    pub fn file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            backend: Box::new(FileStorage::new(path)?),
            namespace: None,
        })
    }

    /// Create storage with custom backend
    pub fn with_backend(backend: impl StorageBackend + 'static) -> Self {
        Self {
            backend: Box::new(backend),
            namespace: None,
        }
    }

    /// Set namespace for all operations
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    fn full_key(&self, key: &str) -> String {
        match &self.namespace {
            Some(ns) => format!("{}:{}", ns, key),
            None => key.to_string(),
        }
    }

    /// Get a value from storage
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.backend.get(&self.full_key(key))
    }

    /// Get a value or return default
    pub fn get_or(&self, key: &str, default: serde_json::Value) -> serde_json::Value {
        self.get(key).unwrap_or(default)
    }

    /// Get a typed value from storage
    pub fn get_typed<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.get(key).and_then(|v| serde_json::from_value(v).ok())
    }

    /// Set a value in storage
    pub fn set(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        self.backend.set(&self.full_key(key), value)
    }

    /// Set a typed value in storage
    pub fn set_typed<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        let json = serde_json::to_value(value)
            .map_err(|e| error::serialization_error(e.to_string()))?;
        self.set(key, json)
    }

    /// Delete a value from storage
    pub fn delete(&mut self, key: &str) -> Result<()> {
        self.backend.delete(&self.full_key(key))
    }

    /// Check if a key exists
    pub fn exists(&self, key: &str) -> bool {
        self.backend.exists(&self.full_key(key))
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<String> {
        let prefix = self.namespace.as_ref().map(|ns| format!("{}:", ns));
        self.backend
            .keys()
            .into_iter()
            .filter_map(|k| {
                if let Some(ref p) = prefix {
                    k.strip_prefix(p).map(|s| s.to_string())
                } else {
                    Some(k)
                }
            })
            .collect()
    }

    /// Clear all values
    pub fn clear(&mut self) -> Result<()> {
        self.backend.clear()
    }

    // ========================================================================
    // Checkpoint support
    // ========================================================================

    /// Save a checkpoint
    pub fn checkpoint(&mut self, name: &str, data: serde_json::Value) -> Result<()> {
        let key = format!("_checkpoint:{}", name);
        self.backend.set(&key, data)
    }

    /// Load a checkpoint
    pub fn load_checkpoint(&self, name: &str) -> Option<serde_json::Value> {
        let key = format!("_checkpoint:{}", name);
        self.backend.get(&key)
    }

    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Vec<String> {
        self.backend
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_prefix("_checkpoint:").map(|s| s.to_string()))
            .collect()
    }

    /// Delete a checkpoint
    pub fn delete_checkpoint(&mut self, name: &str) -> Result<()> {
        let key = format!("_checkpoint:{}", name);
        self.backend.delete(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_memory_storage() {
        let mut storage = Storage::memory();

        storage.set("key1", json!("value1")).unwrap();
        storage.set("key2", json!(42)).unwrap();

        assert_eq!(storage.get("key1"), Some(json!("value1")));
        assert_eq!(storage.get("key2"), Some(json!(42)));
        assert_eq!(storage.get("key3"), None);

        storage.delete("key1").unwrap();
        assert_eq!(storage.get("key1"), None);
    }

    #[test]
    fn test_namespace() {
        let mut storage = Storage::memory().with_namespace("test");

        storage.set("key", json!("value")).unwrap();

        // Key should be namespaced internally
        assert_eq!(storage.get("key"), Some(json!("value")));
    }

    #[test]
    fn test_typed_storage() {
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct Config {
            name: String,
            count: i32,
        }

        let mut storage = Storage::memory();

        let config = Config { name: "test".to_string(), count: 42 };
        storage.set_typed("config", &config).unwrap();

        let loaded: Option<Config> = storage.get_typed("config");
        assert_eq!(loaded, Some(config));
    }

    #[test]
    fn test_get_or() {
        let storage = Storage::memory();

        let value = storage.get_or("missing", json!("default"));
        assert_eq!(value, json!("default"));
    }

    #[test]
    fn test_checkpoints() {
        let mut storage = Storage::memory();

        storage.checkpoint("before_change", json!({"state": 1})).unwrap();
        storage.checkpoint("after_change", json!({"state": 2})).unwrap();

        let checkpoints = storage.list_checkpoints();
        assert!(checkpoints.contains(&"before_change".to_string()));
        assert!(checkpoints.contains(&"after_change".to_string()));

        let data = storage.load_checkpoint("before_change");
        assert_eq!(data, Some(json!({"state": 1})));

        storage.delete_checkpoint("before_change").unwrap();
        assert_eq!(storage.load_checkpoint("before_change"), None);
    }
}
