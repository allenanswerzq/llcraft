//! # LLM Provider Interface
//!
//! A trait-based abstraction for communicating with LLM backends.
//! Supports streaming, tool calls, and multiple providers.
//!
//! ## Design
//! - `LlmProvider` trait defines the core interface
//! - Implementations for OpenAI, Anthropic, Bridge (local Copilot), and local models
//! - Streaming via async iterators
//! - Tool/function calling support
//! - Usage tracking

pub mod openai;
pub mod anthropic;
pub mod bridge;

pub use openai::OpenAIProvider;
pub use anthropic::AnthropicProvider;
pub use bridge::BridgeProvider;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;

// ============================================================================
// Core Types
// ============================================================================

/// A chat message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Pretty print the message to stdout
    pub fn pretty_print(&self) {
        let role_str = match self.role {
            Role::System => "SYSTEM",
            Role::User => "USER",
            Role::Assistant => "ASSISTANT",
            Role::Tool => "TOOL",
        };
        println!("[{}]", role_str);
        if let Some(content) = &self.content {
            println!("{}", content);
        }
        if let Some(tool_calls) = &self.tool_calls {
            for tc in tool_calls {
                println!("  tool_call: {}({})", tc.name, tc.arguments);
            }
        }
        if let Some(id) = &self.tool_call_id {
            println!("  tool_call_id: {}", id);
        }
        println!();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A tool/function that the model can call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = parameters;
        self
    }
}

/// A tool call requested by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    /// Parse arguments as JSON
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

/// Request parameters for a completion
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    pub stream: bool,
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

/// Response from a completion request
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Unknown,
}

/// Token usage information
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// A streaming chunk from the model
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Text content delta
    Text(String),
    /// Tool call being built
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    /// Stream finished
    Done {
        finish_reason: FinishReason,
        usage: Option<Usage>,
    },
    /// Error occurred
    Error(String),
}

// ============================================================================
// Provider Trait
// ============================================================================

/// Error type for provider operations
#[derive(Debug)]
pub enum ProviderError {
    /// Network/connection error
    Network(String),
    /// API returned an error
    Api { status: u16, message: String },
    /// Failed to parse response
    Parse(String),
    /// Rate limited
    RateLimited { retry_after: Option<u64> },
    /// Invalid request
    InvalidRequest(String),
    /// Model not found
    ModelNotFound(String),
    /// Authentication failed
    AuthenticationFailed,
    /// Other error
    Other(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(e) => write!(f, "Network error: {}", e),
            Self::Api { status, message } => write!(f, "API error ({}): {}", status, message),
            Self::Parse(e) => write!(f, "Parse error: {}", e),
            Self::RateLimited { retry_after } => {
                write!(f, "Rate limited")?;
                if let Some(secs) = retry_after {
                    write!(f, " (retry after {}s)", secs)?;
                }
                Ok(())
            }
            Self::InvalidRequest(e) => write!(f, "Invalid request: {}", e),
            Self::ModelNotFound(m) => write!(f, "Model not found: {}", m),
            Self::AuthenticationFailed => write!(f, "Authentication failed"),
            Self::Other(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ProviderError {}

/// The main LLM provider trait
#[allow(async_fn_in_trait)]
pub trait LlmProvider: Send + Sync {
    /// Get the provider name (e.g., "openai", "anthropic")
    fn name(&self) -> &str;

    /// Get available models
    fn models(&self) -> Vec<String>;

    /// Get the default model
    fn default_model(&self) -> &str;

    /// Send a completion request and get a full response
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError>;

    /// Send a completion request and stream the response
    async fn stream(&self, request: CompletionRequest) -> Result<StreamReceiver, ProviderError>;

    /// Simple prompt -> response helper
    async fn prompt(&self, prompt: &str) -> Result<String, ProviderError> {
        let request = CompletionRequest::new(vec![ChatMessage::user(prompt)]);
        let response = self.complete(request).await?;
        response.content.ok_or_else(|| ProviderError::Other("No content in response".into()))
    }

    /// Chat with message history
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String, ProviderError> {
        let request = CompletionRequest::new(messages);
        let response = self.complete(request).await?;
        response.content.ok_or_else(|| ProviderError::Other("No content in response".into()))
    }
}

/// Receiver for streaming responses
pub struct StreamReceiver {
    inner: Pin<Box<dyn futures_core::Stream<Item = StreamChunk> + Send>>,
}

impl StreamReceiver {
    pub fn new<S>(stream: S) -> Self
    where
        S: futures_core::Stream<Item = StreamChunk> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Collect all text chunks into a single string
    pub async fn collect_text(mut self) -> Result<String, ProviderError> {
        use futures_core::Stream;
        use std::task::{Context, Poll};

        let mut text = String::new();
        let waker = futures_task::noop_waker();
        let mut cx = Context::from_waker(&waker);

        loop {
            match Pin::new(&mut self.inner).poll_next(&mut cx) {
                Poll::Ready(Some(chunk)) => match chunk {
                    StreamChunk::Text(t) => text.push_str(&t),
                    StreamChunk::Done { .. } => break,
                    StreamChunk::Error(e) => return Err(ProviderError::Other(e)),
                    _ => {}
                },
                Poll::Ready(None) => break,
                Poll::Pending => {
                    // In real async context, this would yield
                    // For now, just continue
                    continue;
                }
            }
        }
        Ok(text)
    }
}

// ============================================================================
// Provider Configuration
// ============================================================================

/// Configuration for creating providers
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider_type: ProviderType,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub headers: HashMap<String, String>,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    Bridge,
    Local,
    Custom,
}

impl ProviderConfig {
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::OpenAI,
            api_key: Some(api_key.into()),
            base_url: Some("https://api.openai.com/v1".into()),
            default_model: Some("gpt-4o".into()),
            headers: HashMap::new(),
            timeout_secs: Some(120),
        }
    }

    pub fn anthropic(api_key: impl Into<String>) -> Self {
        let mut headers = HashMap::new();
        headers.insert("anthropic-version".into(), "2023-06-01".into());

        Self {
            provider_type: ProviderType::Anthropic,
            api_key: Some(api_key.into()),
            base_url: Some("https://api.anthropic.com/v1".into()),
            default_model: Some("claude-sonnet-4-20250514".into()),
            headers,
            timeout_secs: Some(120),
        }
    }

    /// Connect to the local Copilot API Bridge (VS Code extension)
    /// Default port: 5168
    pub fn bridge() -> Self {
        Self {
            provider_type: ProviderType::Bridge,
            api_key: None,
            base_url: Some("http://localhost:5168".into()),
            default_model: Some("claude-opus-4".into()),
            headers: HashMap::new(),
            timeout_secs: Some(300),
        }
    }

    /// Connect to the bridge on a custom port
    pub fn bridge_with_port(port: u16) -> Self {
        Self {
            provider_type: ProviderType::Bridge,
            api_key: None,
            base_url: Some(format!("http://localhost:{}", port)),
            default_model: Some("claude-opus-4".into()),
            headers: HashMap::new(),
            timeout_secs: Some(300),
        }
    }

    pub fn local(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider_type: ProviderType::Local,
            api_key: None,
            base_url: Some(base_url.into()),
            default_model: Some(model.into()),
            headers: HashMap::new(),
            timeout_secs: Some(300),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

// ============================================================================
// Usage Tracking
// ============================================================================

/// Tracks token usage across multiple calls
#[derive(Debug, Clone, Default)]
pub struct UsageTracker {
    pub total_calls: usize,
    pub total_prompt_tokens: usize,
    pub total_completion_tokens: usize,
    pub by_model: HashMap<String, Usage>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn track(&mut self, model: &str, usage: &Usage) {
        self.total_calls += 1;
        self.total_prompt_tokens += usage.prompt_tokens;
        self.total_completion_tokens += usage.completion_tokens;

        let entry = self.by_model.entry(model.to_string()).or_default();
        entry.prompt_tokens += usage.prompt_tokens;
        entry.completion_tokens += usage.completion_tokens;
        entry.total_tokens += usage.total_tokens;
    }

    pub fn total_tokens(&self) -> usize {
        self.total_prompt_tokens + self.total_completion_tokens
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("You are helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content.as_deref(), Some("You are helpful"));

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, Role::User);

        let asst = ChatMessage::assistant("Hi there!");
        assert_eq!(asst.role, Role::Assistant);
    }

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition::new("read_file", "Read contents of a file")
            .with_parameters(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path" }
                },
                "required": ["path"]
            }));

        assert_eq!(tool.name, "read_file");
        assert!(tool.parameters["properties"]["path"].is_object());
    }

    #[test]
    fn test_completion_request_builder() {
        let request = CompletionRequest::new(vec![ChatMessage::user("Hello")])
            .with_model("gpt-4o")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_streaming(true);

        assert_eq!(request.model, Some("gpt-4o".into()));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_tokens, Some(1000));
        assert!(request.stream);
    }

    #[test]
    fn test_provider_config() {
        let config = ProviderConfig::openai("sk-test");
        assert_eq!(config.provider_type, ProviderType::OpenAI);
        assert_eq!(config.default_model, Some("gpt-4o".into()));

        let config = ProviderConfig::anthropic("sk-ant-test");
        assert_eq!(config.provider_type, ProviderType::Anthropic);
        assert!(config.headers.contains_key("anthropic-version"));
    }

    #[test]
    fn test_usage_tracker() {
        let mut tracker = UsageTracker::new();

        tracker.track("gpt-4o", &Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        });

        tracker.track("gpt-4o", &Usage {
            prompt_tokens: 200,
            completion_tokens: 100,
            total_tokens: 300,
        });

        assert_eq!(tracker.total_calls, 2);
        assert_eq!(tracker.total_prompt_tokens, 300);
        assert_eq!(tracker.total_completion_tokens, 150);
        assert_eq!(tracker.total_tokens(), 450);
    }
}
