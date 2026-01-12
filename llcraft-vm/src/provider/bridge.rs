//! Bridge provider - connects to the local Copilot API Bridge
//!
//! The bridge is a VS Code extension that exposes GitHub Copilot's
//! language models via a local HTTP endpoint (OpenAI-compatible format).
//!
//! Default endpoint: http://localhost:5168/v1/chat/completions

use super::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Bridge provider - connects to local Copilot API Bridge
pub struct BridgeProvider {
    client: Client,
    config: ProviderConfig,
}

impl BridgeProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs.unwrap_or(300)))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Create with default local bridge settings
    pub fn local() -> Self {
        Self::new(ProviderConfig::bridge())
    }

    /// Create with custom port
    pub fn with_port(port: u16) -> Self {
        Self::new(ProviderConfig {
            provider_type: ProviderType::Bridge,
            api_key: None,
            base_url: Some(format!("http://localhost:{}", port)),
            default_model: Some("claude-opus-4".into()),
            headers: std::collections::HashMap::new(),
            timeout_secs: Some(300),
        })
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or("http://localhost:5168")
    }

    /// Check if the bridge is running
    pub async fn health_check(&self) -> Result<bool, ProviderError> {
        let response = self.client
            .get(format!("{}/health", self.base_url()))
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        Ok(response.status().is_success())
    }

    /// Get available models from the bridge
    pub async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let response = self.client
            .get(format!("{}/v1/models", self.base_url()))
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ProviderError::Api {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let models: ModelsResponse = response.json().await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(models.data.into_iter().map(|m| m.id).collect())
    }
}

impl LlmProvider for BridgeProvider {
    fn name(&self) -> &str {
        "bridge"
    }

    fn models(&self) -> Vec<String> {
        // These are the models typically available through Copilot
        vec![
            "claude-sonnet-4".into(),
            "claude-opus-4".into(),
            "gpt-4o".into(),
            "gpt-4o-mini".into(),
            "o1".into(),
            "o1-mini".into(),
            "gemini-2.5-pro".into(),
        ]
    }

    fn default_model(&self) -> &str {
        self.config.default_model.as_deref().unwrap_or("claude-opus-4")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let model = request.model.as_deref().unwrap_or(self.default_model());

        let api_request = BridgeRequest {
            model: model.to_string(),
            messages: request.messages.iter().map(|m| BridgeMessage::from(m.clone())).collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(false),
            tools: request.tools.as_ref().map(|tools| {
                tools.iter().map(|t| BridgeTool {
                    r#type: "function".into(),
                    function: BridgeFunction {
                        name: t.name.clone(),
                        description: Some(t.description.clone()),
                        parameters: Some(t.parameters.clone()),
                    },
                }).collect()
            }),
            tool_choice: request.tool_choice.as_ref().map(|tc| match tc {
                ToolChoice::Auto => serde_json::json!("auto"),
                ToolChoice::None => serde_json::json!("none"),
                ToolChoice::Required => serde_json::json!("required"),
                ToolChoice::Function { name } => serde_json::json!({
                    "type": "function",
                    "function": { "name": name }
                }),
            }),
            stop: request.stop,
        };

        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url()))
            .json(&api_request)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();

            if status == 503 {
                return Err(ProviderError::Other(
                    "Bridge not available. Make sure VS Code with the bridge extension is running.".into()
                ));
            }

            return Err(ProviderError::Api { status, message: text });
        }

        let api_response: BridgeResponse = response.json().await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        let choice = api_response.choices.first()
            .ok_or_else(|| ProviderError::Other("No choices in response".into()))?;

        let tool_calls = choice.message.tool_calls.as_ref()
            .map(|tcs| tcs.iter().map(|tc| ToolCall {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            }).collect())
            .unwrap_or_default();

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        };

        let usage = api_response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }).unwrap_or_default();

        Ok(CompletionResponse {
            id: api_response.id,
            model: api_response.model,
            content: choice.message.content.clone(),
            tool_calls,
            finish_reason,
            usage,
        })
    }

    async fn stream(&self, request: CompletionRequest) -> Result<StreamReceiver, ProviderError> {
        let model = request.model.as_deref().unwrap_or(self.default_model());

        let api_request = BridgeRequest {
            model: model.to_string(),
            messages: request.messages.iter().map(|m| BridgeMessage::from(m.clone())).collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(true),
            tools: request.tools.as_ref().map(|tools| {
                tools.iter().map(|t| BridgeTool {
                    r#type: "function".into(),
                    function: BridgeFunction {
                        name: t.name.clone(),
                        description: Some(t.description.clone()),
                        parameters: Some(t.parameters.clone()),
                    },
                }).collect()
            }),
            tool_choice: None,
            stop: request.stop,
        };

        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url()))
            .json(&api_request)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api { status, message: text });
        }

        // Create async stream from SSE response
        let stream = async_stream::stream! {
            use futures_util::StreamExt;

            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process complete SSE events
                        while let Some(pos) = buffer.find("\n\n") {
                            let event = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            for line in event.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if data == "[DONE]" {
                                        yield StreamChunk::Done {
                                            finish_reason: FinishReason::Stop,
                                            usage: None,
                                        };
                                        return;
                                    }

                                    if let Ok(chunk) = serde_json::from_str::<BridgeStreamChunk>(data) {
                                        if let Some(choice) = chunk.choices.first() {
                                            if let Some(content) = &choice.delta.content {
                                                yield StreamChunk::Text(content.clone());
                                            }

                                            if let Some(tool_calls) = &choice.delta.tool_calls {
                                                for tc in tool_calls {
                                                    yield StreamChunk::ToolCallDelta {
                                                        index: tc.index,
                                                        id: tc.id.clone(),
                                                        name: tc.function.as_ref().and_then(|f| f.name.clone()),
                                                        arguments_delta: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                                                    };
                                                }
                                            }

                                            if let Some(reason) = &choice.finish_reason {
                                                let fr = match reason.as_str() {
                                                    "stop" => FinishReason::Stop,
                                                    "length" => FinishReason::Length,
                                                    "tool_calls" => FinishReason::ToolCalls,
                                                    _ => FinishReason::Unknown,
                                                };
                                                yield StreamChunk::Done {
                                                    finish_reason: fr,
                                                    usage: None,
                                                };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield StreamChunk::Error(e.to_string());
                        return;
                    }
                }
            }
        };

        Ok(StreamReceiver::new(stream))
    }
}

// ============================================================================
// Bridge API Types (OpenAI-compatible format)
// ============================================================================

#[derive(Debug, Serialize)]
struct BridgeRequest {
    model: String,
    messages: Vec<BridgeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<BridgeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BridgeMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<BridgeToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<ChatMessage> for BridgeMessage {
    fn from(msg: ChatMessage) -> Self {
        Self {
            role: match msg.role {
                Role::System => "system".into(),
                Role::User => "user".into(),
                Role::Assistant => "assistant".into(),
                Role::Tool => "tool".into(),
            },
            content: msg.content,
            tool_calls: msg.tool_calls.map(|tcs| {
                tcs.into_iter().map(|tc| BridgeToolCall {
                    id: tc.id,
                    r#type: "function".into(),
                    function: BridgeFunctionCall {
                        name: tc.name,
                        arguments: tc.arguments,
                    },
                }).collect()
            }),
            tool_call_id: msg.tool_call_id,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct BridgeTool {
    r#type: String,
    function: BridgeFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct BridgeFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BridgeToolCall {
    id: String,
    r#type: String,
    function: BridgeFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct BridgeFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct BridgeResponse {
    id: String,
    model: String,
    choices: Vec<BridgeChoice>,
    usage: Option<BridgeUsage>,
}

#[derive(Debug, Deserialize)]
struct BridgeChoice {
    message: BridgeMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BridgeUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct BridgeStreamChunk {
    choices: Vec<BridgeStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct BridgeStreamChoice {
    delta: BridgeStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BridgeStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<BridgeToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct BridgeToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<BridgeFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct BridgeFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_provider_config() {
        let provider = BridgeProvider::local();
        assert_eq!(provider.name(), "bridge");
        assert_eq!(provider.default_model(), "claude-sonnet-4");
    }

    #[test]
    fn test_bridge_with_port() {
        let provider = BridgeProvider::with_port(8080);
        assert_eq!(provider.base_url(), "http://localhost:8080");
    }
}
