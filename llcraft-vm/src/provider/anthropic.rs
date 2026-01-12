//! Anthropic Claude provider implementation

use super::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Anthropic Claude provider
pub struct AnthropicProvider {
    client: Client,
    config: ProviderConfig,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs.unwrap_or(120)))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or("https://api.anthropic.com/v1")
    }
}

impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn models(&self) -> Vec<String> {
        vec![
            "claude-sonnet-4-20250514".into(),
            "claude-opus-4-20250514".into(),
            "claude-3-5-sonnet-20241022".into(),
            "claude-3-5-haiku-20241022".into(),
            "claude-3-opus-20240229".into(),
        ]
    }

    fn default_model(&self) -> &str {
        self.config.default_model.as_deref().unwrap_or("claude-sonnet-4-20250514")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let model = request.model.as_deref().unwrap_or(self.default_model());
        
        // Extract system message
        let (system, messages): (Option<String>, Vec<_>) = {
            let mut sys = None;
            let mut msgs = Vec::new();
            for msg in &request.messages {
                if msg.role == Role::System {
                    sys = msg.content.clone();
                } else {
                    msgs.push(AnthropicMessage::from(msg.clone()));
                }
            }
            (sys, msgs)
        };

        let api_request = AnthropicRequest {
            model: model.to_string(),
            messages,
            system,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            stream: Some(false),
            tools: request.tools.as_ref().map(|tools| {
                tools.iter().map(|t| AnthropicTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.parameters.clone(),
                }).collect()
            }),
            stop_sequences: request.stop,
        };

        let api_key = self.config.api_key.as_ref()
            .ok_or(ProviderError::AuthenticationFailed)?;

        let mut req = self.client
            .post(format!("{}/messages", self.base_url()))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&api_request);

        for (key, value) in &self.config.headers {
            req = req.header(key, value);
        }

        let response = req.send().await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            
            if status == 429 {
                return Err(ProviderError::RateLimited { retry_after: None });
            } else if status == 401 {
                return Err(ProviderError::AuthenticationFailed);
            }
            
            return Err(ProviderError::Api { status, message: text });
        }

        let api_response: AnthropicResponse = response.json().await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        // Extract text content and tool calls
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for block in &api_response.content {
            match block {
                ContentBlock::Text { text } => {
                    content.push_str(text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    });
                }
            }
        }

        let finish_reason = match api_response.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            Some("tool_use") => FinishReason::ToolCalls,
            _ => FinishReason::Unknown,
        };

        let usage = Usage {
            prompt_tokens: api_response.usage.input_tokens,
            completion_tokens: api_response.usage.output_tokens,
            total_tokens: api_response.usage.input_tokens + api_response.usage.output_tokens,
        };

        Ok(CompletionResponse {
            id: api_response.id,
            model: api_response.model,
            content: if content.is_empty() { None } else { Some(content) },
            tool_calls,
            finish_reason,
            usage,
        })
    }

    async fn stream(&self, request: CompletionRequest) -> Result<StreamReceiver, ProviderError> {
        let model = request.model.as_deref().unwrap_or(self.default_model());
        
        // Extract system message
        let (system, messages): (Option<String>, Vec<_>) = {
            let mut sys = None;
            let mut msgs = Vec::new();
            for msg in &request.messages {
                if msg.role == Role::System {
                    sys = msg.content.clone();
                } else {
                    msgs.push(AnthropicMessage::from(msg.clone()));
                }
            }
            (sys, msgs)
        };

        let api_request = AnthropicRequest {
            model: model.to_string(),
            messages,
            system,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            stream: Some(true),
            tools: request.tools.as_ref().map(|tools| {
                tools.iter().map(|t| AnthropicTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.parameters.clone(),
                }).collect()
            }),
            stop_sequences: request.stop,
        };

        let api_key = self.config.api_key.as_ref()
            .ok_or(ProviderError::AuthenticationFailed)?;

        let mut req = self.client
            .post(format!("{}/messages", self.base_url()))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&api_request);

        for (key, value) in &self.config.headers {
            req = req.header(key, value);
        }

        let response = req.send().await
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
            let mut current_tool_index = 0;
            
            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        
                        // Process complete SSE events
                        while let Some(pos) = buffer.find("\n\n") {
                            let event = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();
                            
                            let mut event_type = None;
                            let mut event_data = None;
                            
                            for line in event.lines() {
                                if let Some(t) = line.strip_prefix("event: ") {
                                    event_type = Some(t.to_string());
                                } else if let Some(d) = line.strip_prefix("data: ") {
                                    event_data = Some(d.to_string());
                                }
                            }
                            
                            if let (Some(etype), Some(data)) = (event_type, event_data) {
                                match etype.as_str() {
                                    "content_block_delta" => {
                                        if let Ok(delta) = serde_json::from_str::<ContentBlockDelta>(&data) {
                                            match delta.delta {
                                                DeltaContent::TextDelta { text } => {
                                                    yield StreamChunk::Text(text);
                                                }
                                                DeltaContent::InputJsonDelta { partial_json } => {
                                                    yield StreamChunk::ToolCallDelta {
                                                        index: current_tool_index,
                                                        id: None,
                                                        name: None,
                                                        arguments_delta: Some(partial_json),
                                                    };
                                                }
                                            }
                                        }
                                    }
                                    "content_block_start" => {
                                        if let Ok(start) = serde_json::from_str::<ContentBlockStart>(&data) {
                                            if let Some(tool_use) = start.content_block.tool_use {
                                                yield StreamChunk::ToolCallDelta {
                                                    index: start.index,
                                                    id: Some(tool_use.id),
                                                    name: Some(tool_use.name),
                                                    arguments_delta: None,
                                                };
                                                current_tool_index = start.index;
                                            }
                                        }
                                    }
                                    "message_stop" => {
                                        yield StreamChunk::Done {
                                            finish_reason: FinishReason::Stop,
                                            usage: None,
                                        };
                                    }
                                    "message_delta" => {
                                        if let Ok(delta) = serde_json::from_str::<MessageDelta>(&data) {
                                            if let Some(reason) = delta.delta.stop_reason {
                                                let fr = match reason.as_str() {
                                                    "end_turn" => FinishReason::Stop,
                                                    "max_tokens" => FinishReason::Length,
                                                    "tool_use" => FinishReason::ToolCalls,
                                                    _ => FinishReason::Unknown,
                                                };
                                                yield StreamChunk::Done {
                                                    finish_reason: fr,
                                                    usage: delta.usage.map(|u| Usage {
                                                        prompt_tokens: 0, // Not available in delta
                                                        completion_tokens: u.output_tokens,
                                                        total_tokens: u.output_tokens,
                                                    }),
                                                };
                                            }
                                        }
                                    }
                                    _ => {}
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
// Anthropic API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

impl From<ChatMessage> for AnthropicMessage {
    fn from(msg: ChatMessage) -> Self {
        let role = match msg.role {
            Role::User | Role::System => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user",
        };

        let content = if msg.role == Role::Tool {
            AnthropicContent::Blocks(vec![AnthropicContentBlock::ToolResult {
                tool_use_id: msg.tool_call_id.unwrap_or_default(),
                content: msg.content.unwrap_or_default(),
            }])
        } else {
            AnthropicContent::Text(msg.content.unwrap_or_default())
        };

        Self {
            role: role.into(),
            content,
        }
    }
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

// Streaming types
#[derive(Debug, Deserialize)]
struct ContentBlockDelta {
    delta: DeltaContent,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum DeltaContent {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
struct ContentBlockStart {
    index: usize,
    content_block: ContentBlockInfo,
}

#[derive(Debug, Deserialize)]
struct ContentBlockInfo {
    #[serde(flatten)]
    tool_use: Option<ToolUseStart>,
}

#[derive(Debug, Deserialize)]
struct ToolUseStart {
    id: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    delta: MessageDeltaContent,
    usage: Option<DeltaUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaContent {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaUsage {
    output_tokens: usize,
}
