//! OpenAI-compatible provider implementation
//!
//! Works with OpenAI, Azure OpenAI, vLLM, Ollama, and other OpenAI-compatible APIs.

use super::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// OpenAI-compatible provider
pub struct OpenAIProvider {
    client: Client,
    config: ProviderConfig,
}

impl OpenAIProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs.unwrap_or(120)))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or("https://api.openai.com/v1")
    }
}

impl LlmProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn models(&self) -> Vec<String> {
        vec![
            "gpt-4o".into(),
            "gpt-4o-mini".into(),
            "gpt-4-turbo".into(),
            "gpt-4".into(),
            "gpt-3.5-turbo".into(),
            "o1".into(),
            "o1-mini".into(),
        ]
    }

    fn default_model(&self) -> &str {
        self.config.default_model.as_deref().unwrap_or("gpt-4o")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let model = request.model.as_deref().unwrap_or(self.default_model());

        let api_request = OpenAIRequest {
            model: model.to_string(),
            messages: request.messages.iter().map(|m| OpenAIMessage::from(m.clone())).collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(false),
            tools: request.tools.as_ref().map(|tools| {
                tools.iter().map(|t| OpenAITool {
                    r#type: "function".into(),
                    function: OpenAIFunction {
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

        let mut req = self.client
            .post(format!("{}/chat/completions", self.base_url()))
            .json(&api_request);

        if let Some(api_key) = &self.config.api_key {
            if !api_key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            }
        }

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

        let api_response: OpenAIResponse = response.json().await
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

        let api_request = OpenAIRequest {
            model: model.to_string(),
            messages: request.messages.iter().map(|m| OpenAIMessage::from(m.clone())).collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(true),
            tools: request.tools.as_ref().map(|tools| {
                tools.iter().map(|t| OpenAITool {
                    r#type: "function".into(),
                    function: OpenAIFunction {
                        name: t.name.clone(),
                        description: Some(t.description.clone()),
                        parameters: Some(t.parameters.clone()),
                    },
                }).collect()
            }),
            tool_choice: None,
            stop: request.stop,
        };

        let mut req = self.client
            .post(format!("{}/chat/completions", self.base_url()))
            .json(&api_request);

        if let Some(api_key) = &self.config.api_key {
            if !api_key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", api_key));
            }
        }

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

                                    if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
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
// OpenAI API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<ChatMessage> for OpenAIMessage {
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
                tcs.into_iter().map(|tc| OpenAIToolCall {
                    id: tc.id,
                    r#type: "function".into(),
                    function: OpenAIFunctionCall {
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
struct OpenAITool {
    r#type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    r#type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<OpenAIFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}
