//! Example: Test the LLM provider with a real model
//!
//! Run with:
//!   # Use local Copilot bridge (default):
//!   cargo run --example test_provider
//!
//!   # Use OpenAI:
//!   OPENAI_API_KEY=sk-xxx cargo run --example test_provider -- --openai
//!
//!   # Use Anthropic:
//!   ANTHROPIC_API_KEY=sk-xxx cargo run --example test_provider -- --anthropic
//!
//!   # Use local Ollama:
//!   cargo run --example test_provider -- --ollama
//!
//!   # Just output the prompt:
//!   cargo run --example test_provider -- --prompt-only

use llcraft_vm::{
    LlmProvider, ProviderConfig, ChatMessage,
    OpenAIProvider, AnthropicProvider, BridgeProvider,
    VmSchema, TaskRequest,
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let use_openai = args.iter().any(|arg| arg == "--openai");
    let use_anthropic = args.iter().any(|arg| arg == "--anthropic");
    let use_ollama = args.iter().any(|arg| arg == "--ollama");
    let prompt_only = args.iter().any(|arg| arg == "--prompt-only");
    let verbose = args.iter().any(|arg| arg == "-v" || arg == "--verbose");

    // Create the VM schema
    let schema = VmSchema::new();

    // Create a task request
    let task = TaskRequest::new(
        "Read the file 'README.md' and summarize its contents. \
         If the file doesn't exist, report an error."
    );

    // Generate prompts using the new separated format
    let system_prompt = TaskRequest::system_prompt(&schema);
    let user_prompt = task.user_prompt();

    if prompt_only {
        println!("=== SYSTEM PROMPT ===\n{}\n", system_prompt);
        println!("=== USER PROMPT ===\n{}\n", user_prompt);
        println!("---\nCopy the above prompts and paste them into ChatGPT/Claude.");
        return Ok(());
    }

    // Create messages with separated system/user prompts
    let messages = vec![
        ChatMessage::system(&system_prompt),
        ChatMessage::user(&user_prompt),
    ];

    if verbose {
        println!("=== SYSTEM ===");
        println!("{}", system_prompt);
        println!("\n=== USER PROMPT ===");
        println!("{}", user_prompt);
        println!("\n=== END PROMPT ===\n");
    }

    // Run based on provider type
    if use_ollama {
        println!("Using Ollama (localhost:11434)...");
        let provider = OpenAIProvider::new(
            ProviderConfig::local("http://localhost:11434/v1", "llama3.3")
        );
        run_with_provider(&provider, messages).await?;
    } else if use_anthropic {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .expect("Set ANTHROPIC_API_KEY environment variable");
        println!("Using Anthropic Claude...");
        let provider = AnthropicProvider::new(ProviderConfig::anthropic(api_key));
        run_with_provider(&provider, messages).await?;
    } else if use_openai {
        let api_key = env::var("OPENAI_API_KEY")
            .expect("Set OPENAI_API_KEY environment variable");
        println!("Using OpenAI...");
        let provider = OpenAIProvider::new(ProviderConfig::openai(api_key));
        run_with_provider(&provider, messages).await?;
    } else {
        // Default: use local bridge
        println!("Using local Copilot bridge (localhost:5168)...");
        let provider = BridgeProvider::local();

        // Check if bridge is running
        match provider.health_check().await {
            Ok(true) => {
                println!("✓ Bridge is running");
            }
            Ok(false) | Err(_) => {
                eprintln!("✗ Bridge not responding. Make sure VS Code with the bridge extension is running.");
                eprintln!("  Or use --openai, --anthropic, or --ollama flags.");
                return Ok(());
            }
        }

        run_with_provider(&provider, messages).await?;
    }

    Ok(())
}

async fn run_with_provider<P: LlmProvider>(provider: &P, messages: Vec<ChatMessage>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Provider: {}", provider.name());
    println!("Default model: {}", provider.default_model());

    println!("\n=== Messages to LLM ===");
    for msg in &messages {
        msg.pretty_print();
    }
    println!("=== Sending... ===\n");

    match provider.chat(messages).await {
        Ok(response) => {
            // Strip markdown code fences if present
            let json_str = extract_json(&response);

            // Try to parse as Program
            match serde_json::from_str::<llcraft_vm::Program>(json_str) {
                Ok(program) => {
                    println!("✓ Successfully parsed!\n");
                    program.pretty_print();
                }
                Err(e) => {
                    println!("=== RAW RESPONSE ===");
                    println!("{}", response);
                    println!("\n✗ Parse error: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}

/// Extract JSON from a response that might have markdown code fences
fn extract_json(response: &str) -> &str {
    let trimmed = response.trim();

    // Check for ```json ... ``` or ``` ... ```
    if trimmed.starts_with("```") {
        // Find the end of the first line (after ```json or ```)
        if let Some(start) = trimmed.find('\n') {
            let rest = &trimmed[start + 1..];
            // Find the closing ```
            if let Some(end) = rest.rfind("```") {
                return rest[..end].trim();
            }
        }
    }

    trimmed
}
