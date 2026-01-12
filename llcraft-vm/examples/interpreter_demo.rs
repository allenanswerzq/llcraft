//! # Interpreter Demo
//!
//! Demonstrates running programs through the LLcraft VM interpreter.

use llcraft_vm::{
    Interpreter, DefaultSyscallHandler, ExecutionResult,
    Opcode, Program, LogLevel,
};

fn main() {
    println!("=== LLcraft VM Interpreter Demo ===\n");

    // Example 1: Simple program that reads a file and stores the result
    demo_file_listing();

    // Example 2: Program with branching logic
    demo_branching();

    // Example 3: Stack operations
    demo_stack_ops();
}

fn demo_file_listing() {
    println!("--- Demo 1: File Listing ---");

    let program = Program::new(
        "list_files",
        "List files in current directory",
        vec![
            Opcode::Label { name: "start".to_string() },
            Opcode::Log {
                level: LogLevel::Info,
                message: "Listing files in current directory...".to_string()
            },
            Opcode::Syscall {
                call: "list_dir".to_string(),
                args: serde_json::json!({"path": "."}),
                store_to: Some("files".to_string()),
            },
            Opcode::Branch {
                condition: "files.success".to_string(),
                if_true: "success".to_string(),
                if_false: "failure".to_string(),
            },
            Opcode::Label { name: "success".to_string() },
            Opcode::Log {
                level: LogLevel::Info,
                message: "Files listed successfully!".to_string(),
            },
            Opcode::Complete {
                result: serde_json::json!({
                    "status": "ok",
                    "files_page": "files"
                }),
            },
            Opcode::Label { name: "failure".to_string() },
            Opcode::Fail { error: "Failed to list files".to_string() },
        ],
    );

    println!("Program:");
    program.pretty_print();
    println!();

    let mut interp = Interpreter::new(program, DefaultSyscallHandler::default())
        .with_log_callback(|level, msg| {
            println!("  [{:?}] {}", level, msg);
        });

    match interp.run() {
        Ok(ExecutionResult::Complete(result)) => {
            println!("✓ Completed: {}", serde_json::to_string_pretty(&result).unwrap());
            if let Some(files) = interp.get_page("files") {
                println!("  Files page: {}", serde_json::to_string_pretty(files).unwrap());
            }
        }
        Ok(ExecutionResult::Failed(error)) => {
            println!("✗ Failed: {}", error);
        }
        Ok(ExecutionResult::NeedsLlm(request)) => {
            println!("? Needs LLM: {:?}", request.request_type);
        }
        Ok(ExecutionResult::StepLimitExceeded) => {
            println!("! Step limit exceeded");
        }
        Err(e) => {
            println!("! Error: {}", e);
        }
    }

    println!("\nExecution trace:");
    for step in interp.trace() {
        println!("  [{}] {} -> {}", step.step, step.opcode, step.result);
    }
    println!();
}

fn demo_branching() {
    println!("--- Demo 2: Branching Logic ---");

    let program = Program::new(
        "branching_demo",
        "Demonstrate conditional branching",
        vec![
            Opcode::Store {
                page_id: "user".to_string(),
                data: serde_json::json!({
                    "name": "Alice",
                    "admin": true,
                }),
            },
            Opcode::Branch {
                condition: "user.admin".to_string(),
                if_true: "admin_path".to_string(),
                if_false: "user_path".to_string(),
            },
            Opcode::Label { name: "admin_path".to_string() },
            Opcode::Store {
                page_id: "result".to_string(),
                data: serde_json::json!({"access": "full", "role": "admin"}),
            },
            Opcode::Jump { target: "done".to_string() },
            Opcode::Label { name: "user_path".to_string() },
            Opcode::Store {
                page_id: "result".to_string(),
                data: serde_json::json!({"access": "limited", "role": "user"}),
            },
            Opcode::Label { name: "done".to_string() },
            Opcode::Complete {
                result: serde_json::json!({"result_page": "result"}),
            },
        ],
    );

    let mut interp = Interpreter::new(program, DefaultSyscallHandler::default());

    match interp.run() {
        Ok(ExecutionResult::Complete(_)) => {
            if let Some(result) = interp.get_page("result") {
                println!("✓ Result: {}", serde_json::to_string_pretty(result).unwrap());
            }
        }
        Ok(other) => println!("Unexpected: {:?}", other),
        Err(e) => println!("Error: {}", e),
    }
    println!();
}

fn demo_stack_ops() {
    println!("--- Demo 3: Stack Operations ---");

    let program = Program::new(
        "stack_demo",
        "Demonstrate stack operations",
        vec![
            // Push some values
            Opcode::Push { value: serde_json::json!(1) },
            Opcode::Push { value: serde_json::json!(2) },
            Opcode::Push { value: serde_json::json!(3) },
            // Get stack depth
            Opcode::Depth { store_to: "depth".to_string() },
            // Peek at top
            Opcode::Peek { store_to: "top".to_string() },
            // Pop and store
            Opcode::PopTo { store_to: "popped".to_string() },
            // Swap remaining
            Opcode::Swap,
            // Pop both to result
            Opcode::PopTo { store_to: "second".to_string() },
            Opcode::PopTo { store_to: "first".to_string() },
            Opcode::Complete {
                result: serde_json::json!({
                    "depth_was": "depth",
                    "top_was": "top",
                    "popped": "popped",
                    "after_swap_first": "first",
                    "after_swap_second": "second",
                }),
            },
        ],
    );

    let mut interp = Interpreter::new(program, DefaultSyscallHandler::default());

    match interp.run() {
        Ok(ExecutionResult::Complete(_)) => {
            println!("✓ Stack operations completed:");
            for page_id in &["depth", "top", "popped", "first", "second"] {
                if let Some(value) = interp.get_page(page_id) {
                    println!("  {}: {}", page_id, value);
                }
            }
        }
        Ok(other) => println!("Unexpected: {:?}", other),
        Err(e) => println!("Error: {}", e),
    }
    println!();
}
