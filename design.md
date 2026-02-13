# LLM-VM: A Virtual Machine for LLM Agents

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  PROCESS TREE                                        │
│  Each process runs a program. Any process can SPAWN  │
│  child processes. The interpreter runs them.          │
│                                                      │
│  Process 0 (root)                                    │
│  ├── Process 1 (child)                               │
│  │   ├── Process 3                                   │
│  │   └── Process 4                                   │
│  └── Process 2 (child)                               │
├──────────────────────────────────────────────────────┤
│  INTERPRETER                                         │
│  Reads opcodes, steps through a program              │
│  SPAWN creates a child. JOIN runs it to completion.  │
│  It's just recursive interpretation — no scheduler.  │
├──────────────────────────────────────────────────────┤
│  STATE (per process)                                 │
│  Memory  = pages (text chunks loaded into context)   │
│  Stack   = values (control flow + data)              │
│  Storage = KV store (persistence across processes)   │
└──────────────────────────────────────────────────────┘
```

### Memory (Pages)

Pages are the only unit of context. Each page is a named chunk of text with a token count. Before every `CALL`, the VM assembles the context window from whatever pages are currently loaded — like a CPU reading from memory.

### Stack

A simple value stack for control flow and data passing. Opcodes push and pop values. `CALL` pushes the LLM response. `BRANCH` pops a condition. `SPAWN` pushes a child process ID. `JOIN` pushes the child's result.

### Storage

A persistent KV store. Survives across processes. Shared between parent and child processes (read snapshot on spawn, write-back on completion). This replaces shared memory, IPC, and message passing.

---

## Instruction Set (16 opcodes)

```
MEMORY (static)    MEMORY (dynamic)    STACK
───────────────    ────────────────    ─────
LOAD  "page"       LOAD_S              PUSH  value
STORE "page"       STORE_S             POP
FREE  "page"       FREE_S

CONTROL            LLM       SYSCALL
───────            ───       ───────
JUMP   label       CALL      SYSCALL tool args
BRANCH label
HALT

PROCESS
───────
SPAWN  task
JOIN   pid
JOIN_ALL
```

### Addressing model

Opcodes that reference pages support two addressing modes:

- **Static**: page name baked into the opcode — `LOAD "task_context"`
- **Dynamic**: page name popped from the stack — `PUSH "task_context"` then `LOAD_S`

Static addressing is for programs with known structure. Dynamic addressing is essential for agent-driven workflows where the LLM decides what to load at runtime.

Labels (`JUMP`, `BRANCH`) are always static — control flow is determined by the program.

Stack addressing is always implicit (top-of-stack).

### What each opcode does

| Opcode | Effect |
|--------|--------|
| `LOAD "page"` | Load a named page into the context window |
| `STORE "page"` | Write stack top to a named page |
| `FREE "page"` | Remove a named page from context |
| `LOAD_S` | Pop page name from stack, load that page |
| `STORE_S` | Pop page name from stack, write stack top to that page |
| `FREE_S` | Pop page name from stack, remove that page from context |
| `PUSH value` | Push a value onto the stack |
| `POP` | Discard top of stack |
| `JUMP label` | Go to a labeled instruction |
| `BRANCH label` | Pop value, jump if truthy |
| `HALT` | Stop execution, top of stack is the result |
| `CALL` | Pop instruction from stack, assemble context (system prompt + loaded pages + instruction + tools), run LLM tool-use loop until done, push final result |
| `SYSCALL tool args` | Invoke external tool, push result |
| `SPAWN task` | Create child process from task string (child runs CALL with tools), push pid |
| `JOIN pid` | Run child to completion, push its result (like a function call) |
| `JOIN_ALL` | Run all pending children, store each result as a page (`_result_<pid>`), push number of results |

---

## Processes & Memory Isolation

A process is one running instance of a program with its own stack, program counter, and page set. Memory isolation follows **fork semantics**:

- `SPAWN` = child inherits a **snapshot** of parent's pages (copy-on-fork)
- Internally uses **copy-on-write** — child shares parent's page data until it writes, then copies. This is an invisible optimization; the program doesn't need to know.
- Child can `LOAD`/`STORE`/`FREE` freely — changes are private, don't affect parent
- Siblings can't see each other's pages
- Parent doesn't see child's new/modified pages
- Results come back only through `JOIN` (top of child's stack)

```
Process 0 (pages: [task, code, spec])
├── Process 1 → inherits [task, code, spec], adds [api_draft]
│   ├── Process 3 → inherits [task, code, spec, api_draft]
│   └── Process 4 → inherits [task, code, spec, api_draft]
├── Process 2 → inherits [task, code, spec], adds [test_plan]
│
│   Process 1 and Process 2 can't see each other's pages.
│   Process 3 can see api_draft (inherited from P1),
│   but Process 2 cannot.
└── JOIN_ALL → collect results
```

No MMU, no virtual addresses, no page tables, no shared libraries. Just: children see parent's pages, siblings don't see each other. Shared read-only pages (like shared libraries) are a future optimization if token duplication becomes a problem — copy-on-write handles this cheaply for now.

### How SPAWN works

`SPAWN` takes a task description (string), not a program reference. The VM creates a child process with a bootstrap program:

```
SPAWN "design the database schema"    ; in the parent

; VM creates child with this bootstrap program:
;   PUSH "design the database schema"
;   CALL
;   HALT
```

The child inherits the parent's pages (fork snapshot) and starts as an autonomous agent — its `CALL` runs a tool-use loop where the LLM reads files, invokes commands, spawns sub-agents, etc. until it calls `done`. No program registry, no lookup tables, no pre-defined programs.

### How JOIN works

`JOIN pid` runs the child to completion and pushes the child's result (top of child's stack at HALT) onto the parent's stack as a **value**.

To make the result visible to the next `CALL`, the parent must store it as a page:

```
  SPAWN "design the API"         ; push pid=1
  JOIN 1                          ; run child, push result
  STORE "api_result"              ; save to page → now visible to CALL
  LOAD "api_result"               ; load into context window
```

### How JOIN_ALL works

`JOIN_ALL` runs all pending children. For each child, it stores the result as a page named `_result_<pid>` and loads it automatically. Pushes the number of results onto the stack.

```
  SPAWN "design DB"               ; pid=1
  SPAWN "design API"              ; pid=2
  SPAWN "design frontend"         ; pid=3
  JOIN_ALL
  ; VM creates pages: _result_1, _result_2, _result_3
  ; VM loads all three pages
  ; stack: [3]  (number of results)
  PUSH "combine these into a final design"
  CALL                             ; LLM sees all result pages
```

This solves the problem of results being invisible to CALL — `JOIN_ALL` auto-stores and auto-loads.

Sequential by default. Parallelism is an **optimization** — the interpreter can run children concurrently when resources allow, but the semantics are the same.

### Coordination

Processes are isolated by default (own page snapshot + stack). Results flow back only through `JOIN`. Storage is namespaced per process tree.

```
Parent:                          Child:
  LOAD "auth_code"                 ; inherits all parent pages
  PUSH "analyze auth"              ; including "auth_code"
  SPAWN analyze_prog               CALL
  ; gets pid=3                     STORE "findings" result
  ...                              HALT
  JOIN 3
  ; gets child's result
  ; child's pages are freed
  ; parent's pages unchanged
```

When a child completes, its pages are freed. Results flow back only through `JOIN` (top of child's stack) or storage writes under its namespace.

---

## How CALL Works

`CALL` is the one special opcode — it invokes the LLM. Like a real CPU, it doesn't take explicit arguments. It reads the current machine state:

```
 CALL reads:                         CALL writes:
 ┌─────────────────────────────┐     ┌──────────────────────┐
 │ 1. System prompt (firmware)  │     │ Final result → stack  │
 │    Always resident, never    │     └──────────────────────┘
 │    paged out                 │
 │ 2. Loaded pages (memory)     │
 │    Whatever LOAD put there   │
 │ 3. Top of stack (instruction)│
 │    Popped as the prompt      │
 │ 4. Tool definitions          │
 │    VM capabilities as tools  │
 └─────────────────────────────┘
```

The VM handles fitting everything into the token budget:

```
1. Pop instruction from stack
2. Compute budget = model_limit - system_prompt - instruction - tool_defs
3. Fill remaining budget with loaded pages (LRU eviction if over)
4. Send [system_prompt | pages | instruction | tools] to LLM
5. Run tool-use loop (see below)
6. Push final result onto stack
```

### CALL uses native function calling (tools)

`CALL` does **not** ask the LLM to emit custom opcodes or structured JSON. Instead, it uses the LLM's native **function calling / tool use** API — the same mechanism models are already trained on.

The VM exposes its capabilities as tools:

```json
{
  "tools": [
    {"name": "read_file",    "parameters": {"path": "string"}},
    {"name": "write_file",   "parameters": {"path": "string", "content": "string"}},
    {"name": "run_command",  "parameters": {"command": "string"}},
    {"name": "spawn_agent",  "parameters": {"task": "string"}},
    {"name": "store_page",   "parameters": {"name": "string", "content": "string"}},
    {"name": "load_page",    "parameters": {"name": "string"}},
    {"name": "done",         "parameters": {"result": "string"}}
  ]
}
```

The LLM responds with standard tool calls. The **VM translates each tool call into internal opcodes** — the LLM never sees opcodes.

#### Example: "Read auth.rs and find the bug"

```
; Parent program
PUSH "read auth.rs and find the bug"
CALL
```

Turn 1 — LLM calls a tool:
```json
{"tool_calls": [{"name": "read_file", "arguments": {"path": "auth.rs"}}]}
```
VM translates → `SYSCALL read_file "auth.rs"`, auto-stores result as page `"auth_code"`, auto-loads it, re-invokes CALL so the LLM sees the file content.

Turn 2 — LLM has seen the file, calls done:
```json
{"tool_calls": [{"name": "done", "arguments": {"result": "Bug on line 42: missing null check"}}]}
```
VM translates → `PUSH "Bug on line 42: missing null check"`, then falls through to next opcode.

#### Example: multi-agent delegation

```json
{"tool_calls": [
  {"name": "spawn_agent", "arguments": {"task": "review auth.rs for security issues"}},
  {"name": "spawn_agent", "arguments": {"task": "review auth.rs for performance issues"}}
]}
```
VM translates → `SPAWN "review security..."`, `SPAWN "review perf..."`, `JOIN_ALL`.

#### Why tools instead of custom opcodes

| Aspect | Custom opcodes (old) | Native function calling |
|--------|---------------------|------------------------|
| LLM prompt | "Here are 16 opcodes, emit JSON arrays" | Standard tools the model already knows |
| Reliability | Fragile — model must learn custom ISA | Robust — models are trained on tool use |
| Execution | Single-shot → LLM emits batch of opcodes | Multi-turn → LLM calls tools, sees results, decides next step |
| VM role | Dumb executor of LLM-generated programs | **Translator** — maps tool calls to internal opcodes |

The VM architecture stays the same — pages, stack, processes, fork isolation, SPAWN/JOIN. The only change is the CALL interface: the LLM uses tools, and the VM compiles tool calls into opcodes internally.

### CALL is a multi-turn loop

Unlike the old single-shot design, `CALL` now runs a **tool-use loop**:

```
1. Pop instruction from stack
2. Compute budget = model_limit - system_prompt - instruction
3. Fill remaining budget with loaded pages (LRU eviction if over)
4. Send [system_prompt | pages | instruction | tools] to LLM
5. If LLM returns tool_calls:
   a. Execute each tool call (translate to opcodes, run them)
   b. Append tool results to conversation
   c. Go to step 4 (re-invoke LLM with updated context)
6. If LLM calls "done" tool → push result onto stack, exit CALL
7. If LLM returns plain text (no tool call) → push text onto stack, exit CALL
```

This means a single `CALL` can do multiple rounds of work — read files, run commands, store pages — before returning a final result. The program doesn't grow at runtime; the LLM's multi-turn reasoning stays inside the CALL boundary.

### Static vs agent-driven programs

Static programs have all steps known upfront — no `CALL` or only `CALL` with `done`:

```
Static:                          Agent-driven:
  LOAD "context"                   LOAD "task_context"
  PUSH "summarize this"            PUSH "find and fix the bug"
  CALL                             CALL
  ; LLM calls done(result)         ; LLM reads files, runs tests,
  ; result pushed to stack         ; spawns agents — all inside CALL
  STORE "summary"                  ; eventually calls done(result)
  HALT                             STORE "result"
                                   HALT
```

Agent-driven programs are simpler than before: the LLM handles its own tool-use loop inside CALL, so the program itself can stay short.

### BRANCH conditions

`BRANCH label` pops the top of stack and jumps if truthy:

- **Truthy** (jump taken): `"true"`, non-empty string, non-zero number
- **Falsy** (fall through): `"false"`, `""`, `0`, `null`

### Loop pattern

Agents that need "keep trying until done" use CALL + BRANCH:

```
  loop:
    LOAD "task"
    PUSH "do the next step toward completing the task"
    CALL                           ; LLM uses tools, returns result
    STORE "progress"               ; save what was done
    LOAD "progress"
    PUSH "is there more work? respond with just true or false"
    CALL                           ; LLM responds "true" or "false"
    BRANCH loop                    ; loop if more work
  HALT
```

`BRANCH` = jump if truthy. The LLM controls the condition by responding "true" or "false" (via plain text or the `done` tool).

Works with any context window size — 8k, 128k, 1M+. The VM adapts automatically.

---

## Why This Instead of Full OS Abstractions

| Full OS concept | Why we drop it | What replaces it |
|----------------|----------------|-----------------|
| Registers, heap, kernel space | Unnecessary indirection | Pages (one memory type) |
| MMU, page tables, virtual addressing | No hardware to protect | Fork-style page snapshots (children inherit, siblings isolated) |
| 6 process states | Over-engineered | No process states — just recursive interpretation |
| Shared mutable memory | Race conditions | Storage snapshots |
| SEND/RECV message passing | Complex protocol | Return values via JOIN |
| Preemptive scheduling | No untrusted code | No scheduler — JOIN runs children directly |

---

## What This Enables

1. **Recursive delegation** — Processes spawn processes, forming a tree
2. **Any scale** — 1 process or 1000, sequential or parallel
3. **Any context window** — VM adapts page loading to fit
4. **Long-running tasks** — Processes persist state to storage, resume later
5. **Deterministic replay** — Program is a sequence of opcodes, tool calls are logged, fully traceable
6. **Composable programs** — Programs call programs via SPAWN/JOIN
7. **Native tool use** — LLM uses standard function calling, VM translates to opcodes internally

---

## Why Current Approaches Fall Short

| Approach | Problem | LLM-VM Solution |
|----------|---------|-----------------|
| RAG | Blind retrieval, no write-back | Read/write pages with eviction |
| Agent loops | Flat prompt + heuristics, no isolation | Structured programs with process isolation and branches |
| Multi-agent | Ad-hoc orchestration, no isolation | Process tree, recursive interpretation |
| Long context | Dump everything in, hope for best | Load what's needed, evict the rest |

---

## Next Steps

1. Align codebase with this simplified architecture
2. Implement CALL as multi-turn tool-use loop
3. Define tool schema (read_file, write_file, run_command, spawn_agent, store_page, load_page, done)
4. Implement process tree (SPAWN/JOIN) in interpreter
5. Automatic context window packing with LRU eviction
6. Run real multi-agent task and measure token usage vs. naive approach
7. Target: 10× context reduction with same task success rate
