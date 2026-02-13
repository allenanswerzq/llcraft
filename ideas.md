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

No heap, no registers, no kernel space. Just pages in or out of the context window.

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
BRANCH cond label
HALT

PROCESS
───────
SPAWN  program args
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
| `BRANCH cond label` | Pop condition, jump if true |
| `HALT` | Stop execution, top of stack is the result |
| `CALL` | Pop instruction from stack, assemble context (system prompt + loaded pages + instruction), send to LLM, push response |
| `SYSCALL tool args` | Invoke external tool, push result |
| `SPAWN program args` | Create child process, push pid (child doesn't run yet) |
| `JOIN pid` | Run child to completion, push its result (like a function call) |
| `JOIN_ALL` | Run all pending children, push results (parallelizable) |

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

### How SPAWN/JOIN work

`SPAWN` is lazy — it creates the child process (program + args + snapshot of parent's pages) and pushes its pid. The child doesn't run until `JOIN`.

`JOIN pid` is where execution happens. The interpreter pauses the parent, runs the child process to completion (recursively interpreting its opcodes), then pushes the child's result onto the parent's stack. This is just a function call.

`JOIN_ALL` runs all pending children. Sequential by default. Parallelism is an **optimization** — the interpreter can run children concurrently when resources allow, but the semantics are the same either way.

```
Parent program:                  What the interpreter does:
─────────────────                ──────────────────────────
  SPAWN analyze_prog             → create child, push pid=1
  SPAWN test_prog                → create child, push pid=2
  JOIN_ALL                       → run child 1 to HALT
                                 → run child 2 to HALT
                                 → push [result1, result2]
```

No scheduler. No ready queue. No blocked state. Just recursive interpretation.

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
 │ 1. System prompt (firmware)  │     │ LLM response → stack │
 │    Always resident, never    │     └──────────────────────┘
 │    paged out                 │
 │ 2. Loaded pages (memory)     │
 │    Whatever LOAD put there   │
 │ 3. Top of stack (instruction)│
 │    Popped as the prompt      │
 └─────────────────────────────┘
```

The VM handles fitting everything into the token budget:

```
1. Pop instruction from stack
2. Compute budget = model_limit - system_prompt - instruction
3. Fill remaining budget with loaded pages (LRU eviction if over)
4. Send [system_prompt | pages | instruction] to LLM
5. Push response onto stack
```

This means the program controls what the LLM sees by loading/freeing pages and pushing the right instruction before `CALL`. The VM just assembles the context and manages the token budget.

```
Example program:                What the LLM sees:
────────────────                ───────────────────
  LOAD "task_context"            system prompt
  LOAD "code_snippet"           + task_context page
  PUSH "find the bug"           + code_snippet page
  CALL                           + "find the bug"
  ; stack now has the answer     → response pushed to stack
```

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
5. **Deterministic replay** — Program is a sequence of opcodes, fully traceable
6. **Composable programs** — Programs call programs via SPAWN/JOIN

---

## Why Current Approaches Fall Short

| Approach | Problem | LLM-VM Solution |
|----------|---------|-----------------|
| RAG | Blind retrieval, no write-back | Read/write pages with eviction |
| Agent loops | Flat prompt + heuristics, fragile | Structured programs with branches |
| Multi-agent | Ad-hoc orchestration, no isolation | Process tree, recursive interpretation |
| Long context | Dump everything in, hope for best | Load what's needed, evict the rest |

---

## Next Steps

1. Align codebase with this simplified architecture
2. Implement process tree (SPAWN/JOIN) in interpreter
3. Automatic context window packing with LRU eviction
4. Run real multi-agent task and measure token usage vs. naive approach
5. Target: 10× context reduction with same task success rate
