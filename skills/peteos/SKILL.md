---
name: peteos
description: Write and use agentic objects — Python classes that can think, decide, and act on their own internal state. Use this skill when you need to create intelligent classes, autonomous agents, multi-agent systems, structured LLM output, objects that maintain state across conversation sessions, or add agentic capabilities to existing Python hierarchies via inheritance.
compatibility: opencode
---

## Comprehensive Example

Five minimal classes demonstrating every PeteOS concept. Reference these from the sections below.

```python
from peteos import AgenticObject, tool, sandbox, agentic_object, Error
from dataclasses import dataclass
from typing import Any

# ─────────────────────────────────────────────────
# Record: simple dataclass used across A, B, C
# ─────────────────────────────────────────────────
@dataclass
class Record:
    id: int
    label: str

# ─────────────────────────────────────────────────
# A: Base — @tool with simple and structured args, @sandbox, plain method
# ─────────────────────────────────────────────────
class A(AgenticObject):
    """System prompt A."""

    def __init__(self):
        super().__init__()
        self._state = 0

    @tool
    def f(self, verbose: bool = False) -> Any:
        """Return a simple status message. Pass verbose=True to get a detailed reply. Return type is unstructured."""
        return "A.f" if verbose else "ok"

    @tool(name="process_items", description="Process labeled data. x is an optional label, y is a list of integers, z maps labels to value lists. Returns a summary.")
    def g(self, x: str | None, y: list[int], z: dict[str, list]) -> str:
        """Process a batch of labeled items."""
        return f"got {x}, {len(y)} items"

    @sandbox(description="Compute statistics on a list of floats. Returns a numpy array for further computation.")
    def _s(self, data: list[float]) -> np.ndarray:
        """Compute statistics on a list of numbers. Returns a numpy array for further computation."""
        import numpy as np
        return np.array(data)

    def _p(self) -> str:
        return "internal"

@agentic_object(allow_code_execution=True)
class B(A, AgenticObject):
    """System prompt B."""

    @tool
    def f(self, data: dict[str, int | str], rec: Record) -> list[dict]:
        """Transform a dictionary into a list of records. rec provides a Record to attach to each entry. Returns a list of dicts, one per entry."""
        return [{"k": k, "v": v, "rec": {"id": rec.id, "label": rec.label}} for k, v in data.items()]

    @tool
    def h(self, tag: str = "misc") -> None:
        """Log a message or tag. Returns nothing — the operation is complete when called."""
        print(f"[{tag}] done")
        return

    @sandbox
    def _s2(self, matrix: np.ndarray) -> np.ndarray:
        """Invert a 2D numpy matrix. Returns the inverted matrix."""
        import numpy as np
        return np.linalg.inv(matrix)

# ─────────────────────────────────────────────────
# C: Composes A and B — shows full tool collection from MRO
# ─────────────────────────────────────────────────
@agentic_object(invoke_sub_agents=True)
class C(A, B, AgenticObject):
    """System prompt C."""                           # 3rd in reverse-MRO: C, B, A
    # Inherits: g (from A), h (from B), _s (from A), _s2 (from B)
    # Prompt concatenates: AgenticObject (empty) + B + A + C  (reverse MRO)

    @tool
    def f(self, data: dict[str, int | str], rec: Record) -> Record:
        """Override of A.f and B.f. Guard-check input, delegate to A.f, then attach record to result. Returns a Record."""
        if not data:
            raise ValueError("data must not be empty")
        A.f(self)
        return Record(id=rec.id, label=f"processed: {rec.label}")

# ─────────────────────────────────────────────────
# D: NOT agentic — missing direct AgenticObject
# ─────────────────────────────────────────────────
class D(A):
    # D is NOT recognized as agentic — no @tool methods exposed
    # Every class in the hierarchy needing agency must directly inherit AgenticObject
    pass

# ─────────────────────────────────────────────────
# E: No code execution — @tool only, @sandbox unreachable
# ─────────────────────────────────────────────────
class E(AgenticObject):
    """System prompt E — no code execution."""

    @tool
    def q(self, data: list[dict]) -> dict[str, int]:
        """Count items in a list of records. Returns a dict with 'count' (total items) and 'items' (sum of something)."""
        return {"count": len(data), "items": sum(1 for d in data)}
```

## Decorator Hierarchy

| Level | Decorator | Agent can | Examples in A, B |
|---|---|---|
| Hidden | none | Cannot call | `_p` in A — internal only |
| Indirect | `@sandbox` | Call from agent-written Python | `_s` in A, `_s2` in B |
| Direct | `@tool` | Decide autonomously | `f`, `g` in A; `f`, `h` in B |

**`@sandbox` with non-serializable args** (`B._s2`): Use for file handles, open connections, or types the LLM cannot represent. The agent writes Python that creates and passes the handle internally — the LLM never needs to provide it as a serializable argument.

**`@tool` with structured args** (`A.g`, `B.f`): The agent can supply `str`, `list`, `dict`, etc. directly as tool call arguments. Use `list[T]`, `dict[K, V]` for generics.

Escalate exposure only when the agent genuinely needs to decide autonomously. Prefer `@sandbox` for composable helpers the agent calls from Python — one `@tool` that loops via a sandbox helper beats many individual `@tool` calls.

## @agentic_object Decorator Arguments

The `@agentic_object` decorator configures what an agent can do. All arguments are optional booleans defaulting to `False` unless noted:

| Argument | Default | What it enables |
|---|---|---|
| `allow_code_execution` | `False` | Agent writes and runs Python. Inside that code, `self` refers to the agentic object, so `@sandbox` and `@tool` methods are callable. |
| `invoke_sub_agents` | `False` | Agent can delegate tasks to other agentic objects via `self.invoke(target=...)` from within a `@tool` method. |
| `define_functions` | `False` | Agent can register new `@tool` methods on the object at runtime. |
| `role` | `None` | Override the agent's role name. Takes a string. |

Boolean flags OR together across the MRO — if any class sets one to `True`, the combined config has it as `True`. The `imports` set is unioned across all parents, and `import_aliases` dict is merged with later entries overriding earlier.

## Argument and Return Types

Each `@tool` in the example demonstrates different argument and return patterns:

| Function | Arg pattern | Return pattern |
|---|---|---|
| `A.f` | Optional with default `bool = False` — agent can omit | `str` |
| `A.g` | Union `str \| None`, multi-structured `list[int]`, `dict[str, list]` | `str` |
| `B.f` | Dict with union values `dict[str, int \| str]` | `list[dict]` — structured |
| `B.h` | Optional `str = "misc"`, returns `None` | `None` — agent gets no value |
| `E.q` | Structured `list[dict]`, no code exec | `dict[str, int]` |

**Union args** (`x: str | None`) — agent can pass the type or None.

**Default args** (`verbose: bool = False`) — agent may skip the argument entirely.

**`return None`** — the agent receives nothing. Use this when the action is stateful and the agent should proceed without needing the return value to decide. If the agent needs feedback, return a summary instead.

**Structured returns** (`list[dict]`, `dict[str, int]`) — the agent receives the parsed structure and can reason over it.

## Multi-Inheritance Patterns

Classes `B` and `C` demonstrate all three patterns:

**Specialization (B overrides A.f):** B inherits A's tools but overrides `f`. MRO resolves to B's version. The system prompt is still B's own docstring.

**Extension (B adds h):** B introduces `h` — a new tool A never had. Tools are collected from the full MRO.

**Composition (C inherits A and B):** C inherits all tools from both ancestors. Tool lookup follows MRO, so C gets `f` from B (the specialized version), `g` from A, `h` from B.

### System Prompt Concatenation

Docstrings concatenate in **reverse MRO order** (base classes first, most-derived last).

For `class C(A, B, AgenticObject)`, MRO is `[C, A, B, AgenticObject]`. Reversed for prompt: `AgenticObject` (empty), `B`, `A`, `C`. Result: the agent sees `B`'s docstring, then `A`'s, then `C`'s. The most-derived class's prompt comes **last**, so it has the final word.

**Direct inheritance requirement:** Class `D` shows that inheriting through an intermediate parent is not enough — every class in the hierarchy that needs to be agentic must directly inherit `AgenticObject`.

### Flags OR Together

`B` shows `@agentic_object(allow_code_execution=True)` — this flag was already `True` from `A`. Boolean flags OR together across the MRO — if any class sets one to `True`, the combined config has it as `True`. The `imports` set is unioned across all parents.

## Error Handling

`invoke_agent` returns `Error` on failure, never raises:

```python
result = await agent.invoke_agent("...")
if isinstance(result, Error):
    print(f"Failed: {result.message}")   # handle gracefully
else:
    print(result)                        # use the result
```

API-level failures (connection, auth) still raise exceptions.

## Invocation Hooks

Inspect or control lifecycle:

```python
await agent.invoke_agent(
    "Do the thing",
    hooks={
        "on_invoke": [lambda ctx: None],                  # return non-None to abort
        "on_tool_call": [lambda ctx: None],              # return non-None to deny
        "before_tool_execution": [lambda tc: None],     # return (False, "reason") to deny
        "after_tool_execution": [lambda r, tc, s, ok: None],
        "before_send_to_chatbot": [lambda r, ctx: None],
        "after_step": [lambda r, status: None],         # return ExecStatus to override
        "after_message_append": [lambda r, msg: None],
        "before_notification_publish": [lambda r, msg: None],
        "on_truncation_exhausted": [lambda c, m: None], # (counter, max_retries)
        "on_invoke_complete": [lambda ctx: None],
    },
)
```

Hooks propagate to sub-agents via `invoke()`. They do not propagate when an agent calls another agent's `invoke_agent()` from within a tool.

## Sub-Agent Delegation

```python
class Supervisor(AgenticObject):
    def __init__(self):
        super().__init__()
        self.worker = Worker()

    @tool
    async def delegate(self, task: str) -> dict:
        result = await self.invoke(
            target=self.worker,
            prompt=task,
            persistent=True,   # inherit parent's thread ID
        )
        return {"result": result}
```

Use `persistent=False` for a fresh session on the sub-agent. **Avoid circular A→B→A** — it deadlocks. Pass `timeout=` to `invoke_agent`.

## Media Handling

Images reach the agent in two ways:

**1. User provides at invoke time:**
```python
result = await agent.invoke_agent("describe this", image="/path/img.jpg")
```

**2. Developer pushes from within a `@tool` (runner auto-injected):**
```python
@tool
async def send_chart(self, runner) -> str:
    with open("chart.png", "rb") as f:
        await self._send_media(data=f.read(), mime_type="image/png", runner=runner)
    return "sent"
```

## Key Principles

1. **Always call `super().__init__()`** — Every agentic object must call `super().__init__()` in `__init__` to initialize the agentic base. The example class `A` shows this pattern.

2. **Tool descriptions drive behavior** — The agent picks tools based on method docstrings. Be specific about arguments and return values.

3. **Return values guide reasoning** — The agent sees return values and uses them to decide next steps. Return meaningful summaries, not `None`.

4. **Keep `@tool` count lean** — Each `@tool` is a decision point. Use `@sandbox` for high-frequency or composable logic.

5. **Type constraints prevent invalid data** — Enum, dataclass, union, and typed args let the type system block invalid inputs before they reach your code.

6. **State lives on the object** — The agent manipulates state through tools. Persists within a persistent thread session.

7. **Compose over duplication** — Build complex agents from atomic ones via multiple inheritance.

8. **Sandboxing for complex logic** — With `allow_code_execution=True`, the agent can write Python that calls `@sandbox` helpers for iterative or bulk operations.

## Writing Agent-Facing Text

The class docstring is the **system prompt**. Method docstrings are **tool descriptions**. Return values become messages the agent sees. All of this text is written for the agent, not for a developer reading the code.

### General

- **Write facing the agent.** Use "you" to address the agent directly. The agent does not see the source code — it only sees these texts.
- **Concise and instructive.** Every word should help the agent decide or act. Omit implementation noise.
- **Treat as the single source of truth.** The agent has no access to the code. If the description is vague, the agent cannot compensate from elsewhere.

### System Prompts

System prompts concatenate in **reverse MRO order** — each one is a fragment in a larger prompt. Write them to read naturally as a continuation of the prior fragment:

- Write as a short role fragment, not a full standalone prompt. Imagine the prior classes' prompts have already been read.
- Focus on how this class's tools compose with the inherited ones. The agent should know when to use this class's tools in combination with others.
- Avoid repeating anything already established by parent classes.

### Tool Descriptions

The method docstring is what the agent reads when deciding whether to call a tool. It should:

- Describe **what** the tool does and **when** to reach for it.
- Cover inputs (what data to provide), outputs (what comes back), and notable side effects.
- In complicated argument cases, guide argument construction — e.g. which arguments are required vs optional, what format nested data should take.

### Return Values

The return value is the agent's feedback signal. It should:

- Be concise — the agent sees this in its reasoning context.
- Inform the agent what happened or what to do next. If the result is structured data, briefly indicate what the fields represent.
- On error paths, the return value can guide the agent toward a fix — for example, naming the missing field rather than returning a raw exception.

## Configuration

PeteOS auto-loads `peteos.json` on import. Discovery order:

1. `PETEOS_CONFIG` env var (full path, always wins)
2. `./peteos.json` (project-local)
3. `$XDG_CONFIG_HOME/peteos/peteos.json` (user-specific)
4. `/etc/peteos/peteos.json` (system-wide)

### Backend Config

Multiple backends can coexist. The model with the highest integer priority value across all backends wins.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | string | _(required)_ | Unique backend identifier. |
| `url` | string | _(required)_ | API base URL. |
| `api_type` | string | auto-detected | API provider: `"openai"`, `"anthropic"`, or `"gemini"`. |
| `api_key` | string | | API key for authentication. |
| `chat_endpoint` | string | API-specific default | Custom chat endpoint path (e.g. `/chat/completions`). |
| `models_endpoint` | string | API-specific default | Custom models endpoint path. |
| `streaming` | bool | `false` | Use streaming mode by default. |
| `max_tokens` | int | `4096` | Maximum tokens to generate. |
| `model_priorities` | object | `{}` | Map of model IDs to priority integers. Higher values take precedence. |
| `retry_delays` | float[] | `[0, 1, 3]` | Delay in seconds before each retry attempt. The list defines one delay per retry — e.g., `[0, 1, 3]` means: no delay on first retry, 1 second before second, 3 seconds before third. |
| `timeout` | float | provider-specific | HTTP request timeout in seconds. |

```json
{
  "backends": [
    {
      "name": "local",
      "url": "http://localhost:11434"
    },
    {
      "name": "production",
      "url": "https://api.example.com/v1",
      "api_type": "openai",
      "api_key": "sk-...",
      "chat_endpoint": "/chat/completions",
      "models_endpoint": "/models",
      "streaming": true,
      "max_tokens": 4096,
      "model_priorities": {"gpt-4-turbo": 100, "gpt-4": 80},
      "retry_delays": [0, 1, 3],
      "timeout": 30
    }
  ]
}
```

The config's parent directory is the config root, with optional `roles/` and `agents/` subdirs.
