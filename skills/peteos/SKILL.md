---
name: peteos
description: Write and use agentic objects with the Peteos sOAP framework - objects that can think, decide, and act on their own internal state.
license: MIT
compatibility: opencode
metadata:
  framework: peteos
  version: "1.0"
---

## What is Peteos

Peteos is a library for agentic software built on **Simple Object-Agentic Programming (sOAP)**. It combines OOP with AI agents - your Python classes become objects that can reason and act autonomously.

## Core Pattern

Derive from `AgenticObject`, expose methods with `@tool`, and the agent will call them to interact with the object's state:

```python
from peteos import AgenticObject, tool

class Assistant(AgenticObject):
    """You are a helpful assistant."""

    @tool
    def say_hello(self) -> str:
        """Return a greeting."""
        return "Hello!"
```

The class **docstring** is the agent's system prompt. The **method docstring** is the tool description the agent sees when deciding what to call.

**Keep both concise** — they accumulate in the agent's context window and can cause truncation or degraded reasoning. Aim for a few sentences at most. For longer guidance, use a hook like `before_send_to_chatbot` to inject detail on demand.

## Basic Agentic Class

```python
from peteos import AgenticObject, tool

class Greeter(AgenticObject):
    """You are a friendly greeter."""

    def __init__(self):
        super().__init__()
        self._greetings = 0

    @tool
    def greet(self, name: str) -> str:
        """Greet the named person."""
        self._greetings += 1
        return f"Hello, {name}! You have been greeted {self._greetings} time(s)."
```

**Key rules:**
- Always call `super().__init__()` in `__init__`
- `@tool` methods must be on classes inheriting from `AgenticObject`
- Method docstring = tool description for the agent
- Return values are visible to the agent - use them to guide reasoning
- Parameters and return types are inferred from the method signature

## Invoking an Agentic Object

```python
import asyncio

agent = Greeter()
result = await agent.invoke_agent("Say hello to Alice.")
print(result)
```

### Output Schema

Controls what you get back from the agent:

| `output_schema` | What you get |
|---|---|
| `None` | Whatever the agent provided (structured data, text, or `None`) |
| `Any` | JSON is parsed if valid, otherwise plain text |
| `SomeType` | Structured output matching that type |

```python
result = await agent.invoke_agent("Send a notification.", output_schema=None)
# result is the agent's text, or None, or whatever it returned

result = await agent.invoke_agent("Analyze this.", output_schema=Any)
# result is parsed JSON or the raw text

result = await agent.invoke_agent("Return the count.", output_schema=int)
# result is an int matching the schema
```

Supported types: `str`, `int`, `float`, `bool`, `list`, `list[T]`, `dict`, `dict[K, V]`, `tuple`, `Enum`, `dataclass`, and nested combinations.

### Persistent Sessions

```python
await agent.invoke_agent(
    "Remember: my favorite color is blue.",
    persistent_thread_id="user-123",
)
result = await agent.invoke_agent(
    "What is my favorite color?",
    persistent_thread_id="user-123",
)
```

The same `persistent_thread_id` continues the conversation. Omit it for anonymous, temporary sessions.

## Constraining with Enums and Dataclasses

Use `Enum` to constrain valid tool arguments:

```python
from enum import Enum

class Item(Enum):
    MILK = "Milk"
    BREAD = "Bread"
    EGG = "Egg"

class GroceryList(AgenticObject):
    """You manage a grocery list."""

    def __init__(self):
        super().__init__()
        self._items: dict[Item, int] = {}

    @tool
    def add_item(self, item: Item, quantity: int) -> str:
        """Add an item to the grocery list."""
        self._items[item] = self._items.get(item, 0) + quantity
        return f"Added {quantity} {item.value}."
```

The agent cannot pass invalid enum values - the type system blocks them.

## Decorator Exposure Hierarchy

Methods are either **agent-facing** (visible to the agent for autonomous decision-making) or **internal** (hidden from the agent).

There are three levels of exposure. Prefer the lowest level that satisfies your need:

| Level | Decorator | When to use |
|---|---|---|
| **Lowest** | None | Internal — only callable by other decorated methods |
| **Medium** | `@sandbox` | Agent calls via Python it writes — hidden from direct selection |
| **Highest** | `@tool` | Agent decides autonomously — use sparingly |

**Principle: Escalate exposure only when necessary.** The `@tool` decorator gives the agent direct agency over that action. Only reach for it when the agent genuinely needs to decide whether to call it.

### No decorator - Internal helpers

Plain methods are invisible to the agent. Use for pure implementation logic composed by other methods:

```python
def _compute_stats(self, data: list) -> tuple[float, float]:
    """Internal helper - not exposed to the agent."""
    import statistics
    return (statistics.mean(data), statistics.stdev(data))
```

### `@sandbox` - Indirect execution

Use when the agent should call the function from within Python code it writes, rather than selecting it directly as a tool. Prefer `@sandbox` over `@tool` when:

- **Non-serializable args or returns** - the function uses types the LLM cannot represent (e.g., open handles, complex objects)
- **High call frequency** - the agent will loop; one `@tool` call that invokes sandbox helpers many times beats many individual `@tool` calls
- **Composable logic** - the function is a building block the agent uses inside larger Python expressions

```python
@sandbox
def _sum(self, items: list[float]) -> float:
    """Sum a list. Callable only from within agent-written Python."""
    return sum(items)
```

The agent writes Python like `self._sum([1, 2, 3])` - it makes one tool call but the helper runs many times.

### `@tool` - Direct agent action (use sparingly)

Only decorate with `@tool` when the agent must decide **autonomously** whether to take this action. This is the highest exposure level.

```python
@tool
def add(self, a: float, b: float) -> float:
    """Add two numbers and return the sum."""

@tool(name="custom_name")
def old_name(self, x: int) -> int:
    """Custom tool name shown to the agent."""

@tool(description="Explicit description instead of docstring")
def other(self) -> None:
    """This docstring is ignored when description= is set."""
```

Every `@tool` is a decision point the agent reasons about. If the agent does not need to decide on this specifically - if it would only ever be called as part of other logic - that is a `@sandbox` candidate or no decorator at all.

**Warning: Return values feed back into the conversation context.** Each `@tool` return value is embedded as a message the agent sees in subsequent reasoning steps. Large return values - dumping long lists, dicts, or full data structures - accumulate in context and can quickly exhaust the model's context window, causing truncation or crashes. Keep return values lean: return a summary or flag, not the full payload. If you need to return bulk data, prefer `@sandbox` methods the agent calls from within Python, where the data stays in code and does not bloat the conversation context.

### `@agentic_object` - Class-level configuration

```python
from peteos import agentic_object

@agentic_object(
    allow_code_execution=True,    # Enable sandboxed Python execution
    invoke_sub_agents=True,       # Allow delegating to other agentic objects
    allow_media_access=False,     # Allow reading images/PDFs
    role="my-agent",              # Override the role name
)
class MyAgent(AgenticObject):
    ...
```

When composing via multi-inheritance, boolean flags OR together and imports are unioned.

## Sandboxed Code Execution

Enable with `@agentic_object(allow_code_execution=True)`. The agent can write and execute Python code. Within that code, `self` refers to the agentic object, so the agent can call `@tool` and `@sandbox` methods:

```python
@agentic_object(allow_code_execution=True)
class Calculator(AgenticObject):
    """You solve math problems with Python."""

    def __init__(self):
        super().__init__()
        self._numbers = [1, 2, 3, 4, 5]

    @tool
    def get_numbers(self) -> list[int]:
        """Return the current list of numbers."""
        return self._numbers

    @sandbox
    def compute_stats(self) -> tuple[float, float]:
        """Compute mean and standard deviation of the numbers."""
        import statistics
        mean = statistics.mean(self._numbers)
        stdev = statistics.stdev(self._numbers)
        return (mean, stdev)
```

The agent can write Python that calls `self.compute_stats()` to get results.

## Composition - Multi-Inheritance

Build complex agents from simpler ones via inheritance:

```python
@agentic_object(allow_code_execution=True)
class Calculator(AgenticObject):
    """You perform calculations."""

    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

@agentic_object(allow_code_execution=True)
class TextProcessor(AgenticObject):
    """You process and format text."""

    @tool
    def upper(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

class ComboAgent(Calculator, TextProcessor, AgenticObject):
    """You combine calculation and text processing."""
```

Tools are collected from all classes in the MRO. The system prompt is built from docstrings in **reverse MRO order** — base classes first, most-derived class last. For `class ComboAgent(Calculator, TextProcessor, AgenticObject)`, the prompt is AgenticObject's docstring, then TextProcessor's, then Calculator's, then ComboAgent's.

### Direct Inheritance Requirement

A class is recognized as an agentic object only when it **directly** inherits from `AgenticObject` — transitive inheritance through intermediate base classes is not sufficient. Every class in an inheritance hierarchy that needs to be agentic must explicitly include `AgenticObject` as a base, even if a parent class in the chain already inherits it. For example, if `Basher` inherits `AgenticObject` and `SandboxedBasher` inherits `Basher`, PeteOS will not recognize `SandboxedBasher` as an agentic object — it must also inherit `AgenticObject` directly:

```python
class Basher(BufferManager, AgenticObject):
    ...

class SandboxedBasher(Basher, AgenticObject):  # AgenticObject required directly
    ...
```

## Sub-Agent Invocation

With `invoke_sub_agents=True`, a `@tool` method can delegate to another agentic object:

```python
@agentic_object(invoke_sub_agents=True)
class Supervisor(AgenticObject):
    """You delegate tasks to specialists."""

    def __init__(self):
        super().__init__()
        self.specialist = Specialist()

    @tool
    def analyze(self, data: str) -> dict:
        """Analyze data using the specialist."""
        result = await self.invoke(
            target=self.specialist,
            prompt=f"Analyze: {data}",
            persistent=True,
        )
        return {"analysis": result}
```

Use `persistent=True` to inherit the parent's thread ID. Use `persistent=False` for a fresh session on the sub-agent.

**Important:** Avoid circular invocation (A->B->A) - it causes deadlock. Pass `timeout` to `invoke_agent` to prevent indefinite blocking.

## Error Handling

`invoke_agent` returns an `Error` value object on failure - it does **not** raise an exception. Check for it:

```python
from peteos import Error

result = await agent.invoke_agent("...")
if isinstance(result, Error):
    print(f"Failed: {result.message}")
else:
    print(f"Success: {result}")
```

Only API/connection failures raise exceptions. Tool failures return `Error`.

## Invocation Hooks

Inspect or control agent invocations at key lifecycle points. Pass a `hooks` dict to `invoke_agent()`:

```python
result = await agent.invoke_agent(
    "Do the thing.",
    hooks={
        "on_invoke": [lambda ctx: print(f"Invoke: {ctx}")],
        "on_tool_call": [lambda ctx: print(f"Tool: {ctx['tool_name']}")],
        "after_step": [lambda runner, status: print(f"Step: {status}")],
    },
)
```

### All Available Hooks

| Hook | When it fires | Can control? |
|---|---|---|
| `on_invoke` | Before the agent starts reasoning | Yes — return non-None to abort |
| `on_tool_call` | Before a tool is approved for execution | Yes — return non-None to deny |
| `before_tool_execution` | Right before a tool executes | Yes — return `(False, "reason")` to deny |
| `after_tool_execution` | After a tool returns | No |
| `before_send_to_chatbot` | Before the LLM receives the context | No |
| `after_step` | After each reasoning step completes | Yes — return `ExecStatus` to override |
| `after_message_append` | After a message is appended to context | No |
| `before_notification_publish` | Before a notification is published | No |
| `on_truncation_exhausted` | When LLM keeps getting truncated past retry limit | No |
| `on_invoke_complete` | After the invocation finishes | No |

### `on_invoke`

Fires before the agent starts. Return `None` to allow, or a string to abort and return an `Error`.

```python
def guard(ctx):
    print(f"[{ctx['role']}] Prompt: {ctx['prompt']!r}")
    return None  # allow
    # return "Too dangerous"  # abort

result = await agent.invoke_agent(
    "Do the thing.",
    hooks={"on_invoke": [guard]},
)
```

### `on_invoke_complete`

Fires after the invocation finishes, regardless of outcome.

```python
def done(ctx):
    print(f"[{ctx['role']}] Result: {ctx['result']}")

result = await agent.invoke_agent("Do the thing.", hooks={"on_invoke_complete": [done]})
```

### `on_tool_call`

Fires before each tool is approved. Return `None` to allow, or a string to deny and skip remaining tools in the group.

```python
def block_code_exec(ctx):
    if ctx["tool_name"] == "python_exec":
        return "Code execution not allowed"
    return None

result = await agent.invoke_agent(
    "Write a file.",
    hooks={"on_tool_call": [block_code_exec]},
)
```

### `before_tool_execution`

Fires right before a tool executes — after arguments are cast but before the function runs. Return `(False, "reason")` to deny, or `None` to allow.

Note: the hook receives only the `ContentPart` tool-call object, not a `runner` argument.

```python
def watch_execution(tool_call):
    print(f"Running {tool_call.name}")
    return None

result = await agent.invoke_agent(
    "Do the thing.",
    hooks={"before_tool_execution": [watch_execution]},
)
```

### `after_tool_execution`

Fires after a tool returns, whether it succeeded, failed, or returned `None`.

```python
def log_result(runner, tool_call, result_str, success):
    print(f"{tool_call.name} → {result_str} (ok={success})")

result = await agent.invoke_agent(
    "Do the thing.",
    hooks={"after_tool_execution": [log_result]},
)
```

### `before_send_to_chatbot`

Fires before the context is sent to the LLM. Use to inspect or enrich the context.

```python
def log_context(runner, active_context):
    print(f"Context has {len(str(active_context))} chars")
    return None

result = await agent.invoke_agent("Do the thing.", hooks={"before_send_to_chatbot": [log_context]})
```

### `after_step`

Fires after each reasoning step. Return `None` to continue, or an `ExecStatus` to override (e.g., `ExecStatus.CRITICAL` to stop the run loop).

```python
def watch_progress(runner, status):
    print(f"Step ended: {status}")
    return None

result = await agent.invoke_agent("Do the thing.", hooks={"after_step": [watch_progress]})
```

### `after_message_append`

Fires after each message is appended to the session context — LLM responses, user messages, and notifications.

```python
def on_message(runner, message):
    print(f"New message ({message.role}): {message.content}")

result = await agent.invoke_agent("Do the thing.", hooks={"after_message_append": [on_message]})
```

### `before_notification_publish`

Fires before an internal notification is published to subscribed channels.

```python
def on_notify(runner, message):
    print(f"Broadcasting: {message.content}")

result = await agent.invoke_agent("Do the thing.", hooks={"before_notification_publish": [on_notify]})
```

### `on_truncation_exhausted`

Fires when the LLM keeps getting cut off by the token limit after repeated retries. Receives `(counter, max_retries)`. Called instead of raising, giving you a chance to handle gracefully.

```python
def on_exhausted(counter, max_retries):
    print(f"Giving up after {counter}/{max_retries} attempts")
    raise RuntimeError("Truncation limit")

result = await agent.invoke_agent("Do the thing.", hooks={"on_truncation_exhausted": [on_exhausted]})
```

### Recursive Propagation

Hooks are automatically forwarded to every sub-agent invocation via `invoke()`. Hooks are **not** forwarded when an agent calls another agent's `invoke_agent()` from within a tool.

### Combining Hooks

Register multiple hooks per point — they execute in registration order:

```python
hooks = {
    "on_invoke": [log1, log2],  # log1 runs before log2
    "after_step": [watch, override],
}
```

## Media (Images, PDFs)

PeteOS supports three media flows: user-provided images, developer-pushed images, and agent self-initiated reading.

### User to Agent

Attach an image via `invoke_agent`:

```python
result = await agent.invoke_agent(
    "Describe this image.",
    image="/path/to/image.jpg",
)
```

The `image` parameter accepts a local file path or HTTP(S) URL. Requires `allow_media_access=True`.

### Developer to Agent

Push an image into the agent's context from within a `@tool`. The `runner` argument is auto-injected:

```python
@tool(description="Send the processed image to the agent for analysis.")
async def send_image_to_agent(self, runner) -> str:
    with open("/path/to/image.jpg", "rb") as f:
        image_bytes = f.read()

    await self._send_media(
        data=image_bytes,
        mime_type="image/jpeg",
        runner=runner,
    )
    return "OK: Image sent to agent."
```

The agent sees the image in its next reasoning iteration. Works with any MIME type supported by the backend (`image/png`, `image/jpeg`, `image/webp`).

### Agent Self-Initiated

When `allow_media_access=True` is set, the agent independently calls `read_media` during reasoning:

```python
@agentic_object(allow_media_access=True)
class ImageAnalyst(AgenticObject):
    """You can read and reason about images."""

result = await ImageAnalyst().invoke_agent("What's in flowers.png?")
```

### Media Flow Diagram

```
User provides image ──→ invoke_agent(image=...) ──→ Agent sees image
                                              │
                                      Agent calls read_media ──→ Image injected
                                              │
                                      Agent calls tool ──→ _send_media(data=...) ──→ Image injected
```

Each injected image becomes part of the conversation history.

## Adaptive Objects

Objects that modify their own capabilities at runtime:

```python
@agentic_object(define_functions=True)
class Adaptive(AgenticObject):
    ...

# The agent can call a hidden tool to register new @tool methods at runtime
```

## Key Principles

1. **Good tool descriptions** - The agent picks tools based on descriptions. Be specific about what each argument means.
2. **Return values guide the agent** - Return meaningful values so the agent can reason about results. Return `None` and the agent sees nothing.
3. **State lives on the object** - The agent manipulates state through tools. State persists within a persistent thread session.
4. **Enums constrain inputs** - Use `Enum` types to enumerate valid values. The agent cannot pass invalid ones.
5. **Multi-inheritance composes** - Combine atomic agents via inheritance. Tools are collected automatically.
6. **Sandboxing for complex logic** - Enable code execution for iterative/complex reasoning the agent handles via Python.
7. **No circular sub-agents** - Avoid A->B->A deadlock chains. Use timeouts.

## Quick Reference

```python
from peteos import AgenticObject, tool, agentic_object, sandbox

# Minimal agentic class
class MyAgent(AgenticObject):
    """System prompt."""
    @tool
    def my_tool(self, arg: str) -> str:
        """Tool description."""
        return "result"

# Invoke
result = await agent.invoke_agent("prompt")                           # None schema
result = await agent.invoke_agent("prompt", output_schema=Any)     # Any schema
result = await agent.invoke_agent("prompt", output_schema=list[str]) # specific schema
result = await agent.invoke_agent("prompt", persistent_thread_id="session-1")

# Configure class
@agentic_object(allow_code_execution=True, invoke_sub_agents=True)
class Configured(AgenticObject):
    """..."""
```

## Configuration

Peteos auto-loads `peteos.json` on `import peteos` - no manual setup required.

### Discovery Order

Peteos searches for `peteos.json` in this order (first match wins):

1. **`PETEOS_CONFIG`** environment variable - full path, always takes precedence
2. **`./peteos.json`** - current working directory (project-local)
3. **`$XDG_CONFIG_HOME/peteos/peteos.json`** - defaults to `~/.config/peteos/peteos.json` (user-specific)
4. **`/etc/peteos/peteos.json`** - system-wide (last resort)

### Config Directory Layout

When `peteos.json` is found, its parent directory becomes the **config root**. Two subdirectories are auto-discovered:

| Directory | Purpose |
|---|---|
| `roles/` | Persistent role overrides for agent personas |
| `agents/` | Agent runtime directory (sessions, contexts stored here) |

Both are optional - Peteos continues without them if absent.

### Example Layout

```
my-project/
|-- peteos.json
|-- agents/
|   |-- MyRole/
|       |-- <session-uuid>/
|           |-- session.json
|-- roles/
    |-- MyRole/
        |-- description.md     # Required
        |-- system_prompt.md   # Optional
        |-- config.json        # Optional
```

### Example `peteos.json`

```json
{
  "backends": [
    {
      "name": "ollama",
      "url": "http://localhost:11434",
      "model_priorities": {
        "glm-4.7-flash:latest": 100
      },
      "max_output": 16384
    }
  ]
}
```

The `backends` array configures LLM endpoints. If no `peteos.json` is found, Peteos fails on import.

## When to Use This Skill

Use this skill when:
- You need to create an agentic class that reasons and acts autonomously
- You want to add agentic capabilities to an existing class hierarchy
- You are building multi-agent systems with delegation
- You need structured output from LLM responses
- You want objects that maintain state across conversation sessions

Ask clarifying questions if the agent class purpose, available tools, or session lifecycle are unclear.
