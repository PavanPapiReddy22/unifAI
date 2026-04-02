# Gemini Adapter

Complete reference for the ModelGate Gemini adapter. Covers every supported feature, how inputs are translated to the Gemini `generateContent` API, and how responses are normalized back.

**API Endpoint:** `https://generativelanguage.googleapis.com/v1beta`

---

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Supported Models](#supported-models)
- [Chat (Non-Streaming)](#chat-non-streaming)
- [Streaming](#streaming)
- [Content Types](#content-types)
  - [Text](#text)
  - [Tool Use](#tool-use)
  - [Tool Results](#tool-results)
  - [Images (Vision)](#images-vision)
- [System Prompts](#system-prompts)
- [Tool Definitions](#tool-definitions)
- [Tool Choice](#tool-choice)
- [Structured Output](#structured-output)
- [Sampling Parameters](#sampling-parameters)
- [Thinking Config](#thinking-config)
- [Safety Settings](#safety-settings)
- [Stop Reasons](#stop-reasons)
- [Error Handling](#error-handling)
- [Translation Reference](#translation-reference)

---

## Quick Start

```python
from modelgate import ModelGate, ModelGateConfig, Message, Role

client = ModelGate(ModelGateConfig(gemini_api_key="AIza..."))

# Non-streaming
response = await client.chat(
    model="gemini/gemini-2.0-flash",
    messages=[Message(role=Role.USER, content="Hello!")],
)
print(response.text)

# Streaming
async for chunk in client.stream(
    model="gemini/gemini-2.0-flash",
    messages=[Message(role=Role.USER, content="Tell me a story")],
):
    if chunk.type == "text":
        print(chunk.text, end="", flush=True)
```

---

## Authentication

The adapter resolves the API key in this order:

1. `api_key` passed to `GeminiAdapter(api_key="...")` directly
2. `gemini_api_key` on `ModelGateConfig`
3. `GEMINI_API_KEY` environment variable

The API key is appended to all request URLs via the `?key=` query parameter.

---

## Supported Models

Any model ID works. Pass via the `gemini/` prefix:

| ModelGate String | Model ID Sent to API |
|---|---|
| `gemini/gemini-2.0-flash` | `gemini-2.0-flash` |
| `gemini/gemini-2.0-flash-thinking-exp` | `gemini-2.0-flash-thinking-exp` |
| `gemini/gemini-2.5-pro` | `gemini-2.5-pro` |

---

## Chat (Non-Streaming)

```python
response = await client.chat(
    model="gemini/gemini-2.0-flash",
    messages=[...],          # required
    tools=[...],             # optional
    system="...",            # optional
    max_tokens=4096,         # optional, default 4096
    temperature=1.0,         # optional, default 1.0
    **kwargs,                # optional, see kwargs section below
)
```

### Response Object

| Field | Type | Description |
|---|---|---|
| `response.id` | `str` | API Response ID |
| `response.model` | `str` | Model ID used |
| `response.content` | `list[ContentBlock]` | All content blocks |
| `response.usage` | `Usage` | Token counts |
| `response.finish_reason` | `FinishReason` | Why the model stopped |
| `response.text` | `str \| None` | Concatenated text |
| `response.tool_calls` | `list[ContentBlock]` | TOOL_USE blocks only |

---

## Streaming

```python
async for chunk in client.stream(
    model="gemini/gemini-2.0-flash",
    messages=[...],
):
    # chunk is ContentBlock or Usage
```

### Stream Protocol

```
┌─────────────────────────────────────────────────────┐
│ Gemini SSE Events             ModelGate Yields       │
├─────────────────────────────────────────────────────┤
│ {"candidates":... "text"}  → ContentBlock(TEXT)  ──► │
│   (streamed immediately)                            │
│ {"functionCall": ...}      → (buffers)              │
│ {"finishReason": "STOP"}   → ContentBlock(TOOL_USE)──► │
│                              Usage               ──► │
└─────────────────────────────────────────────────────┘
```

**Key behaviors:**
- **Tool use:** Buffer until the end of the stream, emits as complete `ContentBlock(TOOL_USE)` right before Usage.
- **Usage:** Arrives on the final payload alongside the finish reason.

---

## Content Types

### Text

**ModelGate:**
```python
Message(role=Role.USER, content="Hello")
```

**Gemini API:**
```json
{"role": "user", "parts": [{"text": "Hello"}]}
```

---

### Tool Use

**Receiving:**
```python
for tc in response.tool_calls:
    tc.tool_call_id        # "fc_001" (or part name if older model)
    tc.tool_name           # "get_weather"
    tc.tool_input          # {"city": "NYC"} — always a parsed dict
    tc.thought_signature   # Optional Gemini 3 signature
```

**Gemini API → ModelGate:**
```json
// Gemini Output
{"functionCall": {
  "id": "fc_001",
  "name": "get_weather",
  "args": {"city": "NYC"}
}, "thoughtSignature": "abc=="}

// Becomes
ContentBlock(TOOL_USE, tool_call_id="fc_001", tool_name="get_weather", tool_input={"city": "NYC"}, thought_signature="abc==")
```

---

### Tool Results

**Sending tool results back:**
```python
Message(role=Role.TOOL, content=[
    ContentBlock(type=ContentType.TOOL_RESULT,
                 tool_call_id="fc_001",
                 tool_name="get_weather",
                 tool_result_content="72°F and sunny"),
])
```

**ModelGate → Gemini API:**
```json
{"role": "user", "parts": [
  {"functionResponse": {
    "name": "get_weather",
    "id": "fc_001",
    "response": {"result": "72°F and sunny"}
  }}
]}
```
> **Note:** Gemini wants tool results grouped within a `"role": "user"` message using `functionResponse` parts. The adapter handles this remapping automatically.

---

### Images (Vision)

**Sending:**
```python
# Base64
Message(role=Role.USER, content=[
    ContentBlock(type=ContentType.IMAGE, image_source_type="base64",
                 image_media_type="image/jpeg", image_data="iVBOR..."),
])

# Media URL
Message(role=Role.USER, content=[
    ContentBlock(type=ContentType.IMAGE, image_source_type="url",
                 image_media_type="image/jpeg", image_data="https://..."),
])
```

**API translation:**
```json
// Base64 becomes inlineData
{"inlineData": {"mimeType": "image/jpeg", "data": "iVBOR..."}}

// URL becomes fileData
{"fileData": {"mimeType": "image/jpeg", "fileUri": "https://..."}}
```

---

## System Prompts

Sent inside the `systemInstruction` top-level payload structure.

```python
await client.chat(..., system="You are helpful.")
```
```json
{
  "systemInstruction": {"parts": [{"text": "You are helpful."}]},
  "contents": [...]
}
```

---

## Tool Definitions

```python
from modelgate import Tool, ToolParameter
tool = Tool(
    name="get_weather",
    description="Get weather",
    parameters={"city": ToolParameter(type="string", description="City")},
    required=["city"]
)
```

**API translation:**
```json
{"tools": [{"functionDeclarations": [
  {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string", "description": "City"}},
      "required": ["city"]
    }
  }
]}]}
```
`raw_schema` is supported and passed verbatim into the `parameters` block.

---

## Tool Choice

```python
tool_choice=None          # omitted
tool_choice="auto"        # mode: "AUTO"
tool_choice="any"         # mode: "ANY"
tool_choice="none"        # mode: "NONE"
tool_choice={"function_calling_config": {...}} # raw dict override
```
Forwarded into `toolConfig` in the payload. Note the string values differ slightly from OpenAI (`any` vs `required`).

---

## Structured Output

```python
await client.chat(...,
    response_mime_type="application/json",
    response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
)
```
Forwarded into `generationConfig`.

---

## Sampling Parameters

```python
await client.chat(...,
    temperature=0.7,
    max_tokens=4096,
    top_p=0.9,
    top_k=40,
    seed=42,
)
```
Forwarded into `generationConfig`.

---

## Thinking Config

Supported for Gemini 2.5/3 model lines:

```python
await client.chat(...,
    thinking_level="high",  # "HIGH", "BALANCED", "LOW", "MINIMAL"
    thinking_budget=1024,   # Legacy 2.0 experimental feature
)
```
Forwarded into `generationConfig.thinkingConfig`.

---

## Safety Settings

```python
await client.chat(...,
    safety_settings=[{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    }]
)
```

---

## Stop Reasons

| Gemini `finishReason` | ModelGate `FinishReason` | Meaning |
|---|---|---|
| `STOP` | `STOP` | Normal completion |
| *derived* | `TOOL_USE` | Model invoked tool. We override this if functionCall parts present. |
| `MAX_TOKENS` | `LENGTH` | Hit max_tokens limit |
| `SAFETY` | `STOP` | Content blocked |
| `OTHER` | `ERROR` | System error |
| `MALFORMED_FUNCTION_CALL` | `ERROR` | API parsing layer failed |

---

## Error Handling

Standard ModelGate typed exception hierarchy:
| HTTP/Event | Exception |
|---|---|
| 401 | `AuthenticationError` |
| 429 | `RateLimitError` |
| 400 | `InvalidRequestError` |
| 5xx | `ProviderError` |

---

## Translation Reference

| Kwarg | Gemini API Target Field |
|---|---|
| `tool_choice="auto"` | `toolConfig.function_calling_config.mode = "AUTO"` |
| `response_mime_type` | `generationConfig.responseMimeType` |
| `response_schema` | `generationConfig.responseSchema` |
| `top_p` | `generationConfig.topP` |
| `top_k` | `generationConfig.topK` |
| `seed` | `generationConfig.seed` |
| `thinking_level` | `generationConfig.thinkingConfig.thinkingLevel` |
| `thinking_budget`| `generationConfig.thinkingConfig.thinkingBudget` |
| `safety_settings`| `safetySettings` |
