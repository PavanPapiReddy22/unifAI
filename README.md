# modelgate

A lightweight, type-safe adapter layer that gives you one consistent interface across every major LLM provider. No bloated SDKs — just `httpx` and `pydantic` under the hood.

## Install

```bash
pip install modelgate

# AWS Bedrock support
pip install "modelgate[aws]"

# Vertex AI support
pip install "modelgate[vertex]"
```

## Quick start

```python
import asyncio
from modelgate import ModelGate, ModelGateConfig

async def main():
    client = ModelGate(ModelGateConfig(
        openai_api_key="sk-...",
        anthropic_api_key="sk-ant-...",
    ))

    response = await client.chat(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(response.text)  # "4"

asyncio.run(main())
```

## Supported providers

| Provider   | Model string prefix | Config key            |
|------------|---------------------|-----------------------|
| OpenAI     | `openai/`           | `openai_api_key`      |
| Anthropic  | `anthropic/`        | `anthropic_api_key`   |
| AWS Bedrock| `bedrock/`          | `aws_region`, `boto3_session` |
| Gemini     | `gemini/`           | `gemini_api_key`      |
| Vertex AI  | `vertex/`           | `vertex_credentials`  |
| Groq       | `groq/`             | `groq_api_key`        |
| Ollama     | `ollama/`           | `ollama_base_url`     |

Any OpenAI-compatible API works via `GenericOpenAIAdapter` — no new adapter code needed.

## Streaming

```python
async for chunk in client.stream(
    model="anthropic/claude-opus-4-5",
    messages=[{"role": "user", "content": "Tell me a story"}],
):
    if chunk.type == "text":
        print(chunk.text, end="", flush=True)
```

## Tool use

```python
from modelgate import Tool, ToolParameter

weather = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={"location": ToolParameter(type="string", description="City name")},
    required=["location"],
)

response = await client.chat(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=[weather],
)

for tc in response.tool_calls:
    print(tc.tool_name)   # "get_weather"
    print(tc.tool_input)  # {"location": "NYC"}
```

Tool calls return the same `ContentBlock` shape regardless of provider.

## Error handling

```python
from modelgate import AuthenticationError, RateLimitError, InvalidRequestError

try:
    response = await client.chat(...)
except AuthenticationError:
    pass  # bad or missing API key
except RateLimitError:
    pass  # hit provider rate limit — retry with backoff
except InvalidRequestError:
    pass  # malformed request
```

Full error hierarchy:

```
ModelGateError
├── AuthenticationError   # 401
├── RateLimitError        # 429
├── InvalidRequestError   # 400
├── ProviderError         # 5xx
│   ├── BedrockError
│   └── VertexError
└── StreamingError        # error mid-stream
```

## `ModelGateConfig` reference

| Field               | Type        | Default                          | Description                     |
|---------------------|-------------|----------------------------------|---------------------------------|
| `openai_api_key`    | `str\|None` | `None`                           | OpenAI API key                  |
| `anthropic_api_key` | `str\|None` | `None`                           | Anthropic API key               |
| `gemini_api_key`    | `str\|None` | `None`                           | Gemini API key                  |
| `groq_api_key`      | `str\|None` | `None`                           | Groq API key                    |
| `aws_region`        | `str`       | `"us-east-1"`                    | AWS region for Bedrock          |
| `boto3_session`     | `Any\|None` | `None`                           | Custom boto3 session            |
| `vertex_credentials`| `Any\|None` | `None`                           | Google auth credentials         |
| `ollama_base_url`   | `str`       | `"http://localhost:11434/v1"`    | Ollama server URL               |
| `groq_base_url`     | `str`       | `"https://api.groq.com/openai/v1"` | Groq API base URL             |

API keys can also be set via environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`.

## `chat()` / `stream()` parameters

| Parameter     | Type                  | Default  | Description                            |
|---------------|-----------------------|----------|----------------------------------------|
| `model`       | `str`                 | required | `"provider/model-id"` format           |
| `messages`    | `list[dict\|Message]` | required | Conversation history                   |
| `tools`       | `list[Tool]\|None`   | `None`   | Tools available to the model           |
| `system`      | `str\|None`           | `None`   | System prompt                          |
| `max_tokens`  | `int`                 | `4096`   | Maximum tokens to generate             |
| `temperature` | `float`               | `1.0`    | Sampling temperature                   |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
