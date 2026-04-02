"""Tests for the Gemini adapter."""

import json

import httpx
import pytest
import respx

from modelgate.providers.gemini import GeminiAdapter
from modelgate.types import (
    ContentBlock,
    ContentType,
    FinishReason,
    Message,
    Role,
    Tool,
    ToolParameter,
)
from modelgate.errors import AuthenticationError, RateLimitError


# ── Mock Responses ───────────────────────────────────────────────────────────

MOCK_TEXT_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello! How can I help you today?"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 8,
        "candidatesTokenCount": 10,
        "totalTokenCount": 18,
    },
    "responseId": "resp-gemini-123",
}

MOCK_TOOL_CALL_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "Let me check the weather for you."},
                    {
                        "functionCall": {
                            "id": "fc_001",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                        }
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 15,
        "candidatesTokenCount": 12,
        "totalTokenCount": 27,
    },
    "responseId": "resp-gemini-tool456",
}

WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get current weather",
    parameters={
        "location": ToolParameter(type="string", description="City name"),
    },
    required=["location"],
)


# ── Non-Streaming Tests ─────────────────────────────────────────────────────


class TestGeminiChat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_text(self) -> None:
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        resp = await adapter.chat(messages=messages, model="gemini-2.0-flash")

        assert resp.id == "resp-gemini-123"
        assert resp.text == "Hello! How can I help you today?"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.input_tokens == 8
        assert resp.usage.output_tokens == 10

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_call_response(self) -> None:
        """Gemini can return text AND functionCall parts in a single response."""
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TOOL_CALL_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What's the weather in NYC?")
        ]
        resp = await adapter.chat(
            messages=messages,
            model="gemini-2.0-flash",
            tools=[WEATHER_TOOL],
        )

        # Verify mixed content preserved in order
        assert len(resp.content) == 2
        assert resp.content[0].type == ContentType.TEXT
        assert resp.content[0].text == "Let me check the weather for you."
        assert resp.content[1].type == ContentType.TOOL_USE
        assert resp.content[1].tool_call_id == "fc_001"
        assert resp.content[1].tool_name == "get_weather"
        assert resp.content[1].tool_input == {"location": "NYC"}
        assert isinstance(resp.content[1].tool_input, dict)

        # Finish reason should be TOOL_USE when tool calls present
        assert resp.finish_reason == FinishReason.TOOL_USE
        assert len(resp.tool_calls) == 1


class TestGeminiMessageFormat:
    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_result_sent_as_function_response(self) -> None:
        """Tool results must be sent as user messages with functionResponse parts."""
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="What's the weather?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.TEXT,
                        text="Let me check.",
                    ),
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id="fc_001",
                        tool_name="get_weather",
                        tool_input={"location": "NYC"},
                    ),
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id="fc_001",
                        tool_name="get_weather",
                        tool_result_content="72°F and sunny",
                    )
                ],
            ),
        ]
        await adapter.chat(messages=messages, model="gemini-2.0-flash")

        sent_body = json.loads(route.calls[0].request.content)

        # Assistant message → role=model with functionCall part
        assistant_msg = sent_body["contents"][1]
        assert assistant_msg["role"] == "model"
        assert "functionCall" in assistant_msg["parts"][1]

        # Tool result → role=user with functionResponse part including id
        tool_msg = sent_body["contents"][2]
        assert tool_msg["role"] == "user"
        assert "functionResponse" in tool_msg["parts"][0]
        fr = tool_msg["parts"][0]["functionResponse"]
        assert fr["name"] == "get_weather"
        assert fr["id"] == "fc_001"
        assert fr["response"]["result"] == "72°F and sunny"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_sent_as_system_instruction(self) -> None:
        """System prompt must be sent as top-level system_instruction."""
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]
        await adapter.chat(
            messages=messages,
            model="gemini-2.0-flash",
            system="You are a helpful assistant",
        )

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["systemInstruction"]["parts"][0]["text"] == "You are a helpful assistant"
        # System should NOT appear in contents
        for content in sent_body["contents"]:
            for part in content.get("parts", []):
                if "text" in part:
                    assert part["text"] != "You are a helpful assistant"

    @respx.mock
    @pytest.mark.asyncio
    async def test_tools_sent_as_function_declarations(self) -> None:
        """Tools should be sent as functionDeclarations."""
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Check weather")]
        await adapter.chat(
            messages=messages, model="gemini-2.0-flash", tools=[WEATHER_TOOL]
        )

        sent_body = json.loads(route.calls[0].request.content)
        tool_decl = sent_body["tools"][0]["functionDeclarations"][0]
        assert tool_decl["name"] == "get_weather"
        assert tool_decl["description"] == "Get current weather"
        assert "location" in tool_decl["parameters"]["properties"]


class TestGeminiErrors:
    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error(self) -> None:
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "API key not valid"}}
            )
        )

        adapter = GeminiAdapter(api_key="bad-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(AuthenticationError):
            await adapter.chat(messages=messages, model="gemini-2.0-flash")

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(
            return_value=httpx.Response(
                429, json={"error": {"message": "Rate limit exceeded"}}
            )
        )

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]

        with pytest.raises(RateLimitError):
            await adapter.chat(messages=messages, model="gemini-2.0-flash")


class TestGeminiThoughtSignature:
    """Gemini 3 returns thoughtSignature on functionCall parts — must round-trip."""

    MOCK_RESPONSE_WITH_THOUGHT_SIG = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "id": "fc_002",
                                "name": "get_weather",
                                "args": {"location": "SF"},
                            },
                            "thoughtSignature": "abc123sig==",
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
        "responseId": "resp-thought-sig",
    }

    @respx.mock
    @pytest.mark.asyncio
    async def test_thought_signature_captured_from_response(self) -> None:
        """thoughtSignature must be captured into ContentBlock.thought_signature."""
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(
            return_value=httpx.Response(200, json=self.MOCK_RESPONSE_WITH_THOUGHT_SIG)
        )

        adapter = GeminiAdapter(api_key="test-key")
        messages = [Message(role=Role.USER, content="Weather in SF?")]
        resp = await adapter.chat(
            messages=messages, model="gemini-2.0-flash", tools=[WEATHER_TOOL]
        )

        assert resp.content[0].type == ContentType.TOOL_USE
        assert resp.content[0].thought_signature == "abc123sig=="

    @respx.mock
    @pytest.mark.asyncio
    async def test_thought_signature_passed_back_in_history(self) -> None:
        """When serializing assistant messages, thoughtSignature must be included."""
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="Weather?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_USE,
                        tool_call_id="fc_002",
                        tool_name="get_weather",
                        tool_input={"location": "SF"},
                        thought_signature="abc123sig==",
                    ),
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        tool_call_id="fc_002",
                        tool_name="get_weather",
                        tool_result_content="65°F foggy",
                    )
                ],
            ),
        ]
        await adapter.chat(messages=messages, model="gemini-2.0-flash")

        sent_body = json.loads(route.calls[0].request.content)
        assistant_part = sent_body["contents"][1]["parts"][0]

        # The functionCall part must carry thoughtSignature back
        assert "functionCall" in assistant_part
        assert assistant_part["thoughtSignature"] == "abc123sig=="


# ── Image Tests ──────────────────────────────────────────────────────────────


class TestGeminiImages:

    @respx.mock
    @pytest.mark.asyncio
    async def test_image_base64_sent_as_inline_data(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content=[
                ContentBlock(type=ContentType.IMAGE, image_source_type="base64",
                             image_media_type="image/png", image_data="iVBOR"),
                ContentBlock(type=ContentType.TEXT, text="What's this?"),
            ])],
            model="gemini-2.0-flash",
        )

        body = json.loads(route.calls[0].request.content)
        parts = body["contents"][0]["parts"]
        assert parts[0]["inlineData"]["mimeType"] == "image/png"
        assert parts[0]["inlineData"]["data"] == "iVBOR"
        assert parts[1]["text"] == "What's this?"

    @respx.mock
    @pytest.mark.asyncio
    async def test_image_url_sent_as_file_data(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content=[
                ContentBlock(type=ContentType.IMAGE, image_source_type="url",
                             image_media_type="image/jpeg",
                             image_data="https://example.com/cat.jpg"),
            ])],
            model="gemini-2.0-flash",
        )

        body = json.loads(route.calls[0].request.content)
        parts = body["contents"][0]["parts"]
        assert parts[0]["fileData"]["mimeType"] == "image/jpeg"
        assert parts[0]["fileData"]["fileUri"] == "https://example.com/cat.jpg"


# ── Tool Config Tests ────────────────────────────────────────────────────────


class TestGeminiToolConfig:

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_auto(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["toolConfig"]["function_calling_config"]["mode"] == "AUTO"

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_any(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            tools=[WEATHER_TOOL],
            tool_choice="any",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["toolConfig"]["function_calling_config"]["mode"] == "ANY"

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_choice_none_omitted(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
        )
        body = json.loads(route.calls[0].request.content)
        assert "toolConfig" not in body


# ── Kwargs Tests ─────────────────────────────────────────────────────────────


class TestGeminiKwargs:

    @respx.mock
    @pytest.mark.asyncio
    async def test_passthrough_params_forwarded(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            top_p=0.9,
            top_k=40,
            seed=42,
        )
        body = json.loads(route.calls[0].request.content)
        gen = body["generationConfig"]
        assert gen["topP"] == 0.9
        assert gen["topK"] == 40
        assert gen["seed"] == 42

    @respx.mock
    @pytest.mark.asyncio
    async def test_response_mime_type_forwarded(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            response_mime_type="application/json",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"

    @respx.mock
    @pytest.mark.asyncio
    async def test_response_schema_forwarded(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            response_mime_type="application/json",
            response_schema=schema,
        )
        body = json.loads(route.calls[0].request.content)
        assert body["generationConfig"]["responseSchema"] == schema

    @respx.mock
    @pytest.mark.asyncio
    async def test_safety_settings_forwarded(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            safety_settings=safety,
        )
        body = json.loads(route.calls[0].request.content)
        assert body["safetySettings"] == safety

    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_level_forwarded(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            thinking_level="high",
        )
        body = json.loads(route.calls[0].request.content)
        assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"

    @respx.mock
    @pytest.mark.asyncio
    async def test_thinking_budget_forwarded(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            thinking_budget=8192,
        )
        body = json.loads(route.calls[0].request.content)
        assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 8192


# ── Raw Schema Tests ─────────────────────────────────────────────────────────


class TestGeminiRawSchema:

    @respx.mock
    @pytest.mark.asyncio
    async def test_raw_schema_used(self) -> None:
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        ).mock(return_value=httpx.Response(200, json=MOCK_TEXT_RESPONSE))

        raw = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        adapter = GeminiAdapter(api_key="test-key")
        await adapter.chat(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            tools=[Tool(name="search", description="Search", raw_schema=raw)],
        )
        body = json.loads(route.calls[0].request.content)
        decl = body["tools"][0]["functionDeclarations"][0]
        assert decl["parameters"] == raw


# ── Streaming Tests ──────────────────────────────────────────────────────────


class TestGeminiStream:

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_text(self) -> None:
        sse = "\n".join([
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
            }),
            '',
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": [{"text": " world"}], "role": "model"}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
            }),
            '',
        ])
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ).mock(return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))

        adapter = GeminiAdapter(api_key="test-key")
        chunks = []
        async for chunk in adapter.stream(
            messages=[Message(role=Role.USER, content="Hi")], model="gemini-2.0-flash"
        ):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if isinstance(c, ContentBlock) and c.type == ContentType.TEXT]
        assert len(text_chunks) == 2
        assert text_chunks[0].text == "Hello"
        assert text_chunks[1].text == " world"

        from modelgate.types import Usage
        usage_chunks = [c for c in chunks if isinstance(c, Usage)]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].input_tokens == 5
        assert usage_chunks[0].output_tokens == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_tool_call(self) -> None:
        sse = "\n".join([
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": [
                    {"functionCall": {"id": "fc_s1", "name": "get_weather", "args": {"location": "NYC"}}}
                ], "role": "model"}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }),
            '',
        ])
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ).mock(return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))

        adapter = GeminiAdapter(api_key="test-key")
        tool_chunks = []
        async for chunk in adapter.stream(
            messages=[Message(role=Role.USER, content="Weather?")], model="gemini-2.0-flash"
        ):
            if isinstance(chunk, ContentBlock) and chunk.type == ContentType.TOOL_USE:
                tool_chunks.append(chunk)

        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_name == "get_weather"
        assert tool_chunks[0].tool_input == {"location": "NYC"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_error_raises(self) -> None:
        from modelgate.errors import StreamingError
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ).mock(return_value=httpx.Response(401, json={"error": {"message": "Bad key"}}))

        adapter = GeminiAdapter(api_key="bad-key")
        with pytest.raises((AuthenticationError, StreamingError)):
            async for _ in adapter.stream(
                messages=[Message(role=Role.USER, content="Hi")], model="gemini-2.0-flash"
            ):
                pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_kwargs_forwarded(self) -> None:
        sse = "\n".join([
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            }),
            '',
        ])
        route = respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ).mock(return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))

        adapter = GeminiAdapter(api_key="test-key")
        async for _ in adapter.stream(
            messages=[Message(role=Role.USER, content="Hi")],
            model="gemini-2.0-flash",
            tool_choice="auto",
            top_p=0.8,
            thinking_level="balanced",
        ):
            pass
        body = json.loads(route.calls[0].request.content)
        assert body["toolConfig"]["function_calling_config"]["mode"] == "AUTO"
        assert body["generationConfig"]["topP"] == 0.8
        assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "BALANCED"
