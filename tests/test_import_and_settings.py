import os
from types import SimpleNamespace

from ai_llm_router import LLMClient, LLMSettings


def test_imports_and_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_API_KEY", "dummy")

    settings = LLMSettings.from_env()

    assert settings.provider == "openai"
    assert settings.model == "gpt-4o-mini"
    assert settings.api_key == "dummy"

    client = LLMClient(settings)
    assert client.supports_streaming("openai") is True


def test_collect_response_metadata_includes_envelope_token_and_oaif_fields():
    client = LLMClient(LLMSettings(provider="google", model="gemini-2.5-flash", api_key="dummy"))

    metadata = client._collect_response_metadata(
        {
            "id": "G1hxapO3I6-f47sPl_XbwQw",
            "created": 1785813019,
            "model": "gemini-2.5-flash",
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "ok"}}],
            "usage": {
                "cache_read_input_tokens": None,
                "completion_tokens": 122,
                "completion_tokens_details": {
                    "reasoning_tokens": 82,
                    "text_tokens": 40,
                },
                "prompt_tokens": 28,
                "prompt_tokens_details": {
                    "audio_tokens": None,
                    "cache_write_tokens": None,
                    "cached_tokens": None,
                    "image_tokens": None,
                    "text_tokens": 28,
                    "video_tokens": None,
                },
                "total_tokens": 150,
            },
            "oaif": {
                "apigee_message_id": "0c6bf1ac-6dd5-41e4-80c7-54de6fe18bcf1972",
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "correlation_id": "8cb8b3a2-4255-4816-b058-e53a7d851268",
                "cost_modifier_pct": 0,
                "dt_trace_id": "8cb8b3a2-4255-4816-b058-e53a7d851268",
                "duration_s": None,
                "input_tokens": 28,
                "model": "gemini-2.5-flash",
                "oaif_cost_usd": 0.0003134,
                "output_tokens": 122,
                "request_id": "0c6bf1ac-6dd5-41e4-80c7-54de6fe18bcf1972",
            },
        }
    )

    assert metadata == {
        "response_id": "G1hxapO3I6-f47sPl_XbwQw",
        "created": 1785813019,
        "response_model": "gemini-2.5-flash",
        "response_object": "chat.completion",
        "finish_reason": "stop",
        "prompt_tokens": 28,
        "completion_tokens": 122,
        "total_tokens": 150,
        "input_tokens": 28,
        "output_tokens": 122,
        "reasoning_tokens": 82,
        "text_tokens": 40,
        "prompt_text_tokens": 28,
        "request_id": "0c6bf1ac-6dd5-41e4-80c7-54de6fe18bcf1972",
        "apigee_message_id": "0c6bf1ac-6dd5-41e4-80c7-54de6fe18bcf1972",
        "correlation_id": "8cb8b3a2-4255-4816-b058-e53a7d851268",
        "dt_trace_id": "8cb8b3a2-4255-4816-b058-e53a7d851268",
        "oaif_cost_usd": 0.0003134,
        "cost_modifier_pct": 0,
        "oaif_model": "gemini-2.5-flash",
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }


def test_collect_response_metadata_includes_stream_context_fields():
    client = LLMClient(LLMSettings(provider="google", model="gemini-2.5-flash", api_key="dummy"))

    metadata = client._collect_response_metadata(
        {
            "stream_context": {
                "response_headers": {"x-request-id": "req-123", "content-type": "text/event-stream"},
                "response_status_code": 200,
                "response_http_version": "HTTP/1.1",
                "request_headers": {"authorization": "***REDACTED***", "x-correlation-id": "corr-123"},
                "request_url": "https://example.test/v1/chat/completions",
            }
        }
    )

    assert metadata == {
        "response_headers": {"x-request-id": "req-123", "content-type": "text/event-stream"},
        "response_status_code": 200,
        "response_http_version": "HTTP/1.1",
        "request_headers": {"authorization": "***REDACTED***", "x-correlation-id": "corr-123"},
        "request_url": "https://example.test/v1/chat/completions",
    }


def test_run_stream_collects_text_and_chunk_payload_in_single_pass():
    client = LLMClient(LLMSettings(provider="google", model="gemini-2.5-flash", api_key="dummy"))
    deltas: list[str] = []

    chunks = [
        {
            "id": "chunk-1",
            "created": 1785813117,
            "model": "gemini-2.5-flash",
            "object": "chat.completion.chunk",
            "choices": [{"finish_reason": None, "index": 0, "delta": {"content": "hello "}}],
        },
        {
            "id": "chunk-2",
            "created": 1785813118,
            "model": "gemini-2.5-flash",
            "object": "chat.completion.chunk",
            "choices": [{"finish_reason": "stop", "index": 0, "delta": {"content": "world"}}],
            "oaif": {
                "request_id": "req-1",
                "correlation_id": "corr-1",
                "input_tokens": 28,
                "output_tokens": 122,
                "oaif_cost_usd": 0.0003134,
            },
        },
    ]

    text, stream_payload, raw_response = client._run_stream(chunks, on_delta=deltas.append)

    assert text == "hello world"
    assert deltas == ["hello ", "world"]
    assert raw_response == chunks[-1]
    assert stream_payload == {"chunks": chunks}


def test_execute_oaif_request_passes_header_as_named_parameter():
    client = LLMClient(LLMSettings(provider="optusaifoundation", model="model"))
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, *, header=None, **kwargs):
            captured["header"] = header
            captured["kwargs"] = kwargs
            return {"choices": [{"message": {"content": "ok"}}]}

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    text, metadata, raw_response = client._execute_oaif_request(
        client=fake_client,
        kwargs={
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
        header={"x-correlation-id": "abc-123"},
        stream=False,
        on_delta=None,
    )

    assert text == "ok"
    assert metadata == {}
    assert raw_response == {"choices": [{"message": {"content": "ok"}}]}
    assert captured["header"] == {"x-correlation-id": "abc-123"}
    assert "header" not in captured["kwargs"]
