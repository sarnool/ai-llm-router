from __future__ import annotations

import inspect
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = None
    sys.modules["requests"] = requests_stub

from ai_llm_router import llm_client as sdk


class _FakeResponse:
    def __init__(self, payload: dict, *, lines: list[bytes] | None = None):
        self.status_code = 200
        self.headers = {"x-request-id": "provider-request-123"}
        self._payload = payload
        self._lines = lines or []
        self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self) -> dict:
        return self._payload

    def iter_lines(self, decode_unicode: bool = True):
        return iter(self._lines)


class LLMClientTests(unittest.TestCase):
    def _custom_client(self) -> sdk.LLMClient:
        return sdk.LLMClient(
            sdk.LLMSettings(
                provider="custom_gateway",
                provider_type="custom",
                model="test-model",
                api_key="secret-key",
                gateway_url="https://example.invalid/chat/completions",
                streaming=False,
            )
        )

    def test_module_has_no_application_config_or_database_dependencies(self) -> None:
        self.assertNotIn("os", sdk.__dict__)
        self.assertNotIn("SqliteProjectRepository", sdk.__dict__)
        self.assertNotIn("get_user_settings_by_group", sdk.__dict__)
        self.assertNotIn("getenv", sdk.__dict__)
        source = inspect.getsource(sdk)
        self.assertNotIn("apps.backend.src.config", source)
        self.assertNotIn("SqliteProjectRepository", source)
        self.assertNotIn("os.getenv", source)

    def test_truncated_json_is_recovered(self) -> None:
        self.assertEqual(
            sdk.LLMClient._parse_json('{"content":"draft text"'),
            {"content": "draft text"},
        )

    def test_custom_transport_requires_explicit_gateway_url(self) -> None:
        client = sdk.LLMClient(
            sdk.LLMSettings(
                provider="custom_gateway",
                provider_type="custom",
                model="test-model",
                api_key="test-key",
            )
        )
        enabled, reason = client.enabled_with_reason()

        self.assertFalse(enabled)
        self.assertIn("gateway_url", reason)

    def test_custom_non_streaming_returns_parsed_output_and_metadata(self) -> None:
        payload = {
            "id": "response-123",
            "model": "test-model",
            "choices": [{"message": {"content": '{"answer":"hello"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        response = _FakeResponse(payload)
        client = self._custom_client()
        metadata_events: list[dict] = []
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        with patch.object(sdk.requests, "post", return_value=response) as post:
            result = client.complete(
                "answer",
                response_schema=schema,
                stream=False,
                on_metadata=metadata_events.append,
            )

        self.assertEqual(result.parsed, {"answer": "hello"})
        self.assertEqual(result.metadata["response_id"], "response-123")
        self.assertEqual(result.metadata["request_id"], "provider-request-123")
        self.assertEqual(result.metadata["total_tokens"], 7)
        self.assertEqual(result.metadata["status"], "ok")
        self.assertEqual(result.metadata["request_headers"]["Authorization"], "[REDACTED]")
        self.assertEqual(metadata_events, [result.metadata])
        self.assertEqual(post.call_args.kwargs["json"]["model"], "test-model")
        self.assertFalse(post.call_args.kwargs["stream"])

    def test_custom_streaming_emits_deltas_and_stream_metadata(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"hello "}}]}',
            b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ]
        response = _FakeResponse({}, lines=lines)
        client = self._custom_client()
        received: list[str] = []
        client.settings.streaming = True

        with patch.object(sdk.requests, "post", return_value=response) as post:
            result = client.complete("say hello", json_output=False, on_delta=received.append)

        self.assertEqual(result.text, "hello world")
        self.assertEqual(received, ["hello ", "world"])
        self.assertEqual(result.metadata["chunk_count"], 2)
        self.assertEqual(result.metadata["finish_reason"], "stop")
        self.assertTrue(post.call_args.kwargs["stream"])

    def test_litellm_receives_explicit_settings_and_returns_metadata(self) -> None:
        fake_module = types.ModuleType("litellm")
        calls: list[dict] = []

        def completion(**kwargs):
            calls.append(kwargs)
            return {
                "id": "lite-response",
                "model": "openai/test-model",
                "choices": [{"message": {"content": "plain text"}, "finish_reason": "stop"}],
            }

        fake_module.completion = completion
        client = sdk.LLMClient(sdk.LLMSettings(provider="openai", model="test-model", api_key="explicit-key"))

        with patch.dict(sys.modules, {"litellm": fake_module}):
            result = client.complete("say hello", json_output=False, stream=False)

        self.assertEqual(result.text, "plain text")
        self.assertEqual(result.metadata["response_id"], "lite-response")
        self.assertEqual(calls[0]["model"], "openai/test-model")
        self.assertEqual(calls[0]["api_key"], "explicit-key")


if __name__ == "__main__":
    unittest.main(verbosity=2)
