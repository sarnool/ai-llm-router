from __future__ import annotations

import importlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SUPPORTED_PROVIDERS = {"openai", "anthropic", "google", "optusaifoundation"}
SUPPORTED_STREAMING_PROVIDERS = {"openai", "anthropic", "google", "optusaifoundation"}
SUPPORTED_NATIVE_RESPONSE_SCHEMA_PROVIDERS = {"openai", "google", "optusaifoundation"}


def _load_env_file() -> None:
    # Prefer CWD .env, then walk upward from this module location.
    candidates: list[Path] = [Path.cwd() / ".env"]
    candidates.extend(parent / ".env" for parent in Path(__file__).resolve().parents)

    env_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# _load_env_file()


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(
        token in normalized
        for token in ("secret", "token", "password", "api_keyxx", "apikeyxx", "authorization", "auth")
    )


def _redact_sensitive_data(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if (
                isinstance(key, str)
                and _is_sensitive_key(key)
                and not (("token" in key.lower()) and isinstance(item, (int, float)))
            ):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_sensitive_data(item)
        return redacted

    if isinstance(value, list):
        return [_redact_sensitive_data(item) for item in value]

    return value


def _to_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                pass
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return value


def _extract_text_fragment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_extract_text_fragment(item) for item in value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        if value.get("type") in {"text", "output_text"}:
            return _extract_text_fragment(value.get("text"))
        if "content" in value:
            return _extract_text_fragment(value.get("content"))
        if "parts" in value:
            return _extract_text_fragment(value.get("parts"))
        return ""

    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(value, "content", None)
    if content_attr is not None:
        return _extract_text_fragment(content_attr)

    parts_attr = getattr(value, "parts", None)
    if parts_attr is not None:
        return _extract_text_fragment(parts_attr)

    return ""


def _parse_json_fallback(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return json.loads(match.group(1))

    balanced_payload = _extract_first_balanced_json(text)
    if balanced_payload is not None:
        return json.loads(balanced_payload)

    raise ValueError("LLM response is not valid JSON")


def _extract_first_balanced_json(text: str) -> str | None:
    start_index: int | None = None
    opening = ""
    closing = ""
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start_index is None:
            if char == "{":
                start_index = index
                opening = "{"
                closing = "}"
                depth = 1
            elif char == "[":
                start_index = index
                opening = "["
                closing = "]"
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == opening:
            depth += 1
            continue

        if char == closing:
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]

    return None


@dataclass
class LLMSettings:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    client_id: str = ""
    client_secret: str = ""
    oauth_url: str = ""
    gateway_url: str = ""
    verify_ssl: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float = 60.0
    log_full_exchanges: bool = False
    streaming: bool = False
    default_response_schema_name: str = "structured_output"
    strict_json_schema: bool = True

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        verify_ssl = _is_truthy(os.getenv("LLM_VERIFY_SSL", "false"))
        log_full_exchanges = _is_truthy(os.getenv("LLM_LOG_FULL_EXCHANGES", "false"))
        streaming = _is_truthy(os.getenv("LLM_STREAMING", "false"))
        strict_json_schema = _is_truthy(os.getenv("LLM_STRICT_JSON_SCHEMA", "true"))

        temperature: float | None = None
        temp_raw = os.getenv("LLM_TEMPERATURE", "").strip()
        if temp_raw:
            try:
                temperature = float(temp_raw)
            except ValueError:
                temperature = None

        max_tokens: int | None = None
        max_tokens_raw = os.getenv("LLM_MAX_TOKENS", "").strip()
        if max_tokens_raw:
            try:
                max_tokens = int(max_tokens_raw)
            except ValueError:
                max_tokens = None

        timeout_seconds = 60.0
        timeout_raw = os.getenv("LLM_TIMEOUT_SECONDS", "").strip()
        if timeout_raw:
            try:
                timeout_seconds = float(timeout_raw)
            except ValueError:
                timeout_seconds = 60.0

        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", "").strip(),
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            client_id=os.getenv("LLM_CLIENT_ID", "").strip(),
            client_secret=os.getenv("LLM_CLIENT_SECRET", "").strip(),
            oauth_url=os.getenv("LLM_OAUTH_URL", "").strip(),
            gateway_url=os.getenv("LLM_GATEWAY_URL", "").strip(),
            verify_ssl=verify_ssl,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            log_full_exchanges=log_full_exchanges,
            streaming=streaming,
            default_response_schema_name=os.getenv("LLM_RESPONSE_SCHEMA_NAME", "structured_output").strip() or "structured_output",
            strict_json_schema=strict_json_schema,
        )


@dataclass
class LLMCompletionResult:
    provider: str
    model: str
    text: str
    parsed: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None
    used_streaming: bool = False


class LLMClient:
    SUPPORTED_PROVIDERS = SUPPORTED_PROVIDERS
    SUPPORTED_STREAMING_PROVIDERS = SUPPORTED_STREAMING_PROVIDERS
    SUPPORTED_NATIVE_RESPONSE_SCHEMA_PROVIDERS = SUPPORTED_NATIVE_RESPONSE_SCHEMA_PROVIDERS

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.last_error: str = ""
        self._logger = logging.getLogger("backend.app")

    def _set_error(self, message: str) -> None:
        self.last_error = message

    @staticmethod
    def _safe_get(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    @staticmethod
    def _excerpt(text: str, max_chars: int = 1200) -> str:
        collapsed = " ".join(str(text).split())
        if len(collapsed) <= max_chars:
            return collapsed
        return f"{collapsed[:max_chars]}...<truncated>"

    def _provider_system_prompt(self, system: str, *, enforce_json: bool) -> str:
        if not enforce_json:
            return system
        base = "Return valid JSON only. Do not include markdown fences or explanatory text."
        if system:
            return f"{system}\n\n{base}"
        return base

    def supports_streaming(self, provider: str | None = None) -> bool:
        return (provider or self.settings.provider) in self.SUPPORTED_STREAMING_PROVIDERS

    def supports_native_response_schema(self, provider: str | None = None) -> bool:
        return (provider or self.settings.provider) in self.SUPPORTED_NATIVE_RESPONSE_SCHEMA_PROVIDERS

    def enabled_with_reason(self) -> tuple[bool, str]:
        if self.settings.provider not in self.SUPPORTED_PROVIDERS:
            return (
                False,
                f"Unsupported LLM_PROVIDER '{self.settings.provider}'. Supported values: {sorted(self.SUPPORTED_PROVIDERS)}",
            )

        if not self.settings.model:
            return False, "LLM_MODEL is required."

        if self.settings.provider == "optusaifoundation":
            if not self.settings.client_id:
                return False, "LLM_CLIENT_ID is required for provider 'optusaifoundation'."
            if not self.settings.client_secret:
                return False, "LLM_CLIENT_SECRET is required for provider 'optusaifoundation'."
            return True, "ok"

        if not self.settings.api_key:
            return False, f"LLM_API_KEY is required for provider '{self.settings.provider}'."
        return True, "ok"

    def _normalize_response_format(
        self,
        *,
        response_schema: dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        response_schema_name: str | None,
        strict_json_schema: bool | None,
    ) -> dict[str, Any] | None:
        if response_schema is not None and response_format is not None:
            raise ValueError("Pass either response_schema or response_format, not both.")

        if response_format is not None:
            return response_format

        if response_schema is None:
            return None

        if "json_schema" in response_schema or response_schema.get("type") == "json_schema":
            normalized = dict(response_schema)
            if "type" not in normalized:
                normalized["type"] = "json_schema"
            return normalized

        if "schema" in response_schema and isinstance(response_schema.get("schema"), dict):
            json_schema_payload = dict(response_schema)
        else:
            json_schema_payload = {
                "name": response_schema_name or self.settings.default_response_schema_name,
                "schema": response_schema,
            }

        if "strict" not in json_schema_payload:
            json_schema_payload["strict"] = self.settings.strict_json_schema if strict_json_schema is None else strict_json_schema

        if "name" not in json_schema_payload:
            json_schema_payload["name"] = response_schema_name or self.settings.default_response_schema_name

        return {
            "type": "json_schema",
            "json_schema": json_schema_payload,
        }

    @staticmethod
    def _schema_fallback_block(response_format: dict[str, Any]) -> str:
        return (
            "\nReturn JSON matching this response_format exactly (fallback mode):\n"
            f"{json.dumps(response_format, ensure_ascii=True)}\n"
            "Do not include markdown fences or explanatory text.\n"
        )

    def _resolve_request_contract(
        self,
        *,
        prompt: str,
        system: str,
        response_schema: dict[str, Any] | None,
        response_format: dict[str, Any] | None,
        response_schema_name: str | None,
        strict_json_schema: bool | None,
        json_output: bool | None,
    ) -> tuple[str, str, dict[str, Any] | None, bool]:
        normalized_response_format = self._normalize_response_format(
            response_schema=response_schema,
            response_format=response_format,
            response_schema_name=response_schema_name,
            strict_json_schema=strict_json_schema,
        )
        should_parse_json = bool(json_output or normalized_response_format)
        provider_system = self._provider_system_prompt(system, enforce_json=should_parse_json)

        if normalized_response_format and not self.supports_native_response_schema():
            return (
                f"{prompt}{self._schema_fallback_block(normalized_response_format)}",
                provider_system,
                None,
                True,
            )

        return prompt, provider_system, normalized_response_format, should_parse_json

    def _build_messages(self, system_prompt: str, prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _merge_metadata(self, base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in incoming.items():
            if value is not None:
                merged[key] = value

        if "prompt_tokens" in merged and "input_tokens" not in merged:
            merged["input_tokens"] = merged["prompt_tokens"]
        if "completion_tokens" in merged and "output_tokens" not in merged:
            merged["output_tokens"] = merged["completion_tokens"]
        if "input_tokens" in merged and "prompt_tokens" not in merged:
            merged["prompt_tokens"] = merged["input_tokens"]
        if "output_tokens" in merged and "completion_tokens" not in merged:
            merged["completion_tokens"] = merged["output_tokens"]
        if "total_tokens" not in merged and "input_tokens" in merged and "output_tokens" in merged:
            merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
        return merged

    def _extract_response_envelope_fields(self, response: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        mapping = {
            "id": "response_id",
            "created": "created",
            "model": "response_model",
            "object": "response_object",
            "system_fingerprint": "system_fingerprint",
        }
        for source_key, target_key in mapping.items():
            value = self._safe_get(response, source_key)
            if value is not None:
                metadata[target_key] = value
        return metadata

    def _collect_response_metadata(self, response: Any) -> dict[str, Any]:
        metadata = self._extract_response_envelope_fields(response)
        if response is None:
            return metadata

        stream_context = self._safe_get(response, "stream_context")
        if isinstance(stream_context, dict):
            for key in (
                "response_headers",
                "response_status_code",
                "response_http_version",
                "request_headers",
                "request_url",
            ):
                value = stream_context.get(key)
                if value is not None:
                    metadata[key] = value

        choices = self._safe_get(response, "choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            finish_reason = self._safe_get(first_choice, "finish_reason")
            if finish_reason is None:
                finish_reason = self._safe_get(first_choice, "stop_reason")
            if finish_reason is not None:
                metadata["finish_reason"] = str(finish_reason)

        candidates = self._safe_get(response, "candidates")
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            finish_reason = self._safe_get(first_candidate, "finishReason")
            if finish_reason is not None:
                metadata["finish_reason"] = str(finish_reason)

        usage = self._safe_get(response, "usage")
        if usage is not None:
            metadata = self._merge_metadata(metadata, self._extract_usage_fields(usage))

        usage_metadata = self._safe_get(response, "usageMetadata")
        if usage_metadata is not None:
            metadata = self._merge_metadata(metadata, self._extract_usage_metadata_fields(usage_metadata))

        oaif = self._safe_get(response, "oaif")
        if oaif is not None:
            metadata = self._merge_metadata(metadata, self._extract_oaif_fields(oaif))

        return metadata

    def _extract_usage_fields(self, usage: Any) -> dict[str, Any]:
        data = _to_dict(usage) or {}
        if not isinstance(data, dict):
            return {}

        metadata: dict[str, Any] = {}
        for source_key, target_key in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
        ):
            value = data.get(source_key)
            if value is not None:
                metadata[target_key] = value

        details = data.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens")
            text_tokens = details.get("text_tokens")
            if reasoning_tokens is not None:
                metadata["reasoning_tokens"] = reasoning_tokens
            if text_tokens is not None:
                metadata["text_tokens"] = text_tokens

        prompt_details = data.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            prompt_detail_mapping = {
                "audio_tokens": "prompt_audio_tokens",
                "cache_write_tokens": "prompt_cache_write_tokens",
                "cached_tokens": "cached_prompt_tokens",
                "image_tokens": "prompt_image_tokens",
                "text_tokens": "prompt_text_tokens",
                "video_tokens": "prompt_video_tokens",
            }
            for source_key, target_key in prompt_detail_mapping.items():
                value = prompt_details.get(source_key)
                if value is not None:
                    metadata[target_key] = value

        cache_read_input_tokens = data.get("cache_read_input_tokens")
        if cache_read_input_tokens is not None:
            metadata["cache_read_input_tokens"] = cache_read_input_tokens

        return metadata

    def _extract_usage_metadata_fields(self, usage_metadata: Any) -> dict[str, Any]:
        data = _to_dict(usage_metadata) or {}
        if not isinstance(data, dict):
            return {}

        metadata: dict[str, Any] = {}
        mapping = {
            "promptTokenCount": "prompt_tokens",
            "candidatesTokenCount": "completion_tokens",
            "totalTokenCount": "total_tokens",
            "cachedContentTokenCount": "cached_prompt_tokens",
        }
        for source_key, target_key in mapping.items():
            value = data.get(source_key)
            if value is not None:
                metadata[target_key] = value

        for source_key, target_key in (
            ("thoughtsTokenCount", "reasoning_tokens"),
            ("toolUsePromptTokenCount", "tool_use_prompt_tokens"),
        ):
            value = data.get(source_key)
            if value is not None:
                metadata[target_key] = value
        return metadata

    def _extract_oaif_fields(self, oaif: Any) -> dict[str, Any]:
        data = _to_dict(oaif) or {}
        if not isinstance(data, dict):
            return {}

        metadata: dict[str, Any] = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "request_id",
            "apigee_message_id",
            "correlation_id",
            "dt_trace_id",
            "oaif_cost_usd",
            "cost_modifier_pct",
            "duration_s",
            "model",
            "cache_read_tokens",
            "cache_creation_tokens",
        ):
            value = data.get(key)
            if value is not None:
                if key == "model":
                    metadata["oaif_model"] = value
                else:
                    metadata[key] = value
        return metadata

    def _extract_response_text(self, response: Any) -> str:
        if response is None:
            return ""

        choices = self._safe_get(response, "choices")
        if isinstance(choices, list) and choices:
            message = self._safe_get(choices[0], "message")
            if message is not None:
                content = self._safe_get(message, "content")
                text = _extract_text_fragment(content)
                if text:
                    return text

        candidates = self._safe_get(response, "candidates")
        if isinstance(candidates, list) and candidates:
            content = self._safe_get(candidates[0], "content")
            parts = self._safe_get(content, "parts") if content is not None else None
            text = _extract_text_fragment(parts)
            if text:
                return text

        return ""

    def _extract_chunk_text(self, chunk: Any) -> str:
        choices = self._safe_get(chunk, "choices")
        if isinstance(choices, list) and choices:
            delta = self._safe_get(choices[0], "delta")
            if delta is not None:
                content = self._safe_get(delta, "content")
                text = _extract_text_fragment(content)
                if text:
                    return text

        candidates = self._safe_get(chunk, "candidates")
        if isinstance(candidates, list) and candidates:
            content = self._safe_get(candidates[0], "content")
            parts = self._safe_get(content, "parts") if content is not None else None
            text = _extract_text_fragment(parts)
            if text:
                return text

        return ""

    def _set_provider_runtime_error(
        self,
        *,
        provider: str,
        prompt: str,
        system_prompt: str,
        exc: Exception,
        raw_output: str = "",
        response: Any = None,
        phase: str = "runtime",
        extra: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        strict_json_schema: bool = False,
    ) -> None:
        metadata = self._collect_response_metadata(response)
        include_full = self.settings.log_full_exchanges
        debug_payload = {
            "provider": provider,
            "model": self.settings.model,
            "phase": phase,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "verify_ssl": self.settings.verify_ssl,
            "response_metadata": metadata,
            "extra": _redact_sensitive_data(extra or {}),
            "full_payload_logging_enabled": include_full,
            "response_format": response_format,
            "strict_json_schema": strict_json_schema,
        }
        if include_full:
            debug_payload["system_prompt_full"] = system_prompt
            debug_payload["user_prompt_full"] = prompt
            debug_payload["raw_output_full"] = raw_output
        else:
            debug_payload["system_prompt_excerpt"] = self._excerpt(system_prompt)
            debug_payload["user_prompt_excerpt"] = self._excerpt(prompt)
            debug_payload["raw_output_excerpt"] = self._excerpt(raw_output)
        self._set_error(f"{provider} completion failed: {json.dumps(debug_payload, ensure_ascii=True)}")

    def _set_structured_parse_error_with_metadata(
        self,
        *,
        provider: str,
        prompt: str,
        system_prompt: str,
        raw_output: str,
        exc: Exception,
        response: Any,
        response_format: dict[str, Any] | None,
        strict_json_schema: bool | None = None,
    ) -> None:
        self._set_provider_runtime_error(
            provider=provider,
            prompt=prompt,
            system_prompt=system_prompt,
            exc=exc,
            raw_output=raw_output,
            response=response,
            phase="parse_json",
            extra={"hint": "LLM response is not valid JSON"},
            response_format=response_format,
            strict_json_schema=self.settings.strict_json_schema if strict_json_schema is None else strict_json_schema,
        )

    def _log_exchange(
        self,
        *,
        provider: str,
        system_prompt: str,
        prompt: str,
        raw_output: str,
        parsed_output: Any,
        response_format: dict[str, Any] | None,
        response: Any,
    ) -> None:
        if not self.settings.log_full_exchanges:
            return

        payload = {
            "event": "llm_exchange",
            "provider": provider,
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "verify_ssl": self.settings.verify_ssl,
            "response_format": response_format,
            "response_metadata": self._collect_response_metadata(response),
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
        }
        self._logger.info("LLM exchange: %s", json.dumps(_redact_sensitive_data(payload), ensure_ascii=True))

    def _resolve_streaming(self, stream: bool | None) -> bool:
        desired_streaming = self.settings.streaming if stream is None else stream
        if not desired_streaming:
            return False
        return self.supports_streaming()

    def _resolve_litellm_model(self) -> str:
        model = self.settings.model.strip()
        if "/" in model:
            return model
        if self.settings.provider == "anthropic":
            return f"anthropic/{model}"
        if self.settings.provider == "google":
            # if self.settings.gateway_url:
            #     return f"openai/{model}"
            return f"gemini/{model}"
        return model

    def _load_litellm_completion(self) -> Callable[..., Any] | None:
        try:
            # Set prior to importing so LiteLLM reads this configuration during initialization
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            litellm_module = importlib.import_module("litellm")
            return getattr(litellm_module, "completion")
        except Exception as exc:
            self._set_error(f"LiteLLM unavailable: import failed ({exc}). Install dependency: pip install litellm")
            return None

    def _build_litellm_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        stream: bool,
        response_format: dict[str, Any] | None,
        header: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._resolve_litellm_model(),
            "messages": messages,
            "stream": stream,
            "timeout": self.settings.timeout_seconds,
        }
        if header:
            kwargs["extra_headers"] = header
        if self.settings.api_key:
            kwargs["api_key"] = self.settings.api_key
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        if self.settings.max_tokens is not None:
            kwargs["max_tokens"] = self.settings.max_tokens
        # if self.settings.gateway_url:
        #     kwargs["api_base"] = self.settings.gateway_url
        if response_format is not None:
            kwargs["response_format"] = response_format
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if not self.settings.verify_ssl:
            kwargs["ssl_verify"] = False
        return kwargs

    def stream_response_to_dict(self, response_stream: Any) -> dict[str, Any]:
        if response_stream is None:
            return {"chunks": []}

        try:
            iterator = iter(response_stream)
        except TypeError:
            single_value = _to_dict(response_stream)
            if single_value is None:
                return {"chunks": []}
            return {"chunks": [single_value]}

        chunks: list[Any] = []
        for chunk in iterator:
            chunks.append(_to_dict(chunk))
        return {"chunks": chunks}

    def _capture_stream_context(self, stream_response: Any) -> dict[str, Any]:
        context: dict[str, Any] = {}
        response = getattr(stream_response, "_response", None)
        if response is None:
            return context

        response_headers = getattr(response, "headers", None)
        if response_headers is not None:
            context["response_headers"] = dict(response_headers)

        response_status_code = getattr(response, "status_code", None)
        if response_status_code is not None:
            context["response_status_code"] = response_status_code

        response_http_version = getattr(response, "http_version", None)
        if response_http_version is not None:
            context["response_http_version"] = response_http_version

        request = getattr(response, "request", None)
        if request is not None:
            request_headers = getattr(request, "headers", None)
            if request_headers is not None:
                context["request_headers"] = _redact_sensitive_data(dict(request_headers))
            request_url = getattr(request, "url", None)
            if request_url is not None:
                context["request_url"] = str(request_url)

        return context

    def _run_stream(
        self,
        stream_response: Any,
        *,
        on_delta: Callable[[str], None] | None,
    ) -> tuple[str, dict[str, Any], Any]:
        parts: list[str] = []
        stream_payload: dict[str, Any] = {"chunks": []}
        stream_context = self._capture_stream_context(stream_response)
        if stream_context:
            stream_payload["stream_context"] = stream_context
        last_chunk_obj: Any = None

        for chunk in stream_response:
            last_chunk_obj = chunk
            stream_payload["chunks"].append(_to_dict(chunk))
            text = self._extract_chunk_text(chunk)
            if text:
                parts.append(text)
                if on_delta is not None:
                    on_delta(text)

        return "".join(parts), stream_payload, last_chunk_obj

    def _complete_via_litellm(
        self,
        *,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any] | None,
        should_parse_json: bool,
        header: dict[str, str] | None,
        stream: bool,
        on_delta: Callable[[str], None] | None,
    ) -> LLMCompletionResult | None:
        completion = self._load_litellm_completion()
        if completion is None:
            return None

        messages = self._build_messages(system_prompt, prompt)
        kwargs = self._build_litellm_kwargs(
            messages=messages,
            stream=stream,
            response_format=response_format,
            header=header,
        )

        effective_api_key = self.settings.api_key or os.environ.get("LLM_API_KEY", "").strip()
        # Set key in kwargs for request-scoped execution
        if effective_api_key and "api_key" not in kwargs:
            kwargs["api_key"] = effective_api_key

        try:
            litellm_module = importlib.import_module("litellm")
            if not self.settings.verify_ssl:
                litellm_module.ssl_verify = False
            # Optional fallback global key (safe if single-tenant / global key app)
            if effective_api_key:
                litellm_module.api_key = effective_api_key
        except Exception:
            pass

        try:
            if stream:
                response_stream = completion(**kwargs)
                text, stream_payload, last_chunk = self._run_stream(response_stream, on_delta=on_delta)
                metadata = self._merge_metadata(
                    self._collect_response_metadata(last_chunk),
                    self._collect_response_metadata(stream_payload),
                )
                raw_response = stream_payload
            else:
                response = completion(**kwargs)
                text = self._extract_response_text(response)
                metadata = self._collect_response_metadata(response)
                raw_response = _to_dict(response)

            if not text:
                self._set_error(f"{self.settings.provider} provider returned empty response content.")
                return None

            parsed: Any | None = None
            if should_parse_json:
                try:
                    parsed = _parse_json_fallback(text)
                except Exception as exc:
                    self._set_structured_parse_error_with_metadata(
                        provider=self.settings.provider,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        raw_output=text,
                        exc=exc,
                        response=raw_response,
                        response_format=response_format,
                        strict_json_schema=self.settings.strict_json_schema,
                    )
                    return None

            result = LLMCompletionResult(
                provider=self.settings.provider,
                model=self.settings.model,
                text=text,
                parsed=parsed,
                metadata=metadata,
                raw_response=raw_response,
                used_streaming=stream,
            )
            self._log_exchange(
                provider=self.settings.provider,
                system_prompt=system_prompt,
                prompt=prompt,
                raw_output=text,
                parsed_output=parsed,
                response_format=response_format,
                response=raw_response,
            )
            return result
        except Exception as exc:
            self._set_provider_runtime_error(
                provider=self.settings.provider,
                prompt=prompt,
                system_prompt=system_prompt,
                exc=exc,
                phase="request",
                extra={"request_kwargs": kwargs, "stream": stream},
                response_format=response_format,
                strict_json_schema=self.settings.strict_json_schema,
            )
            return None

    def _load_oaif_modules(self) -> tuple[Any, Any, Any] | None:
        try:
            optus_module = importlib.import_module("optus_ai")
            optus_auth_module = importlib.import_module("optus_ai.auth")
            optus_config_module = importlib.import_module("optus_ai.config")
            optus_ai_cls = getattr(optus_module, "OptusAI")
            standard_auth = getattr(optus_auth_module, "StandardAuth")
            default_model = getattr(optus_config_module, "DEFAULT_OAIF_MODEL")
            return optus_ai_cls, standard_auth, default_model
        except Exception as exc:
            self._set_error(
                "optusaifoundation provider unavailable: import failed "
                f"({exc}). Install private optus_ai SDK package."
            )
            return None

    def _build_oaif_client(self) -> tuple[Any, str] | None:
        loaded = self._load_oaif_modules()
        if loaded is None:
            return None
        optus_ai_cls, standard_auth, default_model = loaded

        auth_kwargs: dict[str, Any] = {
            "client_id": self.settings.client_id,
            "client_secret": self.settings.client_secret,
        }
        if self.settings.oauth_url:
            auth_kwargs["token_endpoint"] = self.settings.oauth_url

        auth = standard_auth(**auth_kwargs)

        client_kwargs: dict[str, Any] = {
            "auth": auth,
            "verify": self.settings.verify_ssl,
        }
        if self.settings.gateway_url:
            client_kwargs["base_url"] = self.settings.gateway_url
        if self.settings.model:
            client_kwargs["default_model"] = self.settings.model

        client = optus_ai_cls(**client_kwargs)
        return client, (self.settings.model or default_model)

    def _build_oaif_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        stream: bool,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        if self.settings.max_tokens is not None:
            kwargs["max_tokens"] = self.settings.max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def _execute_oaif_request(
        self,
        *,
        client: Any,
        kwargs: dict[str, Any],
        header: dict[str, str] | None,
        stream: bool,
        on_delta: Callable[[str], None] | None,
    ) -> tuple[str, dict[str, Any], Any]:
        if stream:
            response_stream = client.chat.completions.create(header=header, **kwargs)
            text, stream_payload, last_chunk = self._run_stream(response_stream, on_delta=on_delta)
            metadata = self._merge_metadata(
                self._collect_response_metadata(last_chunk),
                self._collect_response_metadata(stream_payload),
            )
            raw_response = stream_payload
            return text, metadata, raw_response

        response = client.chat.completions.create(header=header, **kwargs)
        text = self._extract_response_text(response)
        metadata = self._collect_response_metadata(response)
        raw_response = _to_dict(response)
        return text, metadata, raw_response

    def _complete_via_oaif(
        self,
        *,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any] | None,
        should_parse_json: bool,
        header: dict[str, str] | None,
        stream: bool,
        on_delta: Callable[[str], None] | None,
    ) -> LLMCompletionResult | None:
        built = self._build_oaif_client()
        if built is None:
            return None
        client, model_name = built
        messages = self._build_messages(system_prompt, prompt)
        kwargs = self._build_oaif_kwargs(
            messages=messages,
            model=model_name,
            stream=stream,
            response_format=response_format,
        )

        try:
            text, metadata, raw_response = self._execute_oaif_request(
                client=client,
                kwargs=kwargs,
                header=header,
                stream=stream,
                on_delta=on_delta,
            )
        except Exception as exc:
            self._set_provider_runtime_error(
                provider="optusaifoundation",
                prompt=prompt,
                system_prompt=system_prompt,
                exc=exc,
                phase="request",
                extra={"request_kwargs": kwargs, "request_header": header, "stream": stream},
            )
            return None

        if not text:
            self._set_error("optusaifoundation provider returned empty response content.")
            return None

        parsed: Any | None = None
        if should_parse_json:
            try:
                parsed = _parse_json_fallback(text)
            except Exception as exc:
                self._set_structured_parse_error_with_metadata(
                    provider="optusaifoundation",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    raw_output=text,
                    exc=exc,
                    response=raw_response,
                    response_format=response_format,
                    strict_json_schema=self.settings.strict_json_schema,
                )
                return None

        result = LLMCompletionResult(
            provider="optusaifoundation",
            model=model_name,
            text=text,
            parsed=parsed,
            metadata=metadata,
            raw_response=raw_response,
            used_streaming=stream,
        )
        self._log_exchange(
            provider="optusaifoundation",
            system_prompt=system_prompt,
            prompt=prompt,
            raw_output=text,
            parsed_output=parsed,
            response_format=response_format,
            response=raw_response,
        )
        return result

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        response_schema: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
        strict_json_schema: bool | None = None,
        json_output: bool | None = None,
        header: dict[str, str] | None = None,
        stream: bool | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMCompletionResult | None:
        enabled, reason = self.enabled_with_reason()
        if not enabled:
            self._set_error(reason)
            return None

        self.last_error = ""

        try:
            resolved_prompt, provider_system, native_response_format, should_parse_json = self._resolve_request_contract(
                prompt=prompt,
                system=system,
                response_schema=response_schema,
                response_format=response_format,
                response_schema_name=response_schema_name,
                strict_json_schema=strict_json_schema,
                json_output=json_output,
            )
        except Exception as exc:
            self._set_error(str(exc))
            return None

        resolved_streaming = self._resolve_streaming(stream)
        if stream and not resolved_streaming:
            self._logger.warning(
                "Streaming requested but unsupported for provider=%s; using non-streaming mode.",
                self.settings.provider,
            )

        if self.settings.provider == "optusaifoundation":
            return self._complete_via_oaif(
                prompt=resolved_prompt,
                system_prompt=provider_system,
                response_format=native_response_format,
                should_parse_json=should_parse_json,
                header=header,
                stream=resolved_streaming,
                on_delta=on_delta,
            )

        return self._complete_via_litellm(
            prompt=resolved_prompt,
            system_prompt=provider_system,
            response_format=native_response_format,
            should_parse_json=should_parse_json,
            header=header,
            stream=resolved_streaming,
            on_delta=on_delta,
        )