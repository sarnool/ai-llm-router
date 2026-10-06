"""Standalone chat-completion client with explicitly injected configuration.

This module has no dependency on the application's config, database, logging
bootstrap, or environment variables. Supply an :class:`LLMSettings` instance
and, optionally, a LiteLLM installation when constructing the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any, Callable

import requests


_LOGGER = logging.getLogger(__name__)
_SENSITIVE_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie"}

__all__ = ["LLMClient", "LLMRequestError", "LLMResponse", "LLMSettings"]


@dataclass
class LLMSettings:
	"""All connection and generation settings for :class:`LLMClient`.

	``provider_type="custom"`` selects direct HTTP. Other provider types use
	LiteLLM. No settings are loaded or inferred from the process environment.
	"""

	provider: str
	model: str
	provider_type: str = "litellm"
	api_key: str = ""
	client_id: str = ""
	client_secret: str = ""
	oauth_url: str = ""
	gateway_url: str = ""
	verify_ssl: bool = True
	timeout_seconds: float = 120.0
	max_tokens: int = 4000
	temperature: float | None = 0.0
	streaming: bool = True
	supports_streaming: bool = True
	supports_response_schema: bool = True
	extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
	"""One completion, including provider payload and normalized metadata."""

	text: str
	parsed: Any | None
	raw_response: dict[str, Any]
	metadata: dict[str, Any]


class LLMRequestError(RuntimeError):
	"""Raised when credentials, transport, or response handling fails."""


class LLMClient:
	"""Connect to a chat-completion provider using only injected settings."""

	def __init__(self, settings: LLMSettings):
		self.settings = settings
		self.last_error: str | None = None
		self.last_response_metadata: dict[str, Any] = {}
		self.last_raw_response: dict[str, Any] = {}
		self.last_response_text = ""
		self._oauth_access_token = ""
		self._oauth_access_token_expires_at = 0.0

	def enabled_with_reason(self) -> tuple[bool, str]:
		if not self.settings.model.strip():
			return False, "A model name must be supplied in LLMSettings."
		if self.settings.provider_type.strip().lower() == "custom" and not self.settings.gateway_url.strip():
			return False, "gateway_url must be supplied for direct HTTP providers."
		if self.settings.api_key.strip():
			return True, "configured"
		if self.settings.client_id.strip() and self.settings.client_secret.strip() and self.settings.oauth_url.strip():
			return True, "configured"
		if self.settings.provider_type.lower() == "custom" and self._has_authorization_header():
			return True, "configured"
		return False, "Provide an API key, OAuth client credentials, or an Authorization header."

	def complete(
		self,
		prompt: str,
		*,
		system: str = "",
		response_schema: dict[str, Any] | None = None,
		response_schema_name: str = "structured_response",
		strict_json_schema: bool = True,
		json_output: bool | None = None,
		max_tokens: int | None = None,
		headers: dict[str, str] | None = None,
		stream: bool | None = None,
		on_delta: Callable[[str], None] | None = None,
		on_metadata: Callable[[dict[str, Any]], None] | None = None,
	) -> LLMResponse:
		"""Run a completion and return text, parsed output, raw payload, metadata.

		Provider/network errors raise :class:`LLMRequestError`. If ``on_delta``
		is supplied, streamed text fragments are passed to it as they arrive.
		``on_metadata`` receives a copy of the normalized metadata after the
		response (or after an error).
		"""
		if not isinstance(prompt, str):
			raise TypeError("prompt must be a string")
		if not self.settings.model.strip():
			raise ValueError("LLMSettings.model is required")

		wants_json = response_schema is not None if json_output is None else bool(json_output)
		if wants_json and response_schema is None:
			raise ValueError("response_schema is required when json_output is enabled")

		requested_stream = self.settings.streaming if stream is None else bool(stream)
		use_stream = requested_stream and self.settings.supports_streaming
		normalized_schema = self._normalize_schema(response_schema) if response_schema is not None else None
		request_headers = self._request_headers(headers or {})
		messages: list[dict[str, str]] = []
		if system.strip():
			messages.append({"role": "system", "content": system})
		messages.append({"role": "user", "content": prompt})

		response_format: dict[str, Any] | None = None
		response_format_mode = "none"
		if wants_json and self.settings.supports_response_schema:
			response_format = {
				"type": "json_schema",
				"json_schema": {
					"name": response_schema_name or "structured_response",
					"strict": strict_json_schema,
					"schema": normalized_schema,
				},
			}
			response_format_mode = "json_schema"
		elif wants_json:
			schema_instruction = json.dumps(normalized_schema, ensure_ascii=True)
			messages[-1] = {
				"role": "user",
				"content": f"{prompt}\n\nReturn only valid JSON matching this schema:\n{schema_instruction}",
			}
			response_format_mode = "prompt_schema_instructions"

		request_body: dict[str, Any] = {
			"model": self.settings.model,
			"messages": messages,
			"stream": use_stream,
			"max_tokens": max(1, int(max_tokens if max_tokens is not None else self.settings.max_tokens)),
		}
		if self.settings.temperature is not None:
			request_body["temperature"] = float(self.settings.temperature)
		if response_format is not None:
			request_body["response_format"] = response_format

		started = time.perf_counter()
		metadata: dict[str, Any] = {
			"provider": self.settings.provider,
			"model": self.settings.model,
			"stream": use_stream,
			"supports_streaming": self.settings.supports_streaming,
			"supports_response_schema": self.settings.supports_response_schema,
			"response_format_mode": response_format_mode,
			"request_hyperparams": {
				key: request_body[key]
				for key in ("max_tokens", "temperature")
				if key in request_body
			},
		}
		raw_response: dict[str, Any] = {}
		self.last_error = None
		try:
			if self.settings.provider_type.strip().lower() == "custom":
				text, raw_response = self._request_http(
					request_body, request_headers, use_stream, on_delta, metadata, started
				)
			else:
				text, raw_response = self._request_litellm(
					request_body, request_headers, use_stream, on_delta, metadata, started
				)

			self._extract_response_metadata(metadata, raw_response)
			if not text:
				text = self._extract_text(raw_response)
			parsed = self._parse_json(text) if wants_json else None
			self.last_response_text = text
			self.last_raw_response = raw_response
			self.last_response_metadata = metadata
			result = LLMResponse(text=text, parsed=parsed, raw_response=raw_response, metadata=metadata)
			self._notify_metadata(on_metadata, metadata)
			return result
		except Exception as exc:
			self.last_error = str(exc)
			metadata["status"] = "error"
			metadata["error"] = self.last_error
			metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
			self.last_response_metadata = metadata
			self._notify_metadata(on_metadata, metadata)
			if isinstance(exc, (ValueError, TypeError)):
				raise
			raise LLMRequestError(self.last_error) from exc

	def _request_headers(self, call_headers: dict[str, str]) -> dict[str, str]:
		headers = {str(key): str(value) for key, value in self.settings.extra_headers.items()}
		headers.update({str(key): str(value) for key, value in call_headers.items()})
		if not self._header(headers, "content-type"):
			headers["Content-Type"] = "application/json"
		if not self._header(headers, "authorization"):
			if self.settings.api_key.strip():
				headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"
			elif self.settings.client_id.strip() and self.settings.client_secret.strip() and self.settings.oauth_url.strip():
				headers["Authorization"] = f"Bearer {self._get_oauth_access_token()}"
		return headers

	def _has_authorization_header(self) -> bool:
		return self._header(self.settings.extra_headers, "authorization") is not None

	@staticmethod
	def _header(headers: dict[str, Any], name: str) -> str | None:
		for key, value in headers.items():
			if str(key).lower() == name.lower() and str(value).strip():
				return str(value).strip()
		return None

	@staticmethod
	def _safe_headers(headers: dict[str, Any]) -> dict[str, str]:
		return {
			str(key): "[REDACTED]" if str(key).lower() in _SENSITIVE_HEADERS else str(value)
			for key, value in headers.items()
		}

	def _get_oauth_access_token(self) -> str:
		now = time.time()
		if self._oauth_access_token and now < self._oauth_access_token_expires_at:
			return self._oauth_access_token
		if not (self.settings.oauth_url.strip() and self.settings.client_id.strip() and self.settings.client_secret.strip()):
			raise ValueError("oauth_url, client_id, and client_secret are required for OAuth")

		response = requests.post(
			self.settings.oauth_url,
			data={"grant_type": "client_credentials"},
			auth=(self.settings.client_id, self.settings.client_secret),
			headers={"Content-Type": "application/x-www-form-urlencoded"},
			timeout=self.settings.timeout_seconds,
			verify=self.settings.verify_ssl,
		)
		if response.status_code >= 400:
			raise LLMRequestError(f"OAuth token request failed with HTTP {response.status_code}")
		payload = response.json() if response.content else {}
		token = str(payload.get("access_token", "") or "").strip()
		if not token:
			raise LLMRequestError("OAuth token response did not include access_token")
		try:
			expires_in = max(60, int(payload.get("expires_in", 300)))
		except (TypeError, ValueError):
			expires_in = 300
		self._oauth_access_token = token
		self._oauth_access_token_expires_at = now + max(30, expires_in - 30)
		return token

	def _request_url(self) -> str:
		url = self.settings.gateway_url.strip()
		if url:
			return url
		raise ValueError("gateway_url is required for direct HTTP provider_type='custom'")

	def _request_http(
		self,
		body: dict[str, Any],
		headers: dict[str, str],
		stream: bool,
		on_delta: Callable[[str], None] | None,
		metadata: dict[str, Any],
		started: float,
	) -> tuple[str, dict[str, Any]]:
		url = self._request_url()
		metadata["request_url"] = url
		metadata["request_headers"] = self._safe_headers(headers)
		response = requests.post(
			url,
			json=body,
			headers=headers,
			stream=stream,
			timeout=self.settings.timeout_seconds,
			verify=self.settings.verify_ssl,
		)
		metadata["response_status_code"] = response.status_code
		metadata["response_headers"] = self._safe_headers(dict(response.headers))
		metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
		if response.status_code >= 400:
			raise LLMRequestError(f"LLM request failed with HTTP {response.status_code}: {response.text[:1000]}")

		if not stream:
			payload = response.json() if response.content else {}
			raw = payload if isinstance(payload, dict) else {}
			self._update_usage(metadata, raw)
			return self._extract_text(raw), raw

		chunks: list[str] = []
		last_payload: dict[str, Any] = {}
		self._consume_stream(
			response.iter_lines(decode_unicode=True), chunks, last_payload, metadata, on_delta, started
		)
		return "".join(chunks), last_payload

	def _request_litellm(
		self,
		body: dict[str, Any],
		headers: dict[str, str],
		stream: bool,
		on_delta: Callable[[str], None] | None,
		metadata: dict[str, Any],
		started: float,
	) -> tuple[str, dict[str, Any]]:
		try:
			from litellm import completion
		except Exception as exc:
			raise LLMRequestError("LiteLLM is required for non-custom providers") from exc

		provider = self.settings.provider.strip()
		model = self.settings.model.strip()
		if provider and provider.lower() != "system_supplied" and not model.startswith(f"{provider}/"):
			model = f"{provider}/{model}"
		kwargs: dict[str, Any] = {
			"model": model,
			"messages": body["messages"],
			"stream": stream,
			"max_tokens": body["max_tokens"],
			"timeout": self.settings.timeout_seconds,
		}
		if "temperature" in body:
			kwargs["temperature"] = body["temperature"]
		if self.settings.gateway_url.strip():
			kwargs["api_base"] = self.settings.gateway_url.strip()
		if self.settings.api_key.strip():
			kwargs["api_key"] = self.settings.api_key.strip()
		if "response_format" in body:
			kwargs["response_format"] = body["response_format"]
		safe_headers = self._safe_headers(headers)
		if safe_headers:
			kwargs["extra_headers"] = headers

		metadata["request_url"] = self.settings.gateway_url.strip() or None
		metadata["request_headers"] = safe_headers
		response = completion(**kwargs)
		metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
		if stream:
			chunks: list[str] = []
			last_payload: dict[str, Any] = {}
			self._consume_stream(response, chunks, last_payload, metadata, on_delta, started)
			return "".join(chunks), last_payload
		payload = self._to_dict(response)
		self._update_usage(metadata, payload)
		return self._extract_text(payload), payload

	def _consume_stream(
		self,
		source: Any,
		chunks: list[str],
		last_payload: dict[str, Any],
		metadata: dict[str, Any],
		on_delta: Callable[[str], None] | None,
		started: float,
	) -> None:
		chunk_count = 0
		first_chunk_at: float | None = None
		for item in source:
			payload: dict[str, Any] = {}
			if isinstance(item, (str, bytes)):
				line = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
				line = line.strip()
				if not line or not line.startswith("data:"):
					continue
				line = line[5:].strip()
				if line == "[DONE]":
					break
				try:
					decoded = json.loads(line)
				except json.JSONDecodeError:
					continue
				payload = decoded if isinstance(decoded, dict) else {}
			else:
				payload = self._to_dict(item)
			if not payload:
				continue
			last_payload.update(payload)
			self._update_usage(metadata, payload)
			finish_reason = self._finish_reason(payload)
			if finish_reason:
				metadata["finish_reason"] = finish_reason
			delta = self._extract_chunk_text(payload)
			if delta:
				chunk_count += 1
				if first_chunk_at is None:
					first_chunk_at = time.perf_counter()
				chunks.append(delta)
				if on_delta is not None:
					on_delta(delta)
		metadata["chunk_count"] = chunk_count
		if first_chunk_at is not None:
			metadata["time_to_first_chunk_ms"] = round((first_chunk_at - started) * 1000, 3)

	@staticmethod
	def _to_dict(value: Any) -> dict[str, Any]:
		if isinstance(value, dict):
			return value
		for method_name in ("model_dump", "to_dict"):
			method = getattr(value, method_name, None)
			if callable(method):
				result = method()
				if isinstance(result, dict):
					return result
		return {}

	@staticmethod
	def _extract_chunk_text(payload: dict[str, Any]) -> str:
		choices = payload.get("choices")
		if not isinstance(choices, list) or not choices:
			return ""
		first = choices[0] if isinstance(choices[0], dict) else {}
		delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
		content = delta.get("content")
		if isinstance(content, str):
			return content
		if isinstance(content, list):
			return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
		message = first.get("message") if isinstance(first.get("message"), dict) else {}
		content = message.get("content")
		if isinstance(content, str):
			return content
		if isinstance(content, list):
			return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
		return str(first.get("text", "") or "")

	@staticmethod
	def _extract_text(payload: dict[str, Any]) -> str:
		choices = payload.get("choices")
		if isinstance(choices, list) and choices:
			first = choices[0] if isinstance(choices[0], dict) else {}
			message = first.get("message") if isinstance(first.get("message"), dict) else {}
			content = message.get("content")
			if isinstance(content, str):
				return content
			if isinstance(content, list):
				return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
		return str(payload.get("output_text", "") or "")

	@staticmethod
	def _finish_reason(payload: dict[str, Any]) -> str | None:
		choices = payload.get("choices")
		if isinstance(choices, list):
			for choice in choices:
				if isinstance(choice, dict):
					reason = choice.get("finish_reason") or choice.get("stop_reason")
					if reason:
						return str(reason)
		reason = payload.get("finish_reason") or payload.get("stop_reason")
		return str(reason) if reason else None

	@staticmethod
	def _update_usage(metadata: dict[str, Any], payload: dict[str, Any]) -> None:
		usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
		details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
		oaif = payload.get("oaif") if isinstance(payload.get("oaif"), dict) else {}
		fields = {
			"prompt_tokens": usage.get("prompt_tokens", oaif.get("input_tokens")),
			"completion_tokens": usage.get("completion_tokens", oaif.get("output_tokens")),
			"total_tokens": usage.get("total_tokens"),
			"input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", oaif.get("input_tokens"))),
			"output_tokens": usage.get("output_tokens", usage.get("completion_tokens", oaif.get("output_tokens"))),
			"reasoning_tokens": details.get("reasoning_tokens", oaif.get("reasoning_tokens")),
		}
		for key, value in fields.items():
			if value is None or isinstance(value, bool):
				continue
			try:
				value = int(value)
			except (TypeError, ValueError):
				continue
			if value >= 0:
				metadata[key] = value

	@classmethod
	def _extract_response_metadata(cls, metadata: dict[str, Any], payload: dict[str, Any]) -> None:
		metadata["response_id"] = payload.get("id") or metadata.get("response_id")
		metadata["response_model"] = payload.get("model") or metadata.get("response_model")
		metadata["response_object"] = payload.get("object") or metadata.get("response_object")
		metadata["created"] = payload.get("created") or metadata.get("created")
		finish_reason = cls._finish_reason(payload)
		if finish_reason:
			metadata["finish_reason"] = finish_reason
		cls._update_usage(metadata, payload)
		response_headers = metadata.get("response_headers", {})
		if isinstance(response_headers, dict):
			request_id = cls._header(response_headers, "x-request-id")
			if request_id:
				metadata["request_id"] = request_id
		metadata["status"] = "ok"

	@staticmethod
	def _normalize_schema(schema: Any) -> Any:
		if isinstance(schema, dict):
			schema_type = schema.get("type")
			numeric = schema_type in {"number", "integer"} if isinstance(schema_type, str) else False
			result: dict[str, Any] = {}
			for key, value in schema.items():
				if key in {"minItems", "maxItems"}:
					continue
				if numeric and key in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}:
					continue
				if key == "additionalProperties" and value is True:
					continue
				result[key] = LLMClient._normalize_schema(value)
			return result
		if isinstance(schema, list):
			return [LLMClient._normalize_schema(value) for value in schema]
		return schema

	@classmethod
	def _parse_json(cls, text: str) -> Any:
		normalized = cls._coerce_json_text(text)
		try:
			return json.loads(normalized)
		except json.JSONDecodeError:
			decoder = json.JSONDecoder()
			for index, character in enumerate(text):
				if character not in "[{":
					continue
				try:
					value, _ = decoder.raw_decode(text[index:])
					return value
				except json.JSONDecodeError:
					continue
			repaired = cls._repair_truncated_json(normalized)
			if repaired is not None:
				return json.loads(repaired)
		raise LLMRequestError("The provider response did not contain valid JSON")

	@staticmethod
	def _repair_truncated_json(value: str) -> str | None:
		"""Close an incomplete JSON value, dropping only its trailing fragment."""
		text = str(value or "").strip()
		if not text or text[0] not in "[{":
			return None
		closers: list[str] = []
		in_string = False
		escaped = False
		for character in text:
			if in_string:
				if escaped:
					escaped = False
				elif character == "\\":
					escaped = True
				elif character == '"':
					in_string = False
				continue
			if character == '"':
				in_string = True
			elif character in "[{":
				closers.append("}" if character == "{" else "]")
			elif character in "}]" and closers:
				closers.pop()
		if not closers and not in_string:
			return None

		repaired = text
		if in_string:
			repaired = re.sub(r"\\u[0-9a-fA-F]{0,3}$", "", repaired)
			if repaired.endswith("\\") and not repaired.endswith("\\\\"):
				repaired = repaired[:-1]
			repaired += '"'
		suffix = "".join(reversed(closers))
		while repaired:
			candidate = repaired.rstrip().rstrip(" ,:").rstrip() + suffix
			try:
				json.loads(candidate)
				return candidate
			except json.JSONDecodeError:
				cut = repaired.rstrip().rstrip(" ,:").rstrip().rfind(",")
				if cut <= 0:
					return None
				repaired = repaired[:cut]
		return None

	@staticmethod
	def _coerce_json_text(value: str) -> str:
		text = str(value or "").strip()
		if text.startswith("```"):
			lines = text.splitlines()[1:]
			if lines and lines[-1].strip().startswith("```"):
				lines = lines[:-1]
			text = "\n".join(lines).strip()
		if text.lower().startswith("json\n"):
			text = text[5:].strip()
		starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
		if starts:
			text = text[min(starts):]
		text = text.replace("\\'", "'")
		ends = [index for index in (text.rfind("}"), text.rfind("]")) if index >= 0]
		if ends:
			text = text[: max(ends) + 1]
		return text.strip()

	@staticmethod
	def _notify_metadata(
		callback: Callable[[dict[str, Any]], None] | None,
		metadata: dict[str, Any],
	) -> None:
		if callback is not None:
			try:
				callback(dict(metadata))
			except Exception:
				_LOGGER.exception("LLM metadata callback failed")
