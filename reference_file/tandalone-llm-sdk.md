# Change Report: Standalone LLM Client SDK

## Classification

- Risk: medium
- Type: isolated SDK/client addition; existing backend call path unchanged

## Impacted Files

- `apps/backend/src/standalone_llm_client.py`
- `tests/backend/unit/test_standalone_llm_client.py`
- `doc/change-report-standalone-llm-sdk.md`

## Behavior

- Added a standalone LLM client configured only through a caller-supplied `LLMSettings` object and per-call arguments.
- Added LiteLLM and custom direct-HTTP transports, streaming callbacks, structured JSON parsing, and returned/callback response metadata.
- Structured JSON parsing includes recovery for common wrappers and truncated output; metadata redacts authorization/cookie headers.
- The module does not load application configuration, environment variables, database settings, or backend logging/usage services.
- Existing `apps/backend/src/llm_client.py` and its workflow integration were not changed.

## Validation

- Focused standalone client unit tests: 6 passed (provider transports mocked; no live provider call).
- Pylance diagnostics: no errors in the new source or tests.
- Python syntax checks: passed for the new source and tests.
- `git diff --check`: passed.

## SDK Usage and API Reference

The client has no hidden configuration lookup: callers create `LLMSettings`
and pass it to `LLMClient`. The example below uses a caller-provided API key;
the SDK does not load that key from an environment variable or file.

```python
from apps.backend.src.standalone_llm_client import LLMClient, LLMSettings

settings = LLMSettings(
	provider="openai",
	model="gpt-4.1-mini",
	provider_type="litellm",
	api_key=api_key,  # Obtain and manage this value in the calling application.
	streaming=True,
	supports_streaming=True,
	supports_response_schema=True,
	max_tokens=1200,
	temperature=0.2,
	timeout_seconds=90,
)
client = LLMClient(settings)

response = client.complete(
	"Return a short greeting.",
	system="You are a concise assistant.",
	response_schema={
		"type": "object",
		"properties": {"greeting": {"type": "string"}},
		"required": ["greeting"],
		"additionalProperties": False,
	},
	on_delta=lambda text: print(text, end="", flush=True),
)

print(response.parsed)    # Parsed JSON object
print(response.metadata)  # Normalized response and usage metadata
```

For direct HTTP, set `provider_type="custom"` and provide `gateway_url`. The
consumer must install `requests` for custom HTTP/OAuth and LiteLLM for all other
provider types.

### Constructor and helper methods

- `LLMSettings(...)` accepts the fields listed below.
- `LLMClient(settings)` takes one required argument: the caller-created
	`LLMSettings` instance.
- `enabled_with_reason()` takes no arguments and returns a two-element tuple:
	a boolean and a reason string. It checks the configured model, transport
	endpoint, and available authentication inputs; it does not contact a provider.

### `LLMSettings` fields

| Field | Required / default | Purpose |
| --- | --- | --- |
| `provider` | Required | Provider name; used to qualify the LiteLLM model name. |
| `model` | Required | Model identifier sent with the completion request. |
| `provider_type` | `"litellm"` | Selects transport. `"custom"` selects direct HTTP; all other values select LiteLLM. |
| `api_key` | `""` | Explicit API key. |
| `client_id` | `""` | OAuth client ID when using client-credentials authentication. |
| `client_secret` | `""` | OAuth client secret. |
| `oauth_url` | `""` | OAuth token endpoint. |
| `gateway_url` | `""` | Required endpoint for `custom`; optional LiteLLM `api_base`. |
| `verify_ssl` | `True` | TLS certificate verification for direct HTTP and OAuth requests. |
| `timeout_seconds` | `120.0` | Request timeout. |
| `max_tokens` | `4000` | Default completion-token limit; can be overridden per call. |
| `temperature` | `0.0` | Default sampling temperature; `None` omits it from the provider request. |
| `streaming` | `True` | Whether calls stream by default. |
| `supports_streaming` | `True` | Capability guard; when false, requests are non-streaming even if requested. |
| `supports_response_schema` | `True` | Use provider JSON-schema response formatting when true; otherwise put the schema instructions in the user prompt. |
| `extra_headers` | Empty dictionary | Additional request headers, including a caller-managed authorization header if desired. |

Only `provider` and `model` are mandatory constructor fields. The listed
defaults are library defaults, not values loaded from external configuration.

### `LLMClient.complete(...)` arguments

`prompt` is the required positional prompt. The following keyword arguments are
available:

| Argument | Default | Purpose |
| --- | --- | --- |
| `system` | `""` | Optional system message. |
| `response_schema` | `None` | JSON schema for structured output. Required when `json_output=True`; presence enables JSON parsing by default. |
| `response_schema_name` | `"structured_response"` | Name attached to the provider JSON schema. |
| `strict_json_schema` | `True` | Requests strict schema output when the provider supports it. |
| `json_output` | `None` | Defaults to true when a schema is supplied; false returns text without JSON parsing. |
| `max_tokens` | `None` | Per-call completion-token override; otherwise uses `LLMSettings.max_tokens`. |
| `headers` | `None` | Per-call HTTP headers merged over `LLMSettings.extra_headers`. |
| `stream` | `None` | Per-call streaming override; otherwise uses `LLMSettings.streaming`, subject to `supports_streaming`. |
| `on_delta` | `None` | Receives each non-empty streamed text fragment. |
| `on_metadata` | `None` | Receives a copy of normalized metadata after a provider call succeeds or fails. It is not called for argument validation errors raised before the provider call. |

### Outputs and errors

Successful calls return `LLMResponse`:

| Output | Type | Meaning |
| --- | --- | --- |
| `text` | `str` | Complete response text assembled from the provider response or stream. |
| `parsed` | `Any | None` | Parsed JSON value when JSON output was requested; otherwise `None`. |
| `raw_response` | `dict[str, Any]` | Raw/last provider response payload. |
| `metadata` | `dict[str, Any]` | Normalized call, response, timing, and usage details. |

Metadata always starts with `provider`, `model`, `stream`,
`supports_streaming`, `supports_response_schema`, `response_format_mode`, and
`request_hyperparams`. Depending on transport and provider fields, it can also
contain `request_url`, `request_headers`, `response_status_code`,
`response_headers`, `latency_ms`, `status`, `response_id`, `response_model`,
`response_object`, `created`, `request_id`, `finish_reason`, `prompt_tokens`,
`completion_tokens`, `input_tokens`, `output_tokens`, `total_tokens`,
`reasoning_tokens`, `chunk_count`, and `time_to_first_chunk_ms`. Header values
for authorization, proxy authorization, cookies, and set-cookie are redacted.
Fields absent from a provider response are omitted rather than synthesized.

The client also exposes `last_error` and `last_response_metadata` for the most
recent attempted call. `last_raw_response` and `last_response_text` retain the
latest response accepted by the client; an earlier successful value can remain
there if a later call fails. `on_delta` receives fragments only when streaming
is actually enabled. Invalid arguments can raise `TypeError` or `ValueError`;
transport, authentication, and invalid-response failures raise
`LLMRequestError` and do not return an `LLMResponse`.

## Documentation

- This report now documents the standalone SDK's usage, arguments, outputs, and error behavior.
- The optional `doc/reference/llm-integration.md` update from the initial implementation was reverted by the user and was not changed in this update.
- Reviewed the existing report and `standalone_llm_client.py` public signatures; the referenced harness template path was not present in the workspace.

## Residual Risk

- Provider-specific response shapes and schema support vary. Consumers must set capability flags explicitly and validate against their provider/model.
- LiteLLM is an optional runtime dependency for non-custom providers; the consumer is responsible for installing and versioning it.
- The configured workspace interpreter did not have `requests` installed, so tests used a mocked `requests` module; the standalone client requires `requests` for direct HTTP/OAuth at runtime.