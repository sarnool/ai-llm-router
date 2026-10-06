# AI LLM Router SDK

A standalone Python chat-completion client configured explicitly through `LLMSettings`. The client supports LiteLLM-backed providers and direct HTTP gateways, with optional streaming, structured JSON output, and normalized response metadata.

## Features

- Explicit caller-supplied settings; no configuration or credentials are loaded from environment variables or application services.
- LiteLLM transport for supported providers, or direct HTTP (`provider_type="custom"`).
- API-key, caller-supplied authorization-header, and OAuth client-credentials authentication.
- Streaming text and metadata callbacks.
- Structured JSON output with provider JSON-schema support or schema instructions in the prompt.
- Response, usage, and timing metadata; sensitive authentication and cookie headers are redacted.

## Requirements

- Python 3.10 or later.
- `requests` is required to import the client and for direct HTTP/OAuth requests.
- The optional `litellm` dependency is required when using a non-custom provider.

The project is not yet published as an installable release. Local editable installation and test dependencies are documented under [Local development and testing](#local-development-and-testing). Published SDK users will only need runtime dependencies; they will not need the development extra.

## Quick start

The example prompts securely for an API key and makes a real provider call. It may incur provider charges. To use another LiteLLM provider such as Gemini, set `provider` and `model` to that provider's LiteLLM identifiers.

```python
from getpass import getpass

from ai_llm_router.llm_client import LLMClient, LLMSettings

api_key = getpass("Provider API key: ")
settings = LLMSettings(
    provider="openai",
    model="gpt-4.1-mini",
    provider_type="litellm",
    api_key=api_key,  # Supply a value managed by your application.
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

The example assumes the package has been installed. Tests mock provider responses and do not make live provider calls.

### Direct HTTP gateway

Set `provider_type="custom"` and provide `gateway_url` to use the direct HTTP transport. The endpoint is expected to accept a chat-completions-style JSON request. `requests` is used for both HTTP and OAuth client-credentials token requests.

## Public API

### `LLMSettings`

`provider` and `model` are required. All other fields have library defaults and are not loaded from external configuration.

| Field | Default | Purpose |
| --- | --- | --- |
| `provider` | Required | Provider name; used to qualify the LiteLLM model name. |
| `model` | Required | Model identifier sent with the completion request. |
| `provider_type` | `"litellm"` | Selects transport. `"custom"` selects direct HTTP; all other values select LiteLLM. |
| `api_key` | `""` | Explicit API key. |
| `client_id` | `""` | OAuth client ID for client-credentials authentication. |
| `client_secret` | `""` | OAuth client secret. |
| `oauth_url` | `""` | OAuth token endpoint. |
| `gateway_url` | `""` | Required for `custom`; optional LiteLLM `api_base`. |
| `verify_ssl` | `True` | TLS certificate verification for direct HTTP and OAuth requests. |
| `timeout_seconds` | `120.0` | Request timeout. |
| `max_tokens` | `4000` | Default completion-token limit; can be overridden per call. |
| `temperature` | `0.0` | Sampling temperature; `None` omits it from the provider request. |
| `streaming` | `True` | Whether calls stream by default. |
| `supports_streaming` | `True` | Capability guard; `False` disables streaming even if requested. |
| `supports_response_schema` | `True` | Use provider JSON-schema formatting, or include schema instructions in the prompt when `False`. |
| `extra_headers` | Empty dictionary | Extra request headers, including caller-managed authorization headers. |

### `LLMClient`

Construct with a caller-created `LLMSettings` object. `enabled_with_reason()` returns `(enabled, reason)` after checking the model, endpoint, and available authentication inputs; it does not contact a provider.

`complete(prompt, ...)` accepts these optional keyword arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `system` | `""` | Optional system message. |
| `response_schema` | `None` | JSON schema for structured output. Its presence enables JSON parsing by default. |
| `response_schema_name` | `"structured_response"` | Name attached to the provider JSON schema. |
| `strict_json_schema` | `True` | Request strict schema output where supported. |
| `json_output` | `None` | Defaults to `True` when a schema is provided; set `False` to return text without JSON parsing. |
| `max_tokens` | `None` | Per-call override; otherwise uses `LLMSettings.max_tokens`. |
| `headers` | `None` | Per-call HTTP headers merged over `LLMSettings.extra_headers`. |
| `stream` | `None` | Per-call override; otherwise uses `LLMSettings.streaming`, subject to `supports_streaming`. |
| `on_delta` | `None` | Callback receiving each non-empty streamed text fragment. |
| `on_metadata` | `None` | Callback receiving a copy of metadata after provider success or failure. It is not called for pre-request argument validation errors. |

### Response and errors

Successful calls return an `LLMResponse`:

| Field | Type | Meaning |
| --- | --- | --- |
| `text` | `str` | Complete response text, assembled from the response or stream. |
| `parsed` | `Any \| None` | Parsed JSON value when JSON output was requested; otherwise `None`. |
| `raw_response` | `dict[str, Any]` | Provider response payload (or the last payload received in a stream). |
| `metadata` | `dict[str, Any]` | Normalized request, response, timing, and usage details. |

Metadata includes provider/model, stream and response-format settings, request hyperparameters, and—when supplied by the transport/provider—request and response details, timing, finish reason, response identifiers, and token usage. Sensitive authorization, proxy-authorization, cookie, and set-cookie header values are redacted. Missing provider fields are omitted rather than synthesized.

The client also exposes `last_error`, `last_response_metadata`, `last_raw_response`, and `last_response_text`. The latter two retain the most recent successful response if a later call fails. Invalid arguments can raise `TypeError` or `ValueError`; transport, authentication, or invalid-response failures raise `LLMRequestError`.

## Local development and testing

The following setup is for contributors testing this repository. It installs the checkout in editable mode, along with LiteLLM and development tools; it is not an end-user installation command. Run it from the repository root using the Python environment you want to use for this project:

```powershell
python.exe -m pip install -e ".[litellm,dev]"
```

The `litellm` extra is included so you can run real requests through LiteLLM providers. The `dev` extra installs tools such as `pytest` and `build`. For unit tests, the built-in `unittest` runner is sufficient:

```powershell
python -m unittest discover -s tests/unit -v
```

If your organization intercepts PyPI TLS and pip reports a certificate verification error, use your organization's approved package index or CA certificate, for example `python.exe -m pip install --cert "C:\path\to\corporate-ca.pem" -e ".[litellm,dev]"`. Do not disable certificate verification.

## Tests

The unit tests mock provider responses; they do not make live network requests. The LiteLLM test uses a mocked module. Follow the local setup above before running the test command.

## Live integration check

The integration runner sends a real request to the configured provider. It can incur charges. Provide a key with `--api-key`, or omit it to enter the key through a hidden terminal prompt. For OAuth, pass `--client-id`, `--client-secret`, and `--oauth-url` instead. You can also repeat `--header NAME=VALUE` to set gateway-specific headers.

Run the Python integration script from the project root. It prompts for an API key without echoing it if you don't pass `--api-key`:

```powershell
python tests/integration/run_real_llm_test.py --provider openai --model gpt-4.1-mini --prompt "Say hello in one sentence." --stream
```

For direct HTTP, pass `--provider-type custom --gateway-url "https://your-gateway.example/v1/chat/completions"`. You can pass custom headers with `--header "X-Tenant=team-a"`; do not put secrets in commands that may be stored in shell history.

TLS certificate verification is enabled by default for direct HTTP and OAuth requests. Use `--no-verify-ssl` only when required for a controlled test environment with a self-signed certificate; disabling verification weakens connection security. `--verify-ssl` explicitly enables verification. These switches do not change TLS behavior for LiteLLM requests.

`--provider-type` defaults to `litellm`, so omit it when connecting through LiteLLM, including Gemini. For example:

```powershell
python tests/integration/run_real_llm_test.py --provider gemini --model gemini-2.5-flash --prompt "Say hello in one sentence."
```

Set `--provider-type custom` only when the script should send requests directly to an HTTP-compatible gateway; in that case, also pass `--gateway-url`.

Example using LiteLLM:

```powershell
python tests/integration/run_real_llm_test.py --provider openai --model gpt-4.1-mini --prompt "Say hello in one sentence."
```

The script prompts for an API key if needed. To pass one explicitly, add `--api-key YOUR_KEY` (be aware command-line arguments may be saved in shell history). To use a direct HTTP gateway, add `--provider-type custom --gateway-url https://your-gateway.example/v1/chat/completions`. Add `--stream` to print output as it arrives, or `--response-schema '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}'` to request structured JSON.

Install `requests` before running the script; install `litellm` as well when using the default LiteLLM transport. The direct HTTP integration runner still requires `requests`.

### Available arguments

| Argument | Required | Default | Purpose |
| --- | --- | --- | --- |
| `--prompt` | Yes | — | Prompt sent to the model. |
| `--model` | Yes | — | Model ID. |
| `--provider` | No | `openai` | Provider name; use `gemini` for Gemini models. |
| `--provider-type` | No | `litellm` | Selects `litellm` or `custom` direct HTTP. |
| `--api-key` | No | Secure prompt | API key. |
| `--system` | No | Empty | Optional system message. |
| `--gateway-url` | No | Empty | Gateway endpoint; required for custom HTTP. |
| `--client-id` | No | Empty | OAuth client ID. |
| `--client-secret` | No | Empty | OAuth client secret. |
| `--oauth-url` | No | Empty | OAuth token endpoint. |
| `--header NAME=VALUE` | No | — | Extra request header; repeat to add more. |
| `--timeout-seconds` | No | `120` | Request timeout. |
| `--verify-ssl` / `--no-verify-ssl` | No | `--verify-ssl` | Enable or disable TLS certificate verification for direct HTTP and OAuth requests. Disabling verification is not recommended and does not affect LiteLLM requests. |
| `--max-tokens` | No | `4000` | Maximum completion tokens. |
| `--temperature` | No | `0.0` | Sampling temperature. |
| `--stream` | No | Off | Print response as it arrives. |
| `--response-schema JSON` | No | None | JSON Schema object for structured output. |
| `--no-response-schema-support` | No | Off | Put schema instructions in the prompt instead of using provider schema mode. |

### Understanding the output

For a non-streaming request, the script prints `Response:` followed by the model's response text. If `--response-schema` was supplied, it also prints `Parsed JSON:` with the response parsed as a JSON value. The parsed value is convenient for applications to consume; the `Response:` section remains the original text returned by the model.

After a successful request, `Metadata:` is printed as a JSON object. Common fields include:

| Metadata field | Meaning |
| --- | --- |
| `status` | `"ok"` means the provider call and response handling succeeded. |
| `provider`, `model` | Provider and model configured for the call. |
| `response_model` | Model name reported in the provider response, when supplied. |
| `response_format_mode` | `"none"`, `"json_schema"`, or `"prompt_schema_instructions"`, describing how structured output was requested. |
| `request_hyperparams` | Request settings such as maximum tokens and temperature. |
| `latency_ms` | Request timing measured by the client, in milliseconds. |
| `response_id`, `request_id` | Provider response/request identifiers, when available. |
| `finish_reason` | Provider's reason for stopping generation, when available. |
| `prompt_tokens`, `completion_tokens`, `input_tokens`, `output_tokens`, `total_tokens`, `reasoning_tokens` | Token usage reported by the provider. Some providers report only a subset or none. |
| `chunk_count`, `time_to_first_chunk_ms` | Streaming information; present when the transport reports streamed chunks. |
| `request_url`, `response_status_code` | Transport details when provided by the selected transport. |

Metadata fields vary by provider and transport; absent fields are not necessarily errors. Header values for authorization, proxy authorization, cookies, and set-cookie are redacted. With `--stream`, text is printed as it arrives instead of under the `Response:` label; the metadata block is printed after the stream finishes. Request failures are printed to stderr as `LLM request failed: ...` and the script exits with a non-zero status. The script does not print a metadata block for failed requests.

## Limitations

Provider response formats and JSON-schema capabilities vary. Configure capability flags for the selected provider/model and validate behavior against it. LiteLLM is optional at runtime for the direct HTTP transport; consumers are responsible for installing and versioning dependencies appropriate to their deployment.
