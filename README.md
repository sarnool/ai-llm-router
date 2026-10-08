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
- `litellm` is installed as a core dependency and is used by the LiteLLM transport.

The package is not published on PyPI. Install it from GitHub or from a wheel as shown below. Development and test setup is documented under [Local development and testing](#local-development-and-testing); SDK consumers do not need the development extra.

## Quick start

Install the SDK into the Python environment used by your application. It requires Python 3.10 or later. The package is not published on PyPI; install the wheel distributed with the project, or use the GitHub installation as an alternative.

### Install the distributed wheel

Run pip against the `.whl` file's location. If the wheel is in this repository's `dist` directory, use:

```powershell
python -m pip install ".\dist\ai_llm_router-0.1.0-py3-none-any.whl"
```

If you downloaded or received the wheel in another location, replace the example path with that file's path, for example `C:\Downloads\ai_llm_router-0.1.0-py3-none-any.whl`. Pip installs the SDK's declared dependencies, including LiteLLM, automatically.

### Install from GitHub

```powershell
python -m pip install "ai-llm-router @ git+https://github.com/sarnool/ai-llm-router.git"
```

### Use the SDK

Replace the API-key placeholder with your key. Keep real credentials out of source control. This example makes a real provider call and may incur charges.

```python
from ai_llm_router import LLMClient, LLMSettings

settings = LLMSettings(
    provider="openai",
    model="gpt-4.1-mini",
    api_key="YOUR_API_KEY",
    stream=False,
)
client = LLMClient(settings)
response = client.complete("Say hello in one sentence.")
print(response.text)
```

`LLMClient`, `LLMSettings`, `LLMResponse`, and `LLMRequestError` are available from the top-level `ai_llm_router` package. Tests mock provider responses and do not make live provider calls.

### Direct HTTP gateway

Set `provider_type="custom"` and provide `gateway_url` to use the direct HTTP transport. **The custom transport currently supports only the OpenAI Chat Completions-compatible format**: it sends a request with `messages` and expects a response containing `choices` and message content. The endpoint may belong to any provider or gateway, but it must implement that compatible request/response format. Provider-native formats such as Gemini `generateContent` (`contents` / `parts`) are not supported directly. `requests` is used for both HTTP and OAuth client-credentials token requests.

## Public API

### `LLMSettings`

`model` is required. `provider` is required for LiteLLM transport, but may be omitted when `provider_type="custom"`. All other fields have library defaults and are not loaded from external configuration.

| Field | Default | Purpose |
| --- | --- | --- |
| `provider` | Empty | Provider name; required for LiteLLM and used to qualify the LiteLLM model name. Not required for custom HTTP. |
| `model` | Required | Model identifier sent with the completion request. |
| `provider_type` | `"litellm"` | Selects transport. `"custom"` selects direct HTTP; all other values select LiteLLM. |
| `api_key` | `""` | Explicit API key. |
| `client_id` | `""` | OAuth client ID for client-credentials authentication. |
| `client_secret` | `""` | OAuth client secret. |
| `oauth_url` | `""` | OAuth token endpoint. |
| `gateway_url` | `""` | Required for `custom`; optional LiteLLM `api_base`. |
| `verify_ssl` | `True` | TLS certificate verification for direct HTTP and OAuth requests. |
| `timeout_seconds` | `None` | Optional request timeout; omitted from HTTP, OAuth, and LiteLLM calls when unset. |
| `max_tokens` | `None` | Optional default completion-token limit; omitted from requests when unset. |
| `temperature` | `None` | Optional sampling temperature; omitted from requests when unset. |
| `stream` | `True` | Whether calls stream by default. |
| `supports_streaming` | `True` | Capability guard; `False` disables streaming even if requested. |
| `supports_response_schema` | `True` | Use provider JSON-schema formatting, or include schema instructions in the prompt when `False`. |
| `extra_headers` | Empty dictionary | Extra request headers, including caller-managed authorization headers. |

### `LLMClient`

Construct with a caller-created `LLMSettings` object. `check_llm_configuration()` returns `(enabled, reason)` after checking the model, endpoint, and available authentication inputs; it does not contact a provider.

`complete(prompt, ...)` accepts these optional keyword arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `system` | `""` | Optional system message. |
| `response_schema` | `None` | JSON schema for structured output. Its presence enables JSON parsing by default. |
| `response_schema_name` | `"structured_response"` | Name attached to the provider JSON schema. |
| `strict_json_schema` | `True` | Request strict schema output where supported. |
| `json_output` | `None` | Defaults to `True` when a schema is provided; set `False` to return text without JSON parsing. |
| `max_tokens` | `None` | Per-call override; otherwise uses `LLMSettings.max_tokens`, and is omitted if both are unset. |
| `headers` | `None` | Per-call HTTP headers merged over `LLMSettings.extra_headers`. |
| `stream` | `None` | Per-call override; otherwise uses `LLMSettings.stream`, subject to `supports_streaming`. |
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

## Local development and testing - ONLY FOR CODE MAINTAINER

For local development, install the checkout in editable mode with development tools from the repository root:

```powershell
python.exe -m pip install -e ".[dev]"
```

The `dev` extra installs tools such as `pytest` and `build`. For unit tests, the built-in `unittest` runner is sufficient:

```powershell
python -m unittest discover -s tests/unit -v
```

If your organization intercepts PyPI TLS and pip reports a certificate verification error, use your organization's approved package index or CA certificate, for example `python.exe -m pip install --cert "C:\path\to\corporate-ca.pem" -e ".[dev]"`. As a last resort, and only if permitted by your organization's policy, pip can be told to trust the package hosts; this weakens TLS certificate verification for those hosts.

### Rebuild the wheel after making changes

From the repository root, install the project and its build/test tools, run the unit tests, and build a new wheel:

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests/unit -v
$env:PIP_TRUSTED_HOST = "pypi.org files.pythonhosted.org"
python -m build --wheel
```

The `PIP_TRUSTED_HOST` setting applies only to the current PowerShell session. It allows the isolated build environment to install its build dependencies when certificate verification fails, but weakens certificate verification for these hosts. Prefer the approved package index or CA certificate above whenever possible.

The rebuilt wheel is written to `dist/`. Its filename includes the project version from `pyproject.toml`. Before publishing a changed build, increment the version in both `pyproject.toml` and `src/ai_llm_router/__init__.py` so the SDK and wheel metadata stay aligned. Then distribute the new `.whl` file to users.

To test-install the rebuilt wheel into the current environment, use its exact filename (the example shows version `0.1.0`):

```powershell
python -m pip install --force-reinstall ".\dist\ai_llm_router-0.1.0-py3-none-any.whl"
```

If pip or the build tool cannot download build dependencies because of your organization's TLS inspection, configure pip to use the approved package index or CA certificate described under [Local development and testing](#local-development-and-testing). Use `PIP_TRUSTED_HOST` only as a policy-approved last resort, since it weakens TLS certificate verification for the listed hosts.


## Tests

The unit tests mock provider responses; they do not make live network requests. The LiteLLM test uses a mocked module. Follow the local setup above before running the test command.

## Live integration check

Run the integration script from the repository root. It sends a real request and may incur provider charges. The default transport is LiteLLM; use `--provider` for the provider name. To call an HTTP-compatible endpoint directly, set `--provider-type custom` and `--gateway-url` instead. The custom transport expects the OpenAI Chat Completions-compatible format described above.

Example using LiteLLM:

```powershell
python tests/integration/run_real_llm_test.py --provider openai --model gpt-4.1-mini --prompt "Say hello in one sentence."
```

Example using a custom HTTP endpoint:

```powershell
python tests/integration/run_real_llm_test.py --model gpt-oss:20b-cloud --provider-type custom --gateway-url "http://localhost:11434/v1/chat/completions" --prompt "Say hello in one sentence."
```

Pass credentials with `--api-key`, or use `--client-id`, `--client-secret`, and `--oauth-url` for OAuth. Repeated `--header NAME=VALUE` options add request headers. Avoid putting real secrets in commands that may be saved to shell history.

### Available arguments

| Argument | Purpose |
| --- | --- |
| `--prompt` | **Required.** User prompt. |
| `--model` | **Required.** Model identifier. |
| `--provider` | LiteLLM provider name; required with the LiteLLM transport and omitted for custom HTTP. |
| `--provider-type litellm\|custom` | Select transport. Defaults to `litellm`. |
| `--gateway-url` | Endpoint URL; required for custom HTTP. |
| `--api-key` | API key. |
| `--client-id`, `--client-secret`, `--oauth-url` | OAuth client-credentials settings. |
| `--header NAME=VALUE` | Additional request header; may be repeated. |
| `--system` | Optional system message. |
| `--stream` / `--streaming` | Stream output as it arrives; off if omitted. |
| `--response-schema JSON` | JSON Schema for structured output. |
| `--no-response-schema-support` | Send schema instructions in the prompt instead of using provider schema mode. |
| `--max-tokens`, `--temperature` | Generation settings. |
| `--timeout-seconds` | Request timeout. |
| `--verify-ssl` / `--no-verify-ssl` | Enable or disable TLS certificate verification for custom HTTP and OAuth. Verification is enabled by default; disabling it is not recommended and does not affect LiteLLM requests. |

### Understanding the output

Non-streaming calls print `Response:` followed by the generated text. With `--stream`, text appears as it arrives. If `--response-schema` is supplied and the response parses as JSON, the runner also prints `Parsed JSON:`. Successful calls finish with a `Metadata:` JSON block containing the status, model/provider identifiers, timing, response details, and any usage or streaming statistics available from the endpoint; the exact fields vary by provider. Request failures are printed to stderr and return a non-zero exit status.

## Limitations

Provider response formats and JSON-schema capabilities vary. Configure capability flags for the selected provider/model and validate behavior against it. LiteLLM is installed by default with the SDK, although the direct HTTP transport does not use it.
