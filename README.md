# ai-llm-router

A package-ready multi-provider LLM client that supports:

- `openai`
- `anthropic`
- `google`
- `optusaifoundation`

It exposes `LLMSettings`, `LLMClient`, and `LLMCompletionResult`.

`LLMClient` is the runtime object you call for requests. Use `complete(...)` for both text and structured JSON workflows.

## Install (editable during development)

```bash
pip install -e .[test]
```

## Usage

```python
from ai_llm_router import LLMClient, LLMCompletionResult, LLMSettings

settings = LLMSettings.from_env()
client = LLMClient(settings)


def handle_delta(chunk: str) -> None:
    print(chunk, end="", flush=True)


result: LLMCompletionResult | None = client.complete(
    prompt="Extract the person details: John Smith is 34 years old and works as a software engineer in Sydney.",
    system="you are a helpful Assistant.",
    response_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "job_title": {"type": "string"},
            "city": {"type": "string"},
        },
        "required": ["name", "age", "job_title", "city"],
        "additionalProperties": False,
    },
    stream=True,
    on_delta=handle_delta,
)

if result is None:
    raise RuntimeError("Request failed")

print("\n\nParsed:", result.parsed)
print("Metadata:", result.metadata)
```

## API Reference

### `LLMClient.complete(...)`

Call:

```python
result = client.complete(
    prompt,
    system="",
    response_schema=None,
    response_format=None,
    response_schema_name=None,
    strict_json_schema=None,
    json_output=None,
    stream=None,
    on_delta=None,
)
```

Parameters:

- `prompt` (str, required): user prompt text.
- `system` (str): system instruction.
- `response_schema` (dict): convenience schema input; converted to `response_format` json_schema shape.
- `response_format` (dict): native provider response format payload.
- `response_schema_name` (str): schema name used when wrapping `response_schema`.
- `strict_json_schema` (bool): strict schema mode override.
- `json_output` (bool): force JSON parsing even without schema.
- `stream` (bool): stream tokens/chunks when supported.
- `on_delta` (Callable[[str], None]): callback invoked for each streamed text chunk.

Returns:

- `LLMCompletionResult | None`
- `None` indicates failure; check `client.last_error`.

For JSON-only use cases, call `complete(..., json_output=True)` and read `result.parsed`.

### `LLMCompletionResult`

Fields:

- `provider`
- `model`
- `text`
- `parsed`
- `metadata`
- `raw_response`
- `used_streaming`

`metadata` contains normalized run details (keys appear only when available from the provider):

Common keys (non-streaming and streaming):

- `response_id`
- `created`
- `response_model`
- `response_object`
- `finish_reason`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `input_tokens`
- `output_tokens`

Detailed token keys (provider-dependent):

- `reasoning_tokens`
- `text_tokens`
- `prompt_text_tokens`
- `prompt_audio_tokens`
- `prompt_image_tokens`
- `prompt_video_tokens`
- `cached_prompt_tokens`
- `prompt_cache_write_tokens`
- `cache_read_input_tokens`
- `tool_use_prompt_tokens`

OAIF keys (when returned by OAIF/gateway):

- `request_id`
- `apigee_message_id`
- `correlation_id`
- `dt_trace_id`
- `oaif_cost_usd`
- `cost_modifier_pct`
- `duration_s`
- `oaif_model`
- `cache_read_tokens`
- `cache_creation_tokens`

Streaming-only aggregation keys:

- `stream_chunk_count`
- `first_chunk_id`
- `first_chunk_created`
- `first_chunk_object`
- `first_chunk_model`
- `last_chunk_id`
- `last_chunk_created`
- `last_chunk_object`
- `last_chunk_model`

Other optional keys:

- `system_fingerprint`

## Environment variables

Start by copying the template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then fill required values based on provider:

- All providers: `LLM_PROVIDER`, `LLM_MODEL`
- `openai`, `anthropic`, `google`: `LLM_API_KEY`
- `optusaifoundation`: `LLM_CLIENT_ID`, `LLM_CLIENT_SECRET`

Additional supported variables (optional unless noted):

- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_API_KEY` (for non-OAIF providers)
- `LLM_CLIENT_ID` (OAIF)
- `LLM_CLIENT_SECRET` (OAIF)
- `LLM_OAUTH_URL` (optional OAIF)
- `LLM_GATEWAY_URL` (optional)
- `LLM_VERIFY_SSL` (`true`/`false`)
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_TIMEOUT_SECONDS`
- `LLM_STREAMING` (`true`/`false`)
- `LLM_STRICT_JSON_SCHEMA` (`true`/`false`)
- `LLM_RESPONSE_SCHEMA_NAME`

Set variables in your shell (examples):

Use the command style for your shell:

- Bash/zsh: `export ...`
- PowerShell: `$env:...`
- Windows cmd: `set ...`

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export LLM_API_KEY=your_api_key_here
```

```powershell
$env:LLM_PROVIDER="openai"
$env:LLM_MODEL="gpt-4o-mini"
$env:LLM_API_KEY="your_api_key_here"
```

```cmd
set LLM_PROVIDER=openai
set LLM_MODEL=gpt-4o-mini
set LLM_API_KEY=your_api_key_here
```

For `optusaifoundation` instead of `LLM_API_KEY`:

```powershell
$env:LLM_PROVIDER="optusaifoundation"
$env:LLM_MODEL="your_oaif_model"
$env:LLM_CLIENT_ID="your_client_id"
$env:LLM_CLIENT_SECRET="your_client_secret"
```

```cmd
set LLM_PROVIDER=optusaifoundation
set LLM_MODEL=your_oaif_model
set LLM_CLIENT_ID=your_client_id
set LLM_CLIENT_SECRET=your_client_secret
```

To persist values for future `cmd` sessions, use `setx` (example):

```cmd
setx LLM_PROVIDER optusaifoundation
```

Note: `setx` updates future terminals only. Open a new terminal to use the new value.

## Run Example Manually

After installing dependencies and setting env variables, run the demo from the project root:

```bash
python examples/streaming_schema_demo.py
```

```powershell
python .\examples\streaming_schema_demo.py
```

```cmd
python examples\streaming_schema_demo.py
```

To test OAIF authentication only and fetch a token using `LLM_CLIENT_ID` / `LLM_CLIENT_SECRET`:

```bash
python examples/oaif_token_demo.py
```

```powershell
python .\examples\oaif_token_demo.py
```

```cmd
python examples\oaif_token_demo.py
```

The token example prints only a masked token prefix, not the full bearer token.

## Run tests

```bash
pytest
```

By default only import/smoke tests run. A live integration test is included and only runs when `LLM_RUN_LIVE_TEST=1`.

## Build wheel

```bash
python -m pip install --upgrade build
python -m build
```

Artifacts are generated in `dist/`.
