"""Make a real LLM request from the command line for manual integration checks.

Example:
    python tests/integration/run_real_llm_test.py --provider openai \
        --model gpt-4.1-mini --prompt "Say hello in one sentence."

The API key is requested securely when omitted, unless OAuth credentials or an
Authorization header are supplied. This script can incur provider charges.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def _parse_header(value: str) -> tuple[str, str]:
    name, separator, header_value = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("headers must use NAME=VALUE format")
    return name.strip(), header_value.strip()


def _has_authorization_header(headers: dict[str, str]) -> bool:
    return any(name.lower() == "authorization" and value.strip() for name, value in headers.items())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a real chat-completion request using the project's LLMClient.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--provider", help="Provider name; required for LiteLLM transport")
    parser.add_argument("--model", required=True, help="Provider model identifier")
    parser.add_argument(
        "--provider-type",
        choices=("litellm", "custom"),
        help="Use LiteLLM or direct custom HTTP",
    )
    parser.add_argument("--gateway-url", help="Gateway URL; required for custom HTTP")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--client-id", help="OAuth client ID")
    parser.add_argument("--client-secret", help="OAuth client secret")
    parser.add_argument("--oauth-url", help="OAuth client-credentials token endpoint")
    parser.add_argument(
        "--header",
        action="append",
        type=_parse_header,
        metavar="NAME=VALUE",
        help="Additional request header; may be repeated",
    )
    parser.add_argument("--timeout-seconds", type=float, help="Request timeout in seconds")
    parser.add_argument(
        "--verify-ssl",
        action=argparse.BooleanOptionalAction,
        help="Verify TLS certificates for direct HTTP and OAuth requests",
    )
    parser.add_argument("--max-tokens", type=int, help="Maximum completion tokens")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument(
        "--stream",
        "--streaming",
        dest="stream",
        action="store_true",
        help="Stream output as it arrives",
    )
    parser.add_argument(
        "--response-schema",
        help="Optional JSON Schema object, passed as a JSON string",
    )
    parser.add_argument(
        "--no-response-schema-support",
        action="store_false",
        dest="supports_response_schema",
        help="Send schema instructions in the prompt instead of using provider schema mode",
    )
    return parser


def main(argv: list[str] | argparse.Namespace | None = None) -> int:
    parser = _build_parser()
    args = argv if isinstance(argv, argparse.Namespace) else parser.parse_args(argv)
    try:
        from ai_llm_router import LLMClient, LLMRequestError, LLMSettings
    except ModuleNotFoundError as exc:
        if exc.name == "requests":
            print("Missing dependency 'requests'; install it before making live calls.", file=sys.stderr)
            return 2
        raise

    schema_argument = getattr(args, "response_schema", None)
    if schema_argument:
        try:
            response_schema: Any = json.loads(schema_argument)
        except json.JSONDecodeError as exc:
            print(f"--response-schema must be valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(response_schema, dict):
            print("--response-schema must decode to a JSON object", file=sys.stderr)
            return 2
    else:
        response_schema = None

    settings_values: dict[str, Any] = {}
    if getattr(args, "provider", None):
        settings_values["provider"] = args.provider
    if getattr(args, "model", None):
        settings_values["model"] = args.model
    if getattr(args, "provider_type", None):
        settings_values["provider_type"] = args.provider_type
    if getattr(args, "api_key", None):
        settings_values["api_key"] = args.api_key
    if getattr(args, "client_id", None):
        settings_values["client_id"] = args.client_id
    if getattr(args, "client_secret", None):
        settings_values["client_secret"] = args.client_secret
    if getattr(args, "oauth_url", None):
        settings_values["oauth_url"] = args.oauth_url
    if getattr(args, "gateway_url", None):
        settings_values["gateway_url"] = args.gateway_url
    if getattr(args, "verify_ssl", None) is not None:
        settings_values["verify_ssl"] = args.verify_ssl
    if getattr(args, "timeout_seconds", None) is not None:
        settings_values["timeout_seconds"] = args.timeout_seconds
    if getattr(args, "max_tokens", None) is not None:
        settings_values["max_tokens"] = args.max_tokens
    if getattr(args, "temperature", None) is not None:
        settings_values["temperature"] = args.temperature
    if getattr(args, "stream", None) is not None:
        settings_values["stream"] = args.stream
    elif getattr(args, "streaming", None) is not None:
        settings_values["stream"] = args.streaming
    if getattr(args, "supports_streaming", None) is not None:
        settings_values["supports_streaming"] = args.supports_streaming
    if getattr(args, "supports_response_schema", None) is not None:
        settings_values["supports_response_schema"] = args.supports_response_schema
    if getattr(args, "extra_headers", None) is not None:
        settings_values["extra_headers"] = args.extra_headers

    settings = LLMSettings(**settings_values)
    client = LLMClient(settings)
    enabled, reason = client.check_llm_configuration()
    if not enabled:
        print(f"Client configuration is incomplete: {reason}", file=sys.stderr)
        return 2

    def print_delta(fragment: str) -> None:
        print(fragment, end="", flush=True)


    streaming = getattr(args, "stream", getattr(args, "streaming", False))

    try:
        response = client.complete(
            args.prompt,
            system=getattr(args, "system", ""),
            response_schema=response_schema,
            stream=streaming,
            on_delta=print_delta if streaming else None,
        )
    except (LLMRequestError, TypeError, ValueError) as exc:
        if streaming:
            print(file=sys.stdout)
        print(f"LLM request failed: {exc}", file=sys.stderr)
        return 1

    if streaming:
        print(file=sys.stdout)
    else:
        print("Response:")
        print(response.text)
    if response.parsed is not None:
        print("\nParsed JSON:")
        print(json.dumps(response.parsed, indent=2, ensure_ascii=False))
    print("\nMetadata:")
    print(json.dumps(response.metadata, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    # args = argparse.Namespace(prompt="Say hello in one sentence.", system="", provider="gemini", model="gemini-2.5-flash", provider_type="litellm", gateway_url="", api_key="xxxxxxxxxxxxxx", client_id="", client_secret="", oauth_url="", header=[], timeout_seconds=120.0, verify_ssl=False, max_tokens=4000, temperature=0.0, stream=False, response_schema=None, supports_response_schema=True)
    # main(args)


# example usage:
# using litellm:
# python tests/integration/run_real_llm_test.py --provider gemini --model gemini-2.5-flash --prompt "Explain what an LLM gateway does in one sentence." --api-key "YOUR_GEMINI_API_KEY"

# using gemini api directly
# python tests/integration/run_real_llm_test.py --model gemini-2.5-flash --provider-type custom --gateway-url "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" --prompt "Say hello in one sentence." --api-key "YOUR_GEMINI_API_KEY"

# using ollama
# python tests/integration/run_real_llm_test.py --model gpt-oss:20b-cloud --provider-type custom --gateway-url "http://localhost:11434/v1/chat/completions" --prompt "Say hello in one sentence." --api-key "not_needed_for_ollama"
