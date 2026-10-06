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
        description="Send a real chat-completion request using the project's LLMClient."
    )
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--system", default="", help="Optional system message")
    parser.add_argument("--provider", default="openai", help="Provider name (default: openai)")
    parser.add_argument("--model", required=True, help="Provider model identifier")
    parser.add_argument(
        "--provider-type",
        choices=("litellm", "custom"),
        default="litellm",
        help="Use LiteLLM or direct custom HTTP (default: litellm)",
    )
    parser.add_argument("--gateway-url", default="", help="Gateway URL; required for custom HTTP")
    parser.add_argument("--api-key", default="", help="API key (securely prompted if no credentials are supplied)")
    parser.add_argument("--client-id", default="", help="OAuth client ID")
    parser.add_argument("--client-secret", default="", help="OAuth client secret")
    parser.add_argument("--oauth-url", default="", help="OAuth client-credentials token endpoint")
    parser.add_argument(
        "--header",
        action="append",
        type=_parse_header,
        default=[],
        metavar="NAME=VALUE",
        help="Additional request header; may be repeated",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Request timeout (default: 120)")
    parser.add_argument(
        "--verify-ssl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify TLS certificates for direct HTTP and OAuth requests (default: enabled; use --no-verify-ssl to disable)",
    )
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum completion tokens (default: 4000)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0)")
    parser.add_argument("--stream", action="store_true", help="Stream output as it arrives")
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
    parser.set_defaults(supports_response_schema=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from ai_llm_router.llm_client import LLMClient, LLMRequestError, LLMSettings
    except ModuleNotFoundError as exc:
        if exc.name == "requests":
            print("Missing dependency 'requests'; install it before making live calls.", file=sys.stderr)
            return 2
        raise

    extra_headers = dict(args.header)
    has_oauth = bool(args.client_id.strip() and args.client_secret.strip() and args.oauth_url.strip())
    has_custom_authorization = args.provider_type == "custom" and _has_authorization_header(extra_headers)
    if not args.api_key and not has_oauth and not has_custom_authorization:
        try:
            args.api_key = getpass.getpass("LLM API key (input hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("No API key supplied.", file=sys.stderr)
            return 2

    if args.response_schema:
        try:
            response_schema: Any = json.loads(args.response_schema)
        except json.JSONDecodeError as exc:
            parser.error(f"--response-schema must be valid JSON: {exc}")
        if not isinstance(response_schema, dict):
            parser.error("--response-schema must decode to a JSON object")
    else:
        response_schema = None

    settings = LLMSettings(
        provider=args.provider,
        model=args.model,
        provider_type=args.provider_type,
        api_key=args.api_key,
        client_id=args.client_id,
        client_secret=args.client_secret,
        oauth_url=args.oauth_url,
        gateway_url=args.gateway_url,
        verify_ssl=args.verify_ssl,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        streaming=args.stream,
        supports_response_schema=args.supports_response_schema,
        extra_headers=extra_headers,
    )
    client = LLMClient(settings)
    enabled, reason = client.enabled_with_reason()
    if not enabled:
        print(f"Client configuration is incomplete: {reason}", file=sys.stderr)
        return 2

    def print_delta(fragment: str) -> None:
        print(fragment, end="", flush=True)

    try:
        response = client.complete(
            args.prompt,
            system=args.system,
            response_schema=response_schema,
            stream=args.stream,
            on_delta=print_delta if args.stream else None,
        )
    except (LLMRequestError, TypeError, ValueError) as exc:
        if args.stream:
            print(file=sys.stdout)
        print(f"LLM request failed: {exc}", file=sys.stderr)
        return 1

    if args.stream:
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

# example usage:
# python tests/integration/run_real_llm_test.py --provider openai --model gpt-4.1-mini --prompt "Say hello in one sentence." --stream
# python tests/integration/run_real_llm_test.py --provider gemini --model gemini-2.5-flash --prompt "Explain what an LLM gateway does in one sentence." --api-key "YOUR_GEMINI_API_KEY"