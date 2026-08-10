from pathlib import Path
import os

from ai_llm_router import LLMClient, LLMCompletionResult, LLMSettings


def load_env_file_if_available() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file_if_available()

settings = LLMSettings.from_env()
client = LLMClient(settings)


def handle_delta(chunk: str) -> None:
    print(f'chunk: {chunk}', end="", flush=True)


PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "job_title": {"type": "string"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "job_title", "city"],
    "additionalProperties": False,
}


def print_result(result: LLMCompletionResult) -> None:
    print("\n\nText:", result.text)
    print("Parsed:", result.parsed)
    print("Metadata:", result.metadata)


def run_simple_non_streaming() -> None:
    print("\n=== 1) Simple Completion - Non-Streaming ===")
    result = client.complete(
        prompt="Write one short sentence about Sydney.",
        system="You are a helpful assistant.",
        stream=False,
    )
    if result is None:
        raise RuntimeError(f"LLM request failed: {client.last_error}")
    print_result(result)


def run_simple_streaming() -> None:
    print("\n=== 2) Simple Completion - Streaming ===")
    result = client.complete(
        prompt="Write one short sentence about Sydney.",
        system="You are a helpful assistant.",
        stream=True,
        on_delta=handle_delta,
    )
    if result is None:
        raise RuntimeError(f"LLM request failed: {client.last_error}")
    print_result(result)


def run_json_non_streaming() -> None:
    print("\n=== 3) JSON Completion - Non-Streaming ===")
    result = client.complete(
        prompt="Extract the person details: John Smith is 34 years old and works as a software engineer in Sydney.",
        system="You are a helpful assistant.",
        response_schema=PERSON_SCHEMA,
        stream=False,
    )
    if result is None:
        raise RuntimeError(f"LLM request failed: {client.last_error}")
    print_result(result)


def run_json_streaming() -> None:
    print("\n=== 4) JSON Completion - Streaming ===")
    result = client.complete(
        prompt="Extract the person details: John Smith is 34 years old and works as a software engineer in Sydney.",
        system="You are a helpful assistant.",
        response_schema=PERSON_SCHEMA,
        stream=True,
        on_delta=handle_delta,
    )
    if result is None:
        raise RuntimeError(f"LLM request failed: {client.last_error}")
    print_result(result)


if __name__ == "__main__":
    run_simple_non_streaming()
    run_simple_streaming()
    run_json_non_streaming()
    run_json_streaming()
