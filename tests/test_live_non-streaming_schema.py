import os
import json

import pytest

from ai_llm_router import LLMClient, LLMSettings


def run_live_non_streaming_schema(*, print_output: bool = False) -> None:
    settings = LLMSettings.from_env()
    client = LLMClient(settings)
    import uuid
    correlation_id = str(uuid.uuid4())
    deltas: list[str] = []

    def handle_delta(chunk: str) -> None:
        deltas.append(chunk)

    result = client.complete(
        prompt="Extract the person details: John Smith is 34 years old and works as a software engineer in Sydney.",
        system="you are a helpful Assistant.",
        header={"x-correlation-id": correlation_id},
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
        stream=False,
        on_delta=handle_delta,
    )

    if result is None:
        raise AssertionError(f"LLM call failed: {client.last_error}")

    if print_output:
        print("LLM call succeeded")
        print("provider:", result.provider)
        print("model:", result.model)
        print("used_streaming:", result.used_streaming)
        print("\ntext:\n", result.text)
        print("\nparsed:\n", json.dumps(result.parsed, indent=2, ensure_ascii=True, default=str))
        print("\nmetadata:\n", json.dumps(result.metadata, indent=2, ensure_ascii=True, default=str))
        print("\nraw_response:\n", json.dumps(result.raw_response, indent=2, ensure_ascii=True, default=str))
        print("\nstreamed_deltas_combined:\n", "".join(deltas))

    assert result.parsed is not None
    assert result.parsed["name"].lower() == "john smith"
    assert result.parsed["age"] == 34
    assert "software" in result.parsed["job_title"].lower()
    assert result.parsed["city"].lower() == "sydney"
    assert result.used_streaming is False
    assert len("".join(deltas)) == 0
    assert isinstance(result.metadata, dict)
    assert result.metadata.get("response_id")
    assert result.metadata.get("response_model")
    assert isinstance(result.raw_response, dict)
    assert result.raw_response.get("choices")


@pytest.mark.integration
def test_live_non_streaming_schema():
    if os.getenv("LLM_RUN_LIVE_TEST", "0") != "1":
        pytest.skip("Set LLM_RUN_LIVE_TEST=1 to run live provider integration test")
    run_live_non_streaming_schema()


if __name__ == "__main__":
    run_live_non_streaming_schema(print_output=True)
    print("Live non-streaming schema run passed.")
