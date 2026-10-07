"""The AFlow call archive must preserve live calls and replay without network access."""

import asyncio
import importlib.util
import json
from pathlib import Path

from openai.types.chat import ChatCompletion


SOURCE = (Path(__file__).resolve().parents[1] / "benchmarks" / "COMMON"
          / "aflow-rebuttal" / "upstream" / "call_cache.py")
spec = importlib.util.spec_from_file_location("aflow_call_cache", SOURCE)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def sample_response(content="yes"):
    return ChatCompletion.model_validate({
        "id": "chat-1", "object": "chat.completion", "created": 1,
        "model": "example-model",
        "choices": [{"finish_reason": "stop", "index": 0,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2,
                  "total_tokens": 13},
    })


def test_record_then_replay_two_identical_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("AFLOW_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AFLOW_CACHE_MODE", "record")
    cache._SEEN.clear()
    kwargs = {"model": "example-model", "messages": [{"role": "user", "content": "Q"}],
              "temperature": 1}
    calls = 0

    async def live():
        nonlocal calls
        calls += 1
        return sample_response(str(calls))

    first = asyncio.run(cache.completion(live, "https://example.test/v1", kwargs))
    second = asyncio.run(cache.completion(live, "https://example.test/v1", kwargs))
    assert calls == 2  # Recording never returns a cached answer.
    assert first.choices[0].message.content == "1"
    assert second.choices[0].message.content == "2"
    records = sorted(tmp_path.rglob("*.json"))
    assert len(records) == 2
    assert json.loads(records[0].read_text(encoding="utf-8"))["request"]["messages"] == kwargs["messages"]

    monkeypatch.setenv("AFLOW_CACHE_MODE", "replay")
    cache._SEEN.clear()  # A replay is a fresh process.

    async def no_network():
        raise AssertionError("replay contacted provider")

    replayed = [asyncio.run(cache.completion(no_network, "https://example.test/v1", kwargs))
                for _ in range(2)]
    assert [r.choices[0].message.content for r in replayed] == ["1", "2"]
    assert [r.usage.prompt_tokens for r in replayed] == [11, 11]


def test_missing_replay_entry_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("AFLOW_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AFLOW_CACHE_MODE", "replay")
    cache._SEEN.clear()

    async def no_network():
        raise AssertionError("replay contacted provider")

    try:
        asyncio.run(cache.completion(no_network, "https://example.test/v1",
                                     {"model": "missing", "messages": []}))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing replay entry must fail")
