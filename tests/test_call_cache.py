"""Cold call recording and strict warm replay for MODO evaluations."""

from secretagent import call_cache


def test_record_and_replay_repeated_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRETAGENT_CALL_CACHE_DIR", str(tmp_path))
    call_cache._SEEN.clear()
    settings = {"temperature": 1, "stream": False}
    call_cache.record("question", "model", settings,
                      ("first", {"input_tokens": 2, "output_tokens": 1, "cost": 0.1}))
    call_cache.record("question", "model", settings,
                      ("second", {"input_tokens": 2, "output_tokens": 1, "cost": 0.1}))
    call_cache._SEEN.clear()
    assert call_cache.replay("question", "model", settings)[0] == "first"
    assert call_cache.replay("question", "model", settings)[0] == "second"


def test_replay_missing_request_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRETAGENT_CALL_CACHE_DIR", str(tmp_path))
    call_cache._SEEN.clear()
    try:
        call_cache.replay("missing", "model", {})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("replay must fail instead of calling provider")
