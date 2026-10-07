"""Per-process record/replay of OpenAI-compatible chat completions.

Recording never serves a cached response. It therefore leaves a cold run's
predictions and measured token usage unchanged. Replay is deliberately strict:
a missing or mismatched call raises rather than silently contacting a provider.
The archive contains prompts and model outputs; keep it private until the
benchmark's data-sharing terms have been checked.
"""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from openai.types.chat import ChatCompletion


_SEEN = defaultdict(int)


def _request_key(base_url, kwargs):
    request = {"base_url": str(base_url), **kwargs}
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), request


async def completion(create, base_url, kwargs):
    """Make a live call, record it, or replay the corresponding recorded call."""
    mode = os.environ.get("AFLOW_CACHE_MODE", "off").lower()
    if mode == "off":
        return await create()
    if mode not in {"record", "replay"}:
        raise ValueError(f"unknown AFLOW_CACHE_MODE: {mode}")
    root_value = os.environ.get("AFLOW_CACHE_DIR")
    if not root_value:
        raise ValueError("AFLOW_CACHE_DIR is required for record/replay")
    root = Path(root_value)
    digest, request = _request_key(base_url, kwargs)
    index = _SEEN[digest]
    _SEEN[digest] += 1
    path = root / digest[:2] / digest / f"{index:06d}.json"
    if mode == "replay":
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("request") != request or entry.get("schema") != 1:
            raise ValueError(f"recorded request differs: {path}")
        return ChatCompletion.model_validate(entry["response"])

    if path.exists():
        raise FileExistsError(f"call record already exists: {path}")
    response = await create()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"schema": 1, "request": request,
             "response": response.model_dump(mode="json")}
    temporary = path.with_suffix(".writing")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(entry, stream, ensure_ascii=False)
    temporary.rename(path)
    return response
