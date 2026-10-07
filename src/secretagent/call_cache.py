"""Record or replay model-call results without changing cold evaluations."""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


_SEEN = defaultdict(int)


def _request(prompt, model, settings):
    return {"prompt": prompt, "model": model, "settings": settings}


def _path(request):
    root_value = os.environ.get("SECRETAGENT_CALL_CACHE_DIR")
    if not root_value:
        raise ValueError("SECRETAGENT_CALL_CACHE_DIR is required for record/replay")
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    index = _SEEN[digest]
    _SEEN[digest] += 1
    return Path(root_value) / digest[:2] / digest / f"{index:06d}.json"


def replay(prompt, model, settings):
    request = _request(prompt, model, settings)
    path = _path(request)
    entry = json.loads(path.read_text(encoding="utf-8"))
    if entry.get("schema") != 1 or entry.get("request") != request:
        raise ValueError(f"recorded request differs: {path}")
    return entry["output"], entry["stats"]


def record(prompt, model, settings, result):
    request = _request(prompt, model, settings)
    path = _path(request)
    if path.exists():
        raise FileExistsError(f"call record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"schema": 1, "request": request,
             "output": result[0], "stats": result[1]}
    temporary = path.with_suffix(".writing")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(entry, stream, ensure_ascii=False)
    temporary.rename(path)
