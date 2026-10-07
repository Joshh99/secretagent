"""One-off corrected Murder run. The optimizer, prompts and scorer are unchanged.

Run with uv from the repository root. Four model routes use OpenRouter;
Gemini keeps its original route. Recovery uses a private copy of successful
low-level calls from the same routes. Old caches stay intact.
"""
import ast
import asyncio
import copy
import contextlib
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import math
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(os.environ.get("NSGA_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
BENCH = ROOT / "benchmarks/musr"
HERE = ROOT / "analysis/nsga-val-selection"
MODEL_MAP = {
    "together_ai/deepseek-ai/DeepSeek-V3.1": "openrouter/deepseek/deepseek-chat-v3.1",
    "together_ai/deepseek-ai/DeepSeek-V3": "openrouter/deepseek/deepseek-chat",
    "together_ai/openai/gpt-oss-20b": "openrouter/openai/gpt-oss-20b",
    "together_ai/openai/gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
}
HELDOUT = Path("C:/Users/STUDENT/aamas-aflow-plan/murder_heldout_ids.json")


def encode_invalid_history_arguments(kw):
    """Carry invalid prior arguments as a JSON string, without repairing them.

    The agent receives the original malformed model output and issues its
    normal validation feedback. Some hosts JSON-parse prior tool arguments
    before accepting a continuation. For those invalid history fields only,
    encode the exact original bytes as a JSON string. No tool executes with
    repaired arguments, and valid histories are unchanged.
    """
    changes = []
    messages = kw.get("messages", [])
    for mi, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for ti, tool in enumerate(message.get("tool_calls") or []):
            args = tool.get("function", {}).get("arguments")
            if not isinstance(args, str):
                continue
            try:
                json.loads(args)
            except json.JSONDecodeError:
                changes.append((mi, ti, args))
    if not changes:
        return kw, []
    routed = dict(kw)
    routed["messages"] = copy.deepcopy(messages)
    evidence = []
    for mi, ti, args in changes:
        routed["messages"][mi]["tool_calls"][ti]["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
        evidence.append({"message_index": mi, "tool_index": ti,
                         "original_arguments_sha256": hashlib.sha256(args.encode()).hexdigest(),
                         "encoding": "exact malformed arguments carried as JSON string; not repaired"})
    return routed, evidence


def is_transient_transport_error(error):
    """Recognize transport failures, including LiteLLM-wrapped DNS errors."""
    status = getattr(error, "status_code", None)
    if status in {400, 401, 403, 404, 422}:
        return False
    name = type(error).__name__
    message = str(error).lower()
    return (isinstance(error, TimeoutError)
            or (isinstance(status, int) and 500 <= status < 600)
            or (name == "APIError" and "502 bad gateway" in message)
            or name in {"APIConnectionError", "RateLimitError", "ServiceUnavailableError", "InternalServerError", "Timeout"}
            or "getaddrinfo failed" in message
            or "server disconnected without sending a response" in message
            or "peer closed connection" in message
            or "incomplete chunked read" in message
            or "unable to get json response" in message
            or "temporarily rate-limited" in message)


def transport_retry(fn, on_error, wait=time.sleep):
    for attempt in range(8):
        try:
            return fn()
        except Exception as exc:
            retry = attempt < 7 and is_transient_transport_error(exc)
            on_error(exc, attempt + 1, not retry)
            if not retry:
                raise
            wait(15)


async def async_transport_retry(fn, on_error, wait=asyncio.sleep):
    for attempt in range(8):
        try:
            return await fn()
        except Exception as exc:
            retry = attempt < 7 and is_transient_transport_error(exc)
            on_error(exc, attempt + 1, not retry)
            if not retry:
                raise
            await wait(15)


def copy_successful_llm_cache(source, destination):
    """Only reuse completed low-level responses, never caught agent failures."""
    path = source / ".secretagent.llm_util._llm_impl"
    with path.open("rb") as fp:
        original = pickle.load(fp)
    successful = {}
    for key, entry in original.items():
        if getattr(entry, "_processing", False):
            continue
        value = entry.get("value") if isinstance(entry, dict) else getattr(entry, "value", None)
        if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[1], dict):
            continue
        stats = value[1]
        if all(k in stats and math.isfinite(float(stats[k])) and float(stats[k]) >= 0
               for k in ["input_tokens", "output_tokens", "cost"]):
            successful[key] = entry
    with (destination / path.name).open("wb") as fp:
        pickle.dump(successful, fp)
    return {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_entries": len(original), "copied_entries": len(successful),
            "excluded": "None, unfinished, unknown-cost responses and all agent-level cache entries"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def balance():
    headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
    out = {"checked_utc": utc()}
    for name, endpoint in [("key", "key"), ("account", "credits")]:
        req = urllib.request.Request("https://openrouter.ai/api/v1/" + endpoint, headers=headers)
        out[name] = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    out["account_balance"] = out["account"]["total_credits"] - out["account"]["total_usage"]
    return out


def worker(args, expected_cases=50):
    """Transport logging and explicit held-out filtering around the existing CLI."""
    import litellm
    run_dir = Path(os.environ["NSGA_RERUN_DIR"])
    routes = json.loads((run_dir / "routes.json").read_text())
    tag = next(x.split("=", 1)[1] for x in args if x.startswith("evaluate.expt_name="))
    log_dir = run_dir / "call_logs"
    lock = threading.Lock()
    seen = set()
    unrecovered_transport_errors = []
    original_sync, original_async = litellm.completion, litellm.acompletion
    rates = {}
    for model, route in routes.items():
        pricing = route["pricing"]
        rates[model] = {
            "input_cost_per_token": float(pricing["prompt"]),
            "output_cost_per_token": float(pricing["completion"]),
            "litellm_provider": "openrouter", "mode": "chat",
        }
        if "input_cache_read" in pricing:
            rates[model]["cache_read_input_token_cost"] = float(pricing["input_cache_read"])
    litellm.register_model(rates)

    def prepare(kw):
        model = kw.get("model")
        if model in routes:
            body = dict(kw.get("extra_body") or {})
            body["provider"] = routes[model]["provider"]
            body["usage"] = {"include": True}
            kw["extra_body"] = body
        return kw

    def record_call(kw, start, response=None, error=None, history_encoding=None, attempt=None, terminal=None):
        response_id = getattr(response, "id", None)
        with lock:
            if response_id and response_id in seen:
                return
            if response_id:
                seen.add(response_id)
            public = {k: v for k, v in kw.items() if k in {
                "model", "messages", "tools", "tool_choice", "temperature",
                "max_tokens", "reasoning_effort", "timeout", "stream", "extra_body",
            }}
            row = {"utc": utc(), "experiment": tag, "request": public,
                   "elapsed_s": time.time() - start, "response_id": response_id}
            if attempt is not None:
                row.update(transport_attempt=attempt, terminal_error=terminal)
            if history_encoding:
                row["invalid_history_encoding"] = history_encoding
            if response is not None:
                row["response"] = response.model_dump(mode="json")
                try:
                    row["estimated_cost"] = litellm.completion_cost(completion_response=response)
                except Exception as exc:
                    row["cost_error"] = type(exc).__name__
                hidden = getattr(response, "_hidden_params", {}) or {}
                row["response_cost"] = hidden.get("response_cost")
            if error is not None:
                message = str(error)
                for name in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
                    value = os.environ.get(name)
                    if value:
                        message = message.replace(value, "[REDACTED]")
                row["error"] = message
                row["error_type"] = type(error).__name__
            with (log_dir / (tag + ".jsonl")).open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, default=str) + "\n")

    def completion(*a, **kw):
        kw = prepare(kw)
        kw, history_encoding = encode_invalid_history_arguments(kw)
        start = time.time()
        def failed(exc, attempt, terminal):
            record_call(kw, start, error=exc, history_encoding=history_encoding, attempt=attempt, terminal=terminal)
            if terminal and is_transient_transport_error(exc):
                unrecovered_transport_errors.append(type(exc).__name__)
                save(run_dir / "worker_logs" / (tag + ".transport_failure.json"),
                     {"utc": utc(), "error_type": type(exc).__name__, "transport_attempt": attempt,
                      "reason": "Transport retries exhausted. No score is accepted; search must stop."})
                raise SystemExit(75)
        result = transport_retry(lambda: original_sync(*a, **kw), failed)
        # LiteLLM's async path may call its synchronous entry point to
        # create a coroutine. The outer async wrapper records its result.
        if not inspect.isawaitable(result):
            record_call(kw, start, response=result, history_encoding=history_encoding)
        return result

    async def acompletion(*a, **kw):
        kw = prepare(kw)
        kw, history_encoding = encode_invalid_history_arguments(kw)
        start = time.time()
        def failed(exc, attempt, terminal):
            record_call(kw, start, error=exc, history_encoding=history_encoding, attempt=attempt, terminal=terminal)
            if terminal and is_transient_transport_error(exc):
                unrecovered_transport_errors.append(type(exc).__name__)
                save(run_dir / "worker_logs" / (tag + ".transport_failure.json"),
                     {"utc": utc(), "error_type": type(exc).__name__, "transport_attempt": attempt,
                      "reason": "Transport retries exhausted. No score is accepted; search must stop."})
                raise SystemExit(75)
        result = await async_transport_retry(lambda: original_async(*a, **kw), failed)
        record_call(kw, start, response=result, history_encoding=history_encoding)
        return result

    litellm.completion, litellm.acompletion = completion, acompletion
    sys.path.insert(0, str(BENCH))
    spec = importlib.util.spec_from_file_location("murder_rerun_expt", BENCH / "expt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if os.environ.get("NSGA_HELDOUT_TEST") == "1":
        original_load = module.load_dataset
        ids = [s.split("/", 1)[1] for s in json.loads(HELDOUT.read_text())["held_out_case_names"]]
        assert len(ids) == len(set(ids)) == 50

        def heldout_load(split):
            assert split == "murder_mysteries_test"
            data = original_load(split)
            by_name = {case.name: case for case in data.cases}
            data.cases = [by_name[name] for name in ids]
            return data
        module.load_dataset = heldout_load
    sys.argv = [str(BENCH / "expt.py")] + args
    with (run_dir / "worker_logs" / (tag + ".log")).open("w", encoding="utf-8") as fp:
        class Tee:
            def __init__(self, stream): self.stream = stream
            def write(self, s): self.stream.write(s); fp.write(s); fp.flush()
            def flush(self): self.stream.flush(); fp.flush()
            def isatty(self): return False
        with contextlib.redirect_stdout(Tee(sys.stdout)), contextlib.redirect_stderr(Tee(sys.stderr)):
            module.app(standalone_mode=False)
    # Keep incomplete or unknown-cost evaluations out of the optimizer.
    # The benchmark scorer and each saved row remain unchanged.
    result_root = Path(next(x.split("=", 1)[1] for x in args if x.startswith("evaluate.result_dir=")))
    outputs = list(result_root.glob("*." + tag + "/results.csv"))
    import pandas as pd
    valid = False
    if len(outputs) == 1:
        data = pd.read_csv(outputs[0])
        valid = (len(data) == data.case_name.nunique() == expected_cases and "cost" in data
                 and data.correct.notna().all() and data.cost.notna().all()
                 and all(math.isfinite(float(c)) and float(c) >= 0 for c in data.cost))
    save(run_dir / "worker_logs" / (tag + ".audit.json"), {
        "csv": str(outputs[0]) if len(outputs) == 1 else None,
        "rows": len(data) if len(outputs) == 1 else 0,
        "complete_score_and_cost": bool(valid),
        "unrecovered_transport_errors": unrecovered_transport_errors,
    })
    if unrecovered_transport_errors:
        raise SystemExit(75)
    if not valid:
        raise SystemExit("Incomplete evaluation or unknown per-case cost; configuration is invalid")


def run():
    import yaml
    import pandas as pd
    from secretagent.optimize.encoder import modular_space_from_yaml, decode_dict, decode_modular
    from secretagent.optimize.pareto import run_nsga2

    tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_valsplit_openrouter_r7"
    run_dir = HERE / tag
    run_dir.mkdir()
    for sub in ["call_logs", "worker_logs", "rollouts"]:
        (run_dir / sub).mkdir()
    os.environ["NSGA_RERUN_DIR"] = str(run_dir)
    os.environ["NSGA_REPO_ROOT"] = str(ROOT)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if (HERE / "active_run.json").exists():
        shutil.copyfile(HERE / "active_run.json", run_dir / "previous_active_run.json")
    save(HERE / "active_run.json", {"run_dir": str(run_dir), "pid": os.getpid(), "started_utc": utc(),
                                   "stdout_log": str(HERE / "rerun_stdout.r7.log")})
    before = balance()
    save(run_dir / "balance_before.json", before)
    original_path = BENCH / "nsga2_murder.yaml"
    original = yaml.safe_load(original_path.read_text())
    routed = dict(original)
    routed["models"] = [MODEL_MAP.get(m, m) for m in original["models"]]
    space_file = run_dir / "nsga2_murder.openrouter.yaml"
    space_file.write_text(yaml.safe_dump(routed, sort_keys=False), encoding="utf-8")
    dims, compound, metadata = modular_space_from_yaml(str(space_file))
    original_dims, original_compound, _ = modular_space_from_yaml(str(original_path))
    assert compound == original_compound
    assert [(d.key, d.size) for d in dims] == [(d.key, d.size) for d in original_dims]
    routes = json.loads((HERE / "validated_retry_routes.json").read_text())
    assert set(routes) == set(MODEL_MAP.values())
    for model, route in routes.items():
        assert route["provider"]["allow_fallbacks"] is False
        raw_checks = json.loads(Path(route["compatibility_record"]).read_text())["checks"]
        matching = [c for c in raw_checks if "openrouter/" + c["model"] == model]
        assert len(matching) == 2 and all(c["passed"] for c in matching)
        adapter = json.loads(Path(route["adapter_compatibility_record"]).read_text())
        assert any(c["model"] == model and c["kind"] == "installed_agent_tool_roundtrip" and c["passed"]
                   for c in adapter["checks"])
    save(run_dir / "routes.json", routes)
    fix_dir = Path(json.loads((HERE / "latest_real_history_fix.json").read_text())["directory"])
    fix_checks = json.loads((fix_dir / "checks.json").read_text())
    assert fix_checks["all_passed"]
    shutil.copyfile(fix_dir / "checks.json", run_dir / "real_history_fix_checks.json")
    case_dir = Path(json.loads((HERE / "latest_real_case_fix.json").read_text())["directory"])
    case_checks = json.loads((case_dir / "checks.json").read_text())
    assert case_checks["transport_passed"]
    shutil.copyfile(case_dir / "checks.json", run_dir / "real_case_fix_checks.json")
    retry_dir = Path(json.loads((HERE / "latest_transport_retry_fix.json").read_text())["directory"])
    retry_checks = json.loads((retry_dir / "checks.json").read_text())
    assert retry_checks["all_passed"]
    shutil.copyfile(retry_dir / "checks.json", run_dir / "transport_retry_fix_checks.json")
    (run_dir / "cache").mkdir()
    recovery_source = os.environ.get("NSGA_RECOVERY_CACHE_SOURCE")
    if recovery_source:
        source = Path(recovery_source).resolve()
        assert source.parent == HERE.resolve() and source.name.endswith("_valsplit_openrouter_r6")
        assert json.loads((source / "routes.json").read_text()) == routes
        assert json.loads((source / "run_record.json").read_text())["status"] == "STOPPED"
        save(run_dir / "cache_recovery.json", copy_successful_llm_cache(source / "cache", run_dir / "cache"))
    frozen_worker = run_dir / "rerun_murder_openrouter.launch.py"
    shutil.copyfile(Path(__file__).resolve(), frozen_worker)
    worker_python = sys.executable
    if os.name == "nt":
        # Preserve this virtual environment while avoiding Windows' extra
        # launcher child, which otherwise can survive subprocess timeouts.
        os.environ["__PYVENV_LAUNCHER__"] = sys.executable
        worker_python = sys._base_executable
    cmd = [worker_python, "-u", str(frozen_worker), "worker", "run", "--config-file", "conf/murder.yaml"]
    overrides = ["dataset.split=murder_mysteries_val", "dataset.n=50",
                 "evaluate.result_dir=" + (run_dir / "rollouts").as_posix(),
                 "cachier.cache_dir=" + (run_dir / "cache").as_posix(),
                 "evaluate.record_details=true"]
    audit = json.loads((HERE / "murder_dataset_audit.json").read_text())
    assert audit["actual_input_overlap_all_validation"] == 0
    protected = [original_path, BENCH / "expt.py", ROOT / "src/secretagent/evaluate.py",
                 BENCH / "data/murder_mysteries_val.json", BENCH / "data/murder_mysteries_test.json",
                 ROOT / "benchmarks/COMMON/optimize-results/musr_murder/nsga2_summary.csv"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    record = {"started_utc": utc(), "branch": "nsga-val-selection", "status": "RUNNING_SEARCH",
              "pop_size": 12, "n_gen": 5, "seed": 42, "timeout_s": 1200,
              "model_map": MODEL_MAP, "original_space_sha256": hashes[str(original_path)],
              "command": cmd, "overrides": overrides, "protected_hashes": hashes,
              "spend_cap": None, "cap_removed_by_user": True,
              "selection_rule": "valid final frontier, max validation correct, min cost, file order",
              "provider_change": "OpenRouter routes fixed per model: V3.1 CoreWeave, V3 StreamLake, OSS20B CoreWeave, OSS120B Together; Gemini unchanged",
              "routes": routes, "cache_policy": "private copy of finite-cost completed low-level r6 calls; no agent cache; source preserved" if recovery_source else "fresh per attempt; all old caches preserved",
              "recovery_cache_source": recovery_source,
              "network_retry_amendment": "Only transient DNS, connection, timeout, empty-JSON, 429 and 5xx failures retry up to eight transport attempts with 15-second waits. Original 1200-second configuration limit still applies. Auth, schema and model validation errors are not retried by this transport fix.",
              "transport_amendment": "Malformed prior tool arguments are encoded losslessly as JSON strings in outgoing history only. Original responses and validation feedback remain intact. Tool inputs and answers are not repaired.",
              "worker_cleanup_fix": "Restore stdout/stderr before closing worker logs; launch the same virtual environment directly so the original subprocess timeout kills the actual worker.",
              "transport_fix_checks": [str(fix_dir / "checks.json"), str(case_dir / "checks.json")],
              "transport_retry_fix_checks": str(retry_dir / "checks.json"),
              "provider_recovery_checks": retry_checks.get("provider_recovery_checks", []),
              "recovery_note": "R6 stopped after eight HTTP 503 responses from Together via OpenRouter. Exact failed validation request and installed agent tool round trip succeeded before this same-route restart. No provider change and no saved optimizer population checkpoint.",
              "frozen_worker_sha256": hashlib.sha256(frozen_worker.read_bytes()).hexdigest(),
              "validation_ids": audit["selected_validation_ids"], "heldout_ids": audit["heldout_ids"]}
    save(run_dir / "run_record.json", record)
    print("Run directory:", run_dir, flush=True)
    print("Credit before:", before["account_balance"], flush=True)
    def label(ds, vec):
        d = decode_dict(ds, vec)
        index = vec[1]
        return d["toplevel_method"] + "/" + original["models"][index].split("/")[-1]

    original_subprocess_run = subprocess.run
    def checked_subprocess_run(*args, **kwargs):
        result = original_subprocess_run(*args, **kwargs)
        if result.returncode == 75:
            # EvalCache catches Exception and assigns invalid fitness. Use
            # SystemExit to preserve the attempt and stop the whole search.
            raise SystemExit("Unrecovered transport failure. Search stopped before network damage can influence later generations.")
        return result
    subprocess.run = checked_subprocess_run

    try:
        front, evaluated, generations = run_nsga2(
            dims=dims, fixed_overrides=[], base_command=cmd, base_dotlist=overrides,
            cwd=str(BENCH), timeout=1200, metric="correct", expt_prefix="nsga_valsplit_or",
            pop_size=12, n_gen=5, seed=42, label_fn=label, compound_overrides=compound)
        frontier_keys = {tuple(chrom) for chrom, _, _ in front}
        rows = []
        for i, (chrom, acc, cost) in enumerate(evaluated):
            method, model = label(dims, chrom).split("/", 1)
            rows.append({"config": method + "/" + model, "method": method, "model": model,
                         "correct": acc, "cost": cost, "frontier": tuple(chrom) in frontier_keys,
                         "valid": math.isfinite(cost), "chromosome": json.dumps(chrom), "eval_index": i + 1})
        summary = pd.DataFrame(rows)
        summary.to_csv(run_dir / "nsga2_summary.validation.csv", index=False)
        pd.DataFrame(generations).to_csv(run_dir / "nsga2_generations.csv", index=False)
        eligible = summary[summary.frontier & summary.valid].sort_values(
            ["correct", "cost", "eval_index"], ascending=[False, True, True])
        if eligible.empty:
            raise RuntimeError("No valid validation frontier configuration")
        chosen = eligible.iloc[0].to_dict()
        selected_files = list((run_dir / "rollouts").glob(
            "*.nsga_valsplit_or_" + f"{chosen['eval_index']:03d}" + "/results.csv"))
        assert len(selected_files) == 1
        selected_results = pd.read_csv(selected_files[0])
        assert len(selected_results) == selected_results.case_name.nunique() == 50
        assert set(selected_results.case_name) == set(audit["selected_validation_ids"])
        assert math.isclose(float(selected_results.correct.mean()), chosen["correct"])
        chosen["validation_csv"] = str(selected_files[0])
        chosen["selected_utc"] = utc()
        chosen["overrides"] = decode_modular(dims, json.loads(chosen["chromosome"]), compound)
        save(run_dir / "selection_frozen_before_test.json", chosen)
        record.update(status="RUNNING_HELDOUT_TEST", selected=chosen, search_finished_utc=utc())
        save(run_dir / "run_record.json", record)
        print("Validation selection frozen:", chosen, flush=True)
        test_tag = "valsplit_heldout50.test_pass_" + chosen["method"] + "_" + chosen["model"]
        test_overrides = chosen["overrides"] + [
            "dataset.split=murder_mysteries_test", "dataset.n=50",
            "evaluate.result_dir=" + (ROOT / "benchmarks/COMMON/results/musr/murder").as_posix(),
            "evaluate.expt_name=" + test_tag,
            "cachier.cache_dir=" + (run_dir / "cache").as_posix(), "evaluate.record_details=true"]
        test_env = dict(os.environ, NSGA_HELDOUT_TEST="1")
        result = subprocess.run(cmd + test_overrides, cwd=BENCH, env=test_env,
                                capture_output=True, text=True, timeout=1200)
        (run_dir / "test_stdout.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Heldout test failed; see test_stdout.log")
        csv_path = next(Path(line.split("saved in ", 1)[1].strip()) for line in result.stdout.splitlines()
                        if "saved in " in line and ".csv" in line)
        test = pd.read_csv(csv_path)
        assert len(test) == 50 and set(test.case_name) == set(audit["heldout_ids"])
        assert test.case_name.nunique() == 50
        assert test.correct.notna().all() and test.correct.isin([0, 1, False, True]).all()
        assert test.cost.notna().all()
        record["test"] = {"csv_path": str(csv_path), "n": 50,
                          "correct_count": int(test.correct.sum()), "accuracy": float(test.correct.mean()),
                          "cost_per_100": float(test.cost.mean() * 100)}
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
        canonical = ROOT / "benchmarks/COMMON/optimize-results/musr_murder/nsga2_summary.validation.csv"
        if canonical.exists():
            raise RuntimeError("Corrected summary already exists; refusing to overwrite")
        shutil.copyfile(run_dir / "nsga2_summary.validation.csv", canonical)
        record.update(status="DONE", finished_utc=utc())
        print("DONE:", record["test"], flush=True)
    except BaseException as exc:
        record.update(status="STOPPED", stopped_utc=utc(), error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        try:
            after = balance()
            save(run_dir / "balance_after.json", after)
            record["openrouter_account_usage_delta"] = after["account"]["total_usage"] - before["account"]["total_usage"]
            record["openrouter_key_usage_delta"] = after["key"]["usage"] - before["key"]["usage"]
        except Exception as exc:
            record["balance_after_error"] = str(exc)
        save(run_dir / "run_record.json", record)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic_worker":
        worker(sys.argv[2:], expected_cases=1)
    elif len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2:])
    else:
        run()
