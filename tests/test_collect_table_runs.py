"""Checks for the Table 2/3 run collector: cost split, watch, seed aggregation."""

import json

import pytest

from scripts.collect_table_runs import (
    Pricing, aggregate, costs, load_pricing, scan_cases,
)

AFLOW_DIR = "C:/Users/STUDENT/aflow-b"


def _write_call(root, phase, digest, index, model, tin, tout, provider=None):
    path = root / "model_calls" / phase / digest[:2] / digest / f"{index:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    response = {"usage": {"prompt_tokens": tin, "completion_tokens": tout}}
    if provider:
        response["provider"] = provider
    path.write_text(json.dumps({
        "schema": 1,
        "request": {"model": model},
        "response": response,
    }), encoding="utf-8")


def test_search_and_running_cost_are_reported_separately(tmp_path):
    """The two answer different questions, so they must not be summed."""
    pricing = Pricing({"exec": {"input": 0.001, "output": 0.002},
                       "opt": {"input": 0.010, "output": 0.020}})
    _write_call(tmp_path, "search", "aa" * 32, 0, "exec", 1000, 1000)
    _write_call(tmp_path, "search", "bb" * 32, 0, "opt", 1000, 1000)
    _write_call(tmp_path, "table3_full_test", "cc" * 32, 0, "exec", 2000, 0)

    report = costs(tmp_path, pricing)
    # Search carries the optimizer too, since finding the workflow is what it paid for.
    assert report["search_usd"] == pytest.approx(0.003 + 0.030)
    assert report["running_usd"] == pytest.approx(0.002)
    assert set(report["phases"]) == {"search", "table3_full_test"}


def test_a_call_without_usage_is_skipped_not_counted_as_free(tmp_path):
    pricing = Pricing({"exec": {"input": 0.001, "output": 0.002}})
    path = tmp_path / "model_calls" / "search" / "dd" / ("d" * 64) / "000000.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "request": {"model": "exec"},
                                "response": {}}), encoding="utf-8")
    assert costs(tmp_path, pricing)["search_usd"] == 0.0


def test_an_unpriced_model_raises_instead_of_reporting_zero(tmp_path):
    """A silent $0.00 is indistinguishable from a genuinely free call."""
    pricing = Pricing({"exec": {"input": 0.001, "output": 0.002}})
    _write_call(tmp_path, "search", "ee" * 32, 0, "some-new-model", 10, 10)
    with pytest.raises(KeyError):
        costs(tmp_path, pricing)


def test_pricing_is_read_from_the_aflow_checkout(tmp_path):
    """One source of truth, so reported cost cannot drift from recorded cost."""
    pricing = load_pricing(AFLOW_DIR)
    # The pinned DeepSeek executor must be priced, or every call would fail.
    assert pricing.get_price("deepseek/deepseek-chat-v3.1", "input") == pytest.approx(0.00027)
    assert pricing.get_price("deepseek/deepseek-chat-v3.1", "output") == pytest.approx(0.001)
    assert pricing.get_price("gemini-2.5-flash-lite", "input") == pytest.approx(0.0001)
    with pytest.raises(KeyError):
        pricing.get_price("not-a-model", "input")


def test_watch_finds_the_connection_error_contamination(tmp_path):
    """The failure that silently corrupted six earlier scoring passes."""
    good = tmp_path / "round_1" / "0.5_20260923_120000.csv"
    good.parent.mkdir(parents=True)
    good.write_text("prediction,score\n1,1.0\n0,0.0\n", encoding="utf-8")
    assert scan_cases(tmp_path)["connection_errors"] == 0

    bad = tmp_path / "round_2" / "0.4_20260923_130000.csv"
    bad.parent.mkdir(parents=True)
    bad.write_text("prediction,score\n1,1.0\nConnection error.,0.0\n", encoding="utf-8")
    scan = scan_cases(tmp_path)
    assert scan["connection_errors"] == 1
    assert scan["case_rows"] == 4
    assert scan["bad_files"][0][1] == 1


def test_seed_aggregation_reports_mean_and_standard_error():
    results = [{"test_accuracy": 0.60, "inference_usd_per_100": 1.0, "test_cases": 100, "round": 3},
               {"test_accuracy": 0.70, "inference_usd_per_100": 2.0, "test_cases": 100, "round": 7},
               {"test_accuracy": 0.80, "inference_usd_per_100": 3.0, "test_cases": 100, "round": 5}]
    summary = aggregate(results)
    assert summary["seeds"] == 3
    assert summary["accuracy"] == pytest.approx(0.70)
    # sample sd 0.1 over sqrt(3)
    assert summary["accuracy_stderr"] == pytest.approx(0.1 / 3 ** 0.5)
    assert summary["rounds"] == [3, 7, 5]


def test_a_single_seed_has_no_spread_rather_than_zero_spread():
    """Reporting 0.0 would claim a precision one run cannot support."""
    summary = aggregate([{"test_accuracy": 0.6, "inference_usd_per_100": 1.0,
                          "test_cases": 100, "round": 1}])
    assert summary["seeds"] == 1
    assert summary["accuracy_stderr"] is None
    assert summary["usd_per_100_stderr"] is None


def test_an_archived_patch_exists_for_the_live_musr_object_run():
    """Rebuilding the patch must not make a finished search unscoreable.

    run_aflow_table_musr.py verifies a run against the exact AFlow source it
    ran with. Adding the Table 2/3 datasets required rebuilding the patch, so
    the pre-rebuild patch stays archived and is matched by checksum.
    """
    import hashlib
    from pathlib import Path

    upstream = Path("benchmarks/COMMON/aflow-rebuttal/upstream")
    archived = {hashlib.sha256(p.read_bytes()).hexdigest()
                for p in upstream.glob("aflow_changes*.patch")}
    manifest = Path("C:/Users/STUDENT/aflow/runs/"
                    "table2_musr3__musr_object__matched__seed1__gemini_2_5_flash_lite/"
                    "manifest.json")
    if not manifest.is_file():
        pytest.skip("the live MuSR Object run is not present on this machine")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["aflow_patch_sha256"]
    assert recorded in archived, "no archived patch matches the running search"


def test_each_call_is_priced_by_the_host_that_served_it(tmp_path):
    """The fp8 hosts charge different rates, so one blended rate would be wrong."""
    from scripts.collect_table_runs import PROVIDER_PRICES
    pricing = Pricing({"deepseek/deepseek-chat-v3.1": {"input": 9.0, "output": 9.0}})
    _write_call(tmp_path, "search", "aa" * 32, 0, "deepseek/deepseek-chat-v3.1",
                1000, 1000, provider="SiliconFlow")
    _write_call(tmp_path, "search", "bb" * 32, 0, "deepseek/deepseek-chat-v3.1",
                1000, 1000, provider="AtlasCloud")
    report = costs(tmp_path, pricing)

    expected = sum(PROVIDER_PRICES[p]["input"] + PROVIDER_PRICES[p]["output"]
                   for p in ("SiliconFlow", "AtlasCloud"))
    assert report["search_usd"] == pytest.approx(expected)
    # The model's own table is never consulted when the host is known.
    assert report["search_usd"] < 1.0
    assert set(report["phases"]["search"]) == {
        "deepseek/deepseek-chat-v3.1 @ SiliconFlow",
        "deepseek/deepseek-chat-v3.1 @ AtlasCloud"}


def test_a_call_with_no_host_falls_back_to_the_model_table(tmp_path):
    """The Gemini optimizer is not routed through OpenRouter, so it has none."""
    pricing = Pricing({"gemini-3.1-pro-preview": {"input": 0.002, "output": 0.012}})
    _write_call(tmp_path, "search", "cc" * 32, 0, "gemini-3.1-pro-preview", 1000, 1000)
    report = costs(tmp_path, pricing)
    assert report["search_usd"] == pytest.approx(0.002 + 0.012)
    assert "gemini-3.1-pro-preview" in report["phases"]["search"]


def test_watch_finds_api_limit_rejections(tmp_path):
    """A rejected call is scored as a wrong answer, so it must be found too.

    11,708 of these went unnoticed for two hours because the scan only looked
    for the connection-error string.
    """
    p = tmp_path / "round_1" / "0.0_20260924_030000.csv"
    p.parent.mkdir(parents=True)
    p.write_text("prediction,score\n"
                 "1,1.0\n"
                 "\"Error code: 403 - {'error': {'message': 'Key limit exceeded "
                 "(total limit).'}}\",0.0\n", encoding="utf-8")
    scan = scan_cases(tmp_path)
    assert scan["limit_errors"] == 1
    assert scan["connection_errors"] == 0
    assert scan["case_rows"] == 2
