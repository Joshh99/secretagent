"""No-network check that AFlow charges each concurrent example only its own calls."""

import asyncio
import math
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

from benchmarks.benchmark import BaseBenchmark  # noqa: E402
from scripts.async_llm import _CASE_USAGE  # noqa: E402


class FakeBenchmark(BaseBenchmark):
    def calculate_score(self, expected_output, prediction):
        return 0, prediction

    def get_result_columns(self):
        return ["case", "score", "cost"]

    async def evaluate_problem(self, problem, agent):
        for _ in range(problem):
            _CASE_USAGE.get().add_usage("gemini-2.5-flash-lite", 10, 1)
            await asyncio.sleep(0)
        return problem, 1.0, 999.0  # Deliberately wrong workflow-reported cost.


async def main():
    bench = FakeBenchmark("check", "", "")
    rows = await bench.evaluate_all_problems([1, 3, 2], None, max_concurrent_tasks=3)
    unit_cost = 10 * 0.0001 / 1000 + 1 * 0.0004 / 1000
    assert all(math.isclose(row[2], calls * unit_cost, rel_tol=1e-12)
               for row, calls in zip(rows, [1, 3, 2])), rows
    with tempfile.TemporaryDirectory() as directory:
        bench.log_path = directory
        score, average, total = bench.save_results_to_csv(rows, bench.get_result_columns())
        assert score == 1.0
        assert math.isclose(total, 6 * unit_cost, rel_tol=1e-12)
        assert math.isclose(average, 2 * unit_cost, rel_tol=1e-12)
    print("AFlow concurrent per-case cost isolation passed")


if __name__ == "__main__":
    asyncio.run(main())
