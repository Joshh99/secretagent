import json
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from benchmarks.scorers_natural_plan import eval_trip_single
from scripts.logs import logger


class NaturalPlanTripBenchmark(BaseBenchmark):
    """NaturalPlan trip planning, scored the same way as the secretagent harness.

    benchmarks/natural_plan/expt.py:NaturalPlanEvaluator calls eval_trip_single
    on the raw reply and the whole instance, so this imports that same function
    from a verbatim copy of benchmarks/natural_plan/eval_utils.py rather than
    reimplementing it. Both sides therefore agree on what counts as correct.

    The scorer parses the itinerary out of the reply and requires every city and
    every stay length to match in order, so it needs the instance's city list
    and durations. Those travel in the `target` field as JSON.
    """

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        instance = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
        if "cities" not in instance or "durations" not in instance:
            raise ValueError("NaturalPlan trip gold is missing its cities or durations")
        reply = "" if prediction is None else str(prediction)
        # An unparseable itinerary scores zero rather than raising.
        correct = eval_trip_single(reply, instance)
        return (1.0 if correct else 0.0, reply)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["input"]
        expected_output = problem["target"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted_output = self.calculate_score(expected_output, output)
            return input_text, output, expected_output, score, cost
        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
