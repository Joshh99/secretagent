import json
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from benchmarks.scorers_natural_plan import eval_meeting_single
from scripts.logs import logger


class NaturalPlanMeetingBenchmark(BaseBenchmark):
    """NaturalPlan meeting planning, scored the same way as the secretagent harness.

    benchmarks/natural_plan/expt.py:NaturalPlanEvaluator calls eval_meeting_single
    on the raw reply and the whole instance, so this imports that same function
    from a verbatim copy of benchmarks/natural_plan/eval_utils.py rather than
    reimplementing it. Both sides therefore agree on what counts as correct.

    A meeting plan cannot be graded against a single gold string: the scorer
    replays the plan against the instance's travel times and availability
    windows and counts valid meetings, then compares that count to the golden
    plan's count. The whole instance therefore travels in the `target` field as
    JSON and is parsed back here.
    """

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        instance = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
        if "constraints" not in instance or "dist_matrix" not in instance:
            raise ValueError("NaturalPlan meeting gold is missing its constraints")
        reply = "" if prediction is None else str(prediction)
        # A malformed plan is a wrong answer, not a crash: the scorer stops at
        # the first step it cannot read and scores the meetings up to there.
        correct = eval_meeting_single(reply, instance)
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
