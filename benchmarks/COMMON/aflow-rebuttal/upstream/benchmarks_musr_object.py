import re
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MuSRObjectBenchmark(BaseBenchmark):
    """MuSR object placements, scored the same way as the secretagent harness.

    benchmarks/musr/expt.py:MUSREvaluator compares the predicted choice index to
    answer_index with ==, so this is exact integer match. Both arms therefore
    agree on what counts as correct.
    """

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_index(self, s: str):
        """The answer index, or None when the reply is not an answer.

        The seed prompt asks for a bare integer, so that is the fast path. A
        verbose but unambiguous reply ("Final answer: 3") is still accepted, since
        rejecting it would penalise the arm for wording rather than reasoning.

        What is rejected is list output. A reply beginning "1. object=tablecloth"
        is the model performing a subtask instead of answering, and reading its
        enumeration marker as the choice index scored those replies as correct by
        accident. The marker is stripped, and a line carrying more than one
        integer is treated as ambiguous rather than guessed at.
        """
        if s is None:
            return None
        text = str(s).strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        last = re.sub(r"^-?\d+\s*[.)]\s*", "", lines[-1])
        nums = re.findall(r"-?\d+", last)
        return int(nums[0]) if len(nums) == 1 else None

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        gold = self.extract_index(ground_truth)
        pred = self.extract_index(prediction)
        if gold is None:
            raise ValueError(f"MuSR object case has no integer gold answer: {ground_truth!r}")
        return (1.0 if pred is not None and pred == gold else 0.0, str(pred))

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
