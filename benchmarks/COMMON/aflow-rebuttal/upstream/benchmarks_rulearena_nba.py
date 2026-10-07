import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class RuleArenaNBABenchmark(BaseBenchmark):
    """RuleArena NBA, scored the same way as the secretagent harness.

    benchmarks/rulearena/expt.py:RuleArenaEvaluator branches on the gold type.
    Every NBA gold is a boolean (34 true and 12 false across the 46 test cases),
    so that branch reduces to an exact yes/no match and the 1% numeric tolerance
    path is never reached. Scoring this column as a number would be wrong.

    The harness compares bool(predicted_output) against the gold, where the
    prediction is already a Python bool returned by the workflow. An AFlow graph
    returns text instead, and bool() of any non-empty string is True, so the
    reply has to be read as a yes or no here. A reply that says neither, or
    both, is not an answer and scores zero rather than being guessed at.
    """

    TRUE_WORDS = {"true", "yes", "y", "1", "legal", "allowed", "permitted", "valid"}
    FALSE_WORDS = {"false", "no", "n", "0", "illegal", "disallowed",
                   "not allowed", "not permitted", "invalid"}

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    @staticmethod
    def _as_number(text):
        """The number a reply states, or None when it is not just a number."""
        try:
            return float(text)
        except ValueError:
            return None

    def extract_boolean(self, value) -> Optional[bool]:
        """The yes or no in a reply, or None when it is not an answer."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().casefold()
        if not text:
            return None

        # The seed prompt derived from compute_nba_answer asks for a single
        # number, and the harness stores this gold as 1.0 or 0.0, so a numeric
        # reply is the common case and means exactly true or false. Reading it
        # as text instead found both "1" and "0" inside "1.0" and called the
        # reply ambiguous, which scored almost every case wrong.
        numeric = self._as_number(text)
        if numeric is not None:
            if numeric == 1:
                return True
            if numeric == 0:
                return False
            return None

        # Then a bare word answer.
        if text in self.TRUE_WORDS:
            return True
        if text in self.FALSE_WORDS:
            return False

        # Otherwise read the last non-empty line, which is where a verbose reply
        # puts its conclusion, and accept it only when it points one way.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        last = re.sub(r"^[\s*#>-]*(final\s+)?answer\s*[:\-]\s*", "", lines[-1])
        last = last.strip().strip(".!*` ")
        numeric = self._as_number(last)
        if numeric is not None:
            return True if numeric == 1 else False if numeric == 0 else None
        if last in self.TRUE_WORDS:
            return True
        if last in self.FALSE_WORDS:
            return False

        words = set(re.findall(r"[a-z0-9]+", last))
        says_true = bool(words & self.TRUE_WORDS)
        says_false = bool(words & self.FALSE_WORDS)
        if says_true != says_false:
            return says_true
        return None

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        gold = self.extract_boolean(ground_truth)
        if gold is None:
            raise ValueError(f"RuleArena NBA case has no boolean gold answer: {ground_truth!r}")
        predicted = self.extract_boolean(prediction)
        return (1.0 if predicted is not None and predicted == gold else 0.0, str(predicted))

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
