import re
from typing import Any, Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from benchmarks.scorers_medcalc import calculate_accuracy
from scripts.logs import logger


def extract_number(value: Any) -> Optional[float]:
    """Read a number out of a reply.

    Copied from benchmarks/medcalc/expt.py:_extract_number so that both sides
    parse a prediction the same way. tests/test_aflow_table_adapters.py asserts
    this stays in step with that original.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if s.startswith('**exception'):
        return None
    try:
        return float(s)
    except ValueError:
        pass
    match = re.search(r'<answer>\s*([\d.eE+-]+)\s*</answer>', s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    match = re.search(r'ANSWER:\s*([\d.eE+-]+)', s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    numbers = re.findall(r'-?\d+\.?\d*', s)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


class MedCalcBenchmark(BaseBenchmark):
    """MedCalc-Bench, scored the same way as the secretagent harness.

    benchmarks/medcalc/expt.py:MedCalcEvaluator scores a case with
    calculate_accuracy and reports `correct` as its is_within_tolerance flag,
    so this imports that same function from a verbatim copy of
    benchmarks/medcalc/accuracy.py rather than reimplementing it.

    A case cannot be graded from its gold value alone. calculate_accuracy needs
    the case's lower limit, upper limit, output type and category, because
    formula categories allow a tolerance while rule categories require an exact
    match. Those four fields travel alongside `target` in each row.

    One 1040-case run fills both the MedCalc Formulas and MedCalc Rules columns
    of Tables 2 and 3; the partition by category happens after scoring.
    """

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def score_case(self, problem: dict, prediction) -> Tuple[float, str]:
        output_type = str(problem.get("output_type") or "numeric")
        lowered = output_type.lower()
        # Date and weeks/days replies are scored from the raw text, because
        # extracting a number from "08/31/2023" would collapse it to 2023.
        if lowered == "date" or "week" in lowered or "day" in lowered:
            predicted_for_accuracy = prediction
        else:
            predicted_for_accuracy = extract_number(prediction)

        accuracy = calculate_accuracy(
            predicted=predicted_for_accuracy,
            ground_truth=problem["target"],
            lower_limit=problem.get("lower_limit"),
            upper_limit=problem.get("upper_limit"),
            output_type=output_type,
            category=problem.get("category", "formula-based"),
        )
        return float(accuracy.is_within_tolerance), str(predicted_for_accuracy)

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        # Kept for the BaseBenchmark contract. It cannot see the per-case limits,
        # so scoring goes through score_case, which does.
        raise NotImplementedError(
            "MedCalc needs each case's limits, output type and category; "
            "use score_case(problem, prediction)")

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["input"]
        expected_output = problem["target"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted_output = self.score_case(problem, output)
            return input_text, output, expected_output, score, cost
        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
