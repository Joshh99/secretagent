"""Declarative registry of the Table 2/3 tasks an AFlow run must reproduce.

Tables 2 and 3 each have eight task columns. Every column was produced by a
saved secretagent run, so the split, case order, gold answers, scorer and cost
definition are already fixed on disk. This module records where each of those
runs lives and how to rebuild its exact case set for AFlow, so that both sides
are graded on the same cases by the same rule.

Nothing here calls a model or the network on import. `run_aflow_table_task.py`
uses it to export, and to refuse a paid run whose cases do not match the table.

Two facts recorded here decide whether a number may go into the table at all:

  * `baseline_model` is the executor model of the saved run behind each column.
    Every Table 2/3 column was produced with DeepSeek-V3.1, while the AFlow runs
    are at Gemini 2.5 Flash-Lite, so a Flash-Lite AFlow number would sit in a
    table whose other columns are a different model. `check_model_match` reports
    that instead of letting it pass silently.
  * `export` groups the tasks that come from one saved run. MedCalc formulas and
    rules are one 1040-case run partitioned by category, so they need one search
    and one test pass between them, not two.
"""

import ast
import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "COMMON" / "results"

# The executor model behind every saved Table 2/3 column.
TABLE_BASELINE_MODEL = "together_ai/deepseek-ai/DeepSeek-V3.1"

# The executor model the AFlow Table 2/3 runs use, and the host serving it.
# OpenRouter routes this model to whichever host is cheapest unless pinned, and
# its default serves it at fp4, so config2.yaml pins SiliconFlow at fp8.
AFLOW_EXECUTOR_MODEL = "deepseek/deepseek-chat-v3.1"
AFLOW_EXECUTOR_HOST = "OpenRouter/SiliconFlow (fp8)"
TABLE_BASELINE_HOST = "Together AI"


@dataclass(frozen=True)
class TableTask:
    """One Table 2/3 column and the AFlow work needed to fill it."""

    key: str
    label: str                 # column heading in Tables 2 and 3
    baseline: str              # saved run whose results.csv defines the column
    expected_cases: int
    split: str                 # secretagent dataset split
    configure: dict            # shuffle_seed / n passed to Dataset.configure
    loader: str                # how to load the split
    gold: str                  # how to compare exported gold to the baseline CSV
    scorer: str                # the rule that decides `correct`
    export: str                # AFlow dataset name; shared entries share a run
    aflow_benchmark: str       # AFlow benchmark class, written or to be written
    adapter_status: str
    baseline_model: str = TABLE_BASELINE_MODEL
    category_filter: tuple = ()   # MedCalc partitions one run into two columns
    # The 50-case set AFlow selects its winning workflow on, recovered from
    # the saved MODO optimizer config so both sides select the same way.
    validation: dict | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def baseline_csv(self) -> Path:
        return RESULTS / self.baseline / "results.csv"

    @property
    def baseline_config(self) -> Path:
        return RESULTS / self.baseline / "config.yaml"


# MuSR murder and team use the same loader, prompt shape and scorer as object,
# which is the task already wired, so they are the cheapest of the seven to add.
_MUSR = dict(
    loader="musr",
    gold="int",
    scorer="exact match on the choice index (MUSREvaluator compares with ==)",
    configure={"shuffle_seed": 42},
    aflow_benchmark="benchmarks.musr_object.MuSRObjectBenchmark",
)

TASKS: dict[str, TableTask] = {
    # Already wired and running as table2_musr3. Present so the checks can be
    # reused and so the registry covers all eight columns; the launcher refuses
    # to write into its workspace.
    "musr_object": TableTask(
        key="musr_object", label="MuSR Object",
        baseline="musr/object/20260425.122302.workflow",
        expected_cases=106, split="object_placements_test",
        export="MuSRObjectPlacements",
        validation={"split": "object_placements_val",
                    "configure": {"shuffle_seed": 42, "n": 50}},
        adapter_status="done; running as table2_musr3",
        notes="Do not re-wire. Scored by run_aflow_table_musr.py.",
        **_MUSR),
    "musr_murder": TableTask(
        key="musr_murder", label="MuSR Murder",
        baseline="musr/murder/20260425.113115.workflow",
        expected_cases=100, split="murder_mysteries_test",
        export="MuSRMurderMysteries",
        # MODO selected this column on 50 cases drawn from the same test
        # split the table reports, unlike object and team which used their
        # _val splits. AFlow matches it so both sides select under the same
        # rule; holding only AFlow to a clean split would understate it.
        # The overlap is real for both and belongs in the paper.
        validation={"split": "murder_mysteries_test",
                    "configure": {"shuffle_seed": 42, "n": 50},
                    "overlaps_test": True},
        adapter_status="reuses MuSRObjectBenchmark; needs dataset registration",
        **_MUSR),
    "musr_team": TableTask(
        key="musr_team", label="MuSR Team",
        baseline="musr/team/20260425.125020.workflow",
        expected_cases=100, split="team_allocation_test",
        export="MuSRTeamAllocation",
        validation={"split": "team_allocation_val",
                    "configure": {"shuffle_seed": 42, "n": 50}},
        adapter_status="reuses MuSRObjectBenchmark; needs dataset registration",
        **_MUSR),

    # NaturalPlan gold is the whole instance dict, because the scorer replays
    # the plan against its constraints. A scalar target cannot carry that, so
    # the adapter has to hold the instance and call the secretagent scorer.
    "naturalplan_meeting": TableTask(
        key="naturalplan_meeting", label="NaturalPlan Meeting",
        baseline="natural_plan/meeting/20260504.065725.workflow",
        expected_cases=100, split="meeting",
        configure={"shuffle_seed": 42, "n": 100},
        loader="natural_plan", gold="instance_digest",
        scorer="eval_meeting_single replays the plan against the instance constraints",
        export="NaturalPlanMeeting",
        validation={"split": "meeting", "configure": {"shuffle_seed": 42, "n": 50},
                    "extra": {"partition": "valid", "prompt_mode": "0shot"}},
        aflow_benchmark="benchmarks.naturalplan_meeting.NaturalPlanMeetingBenchmark",
        adapter_status="written: NaturalPlanMeetingBenchmark; needs registration",
        extra={"partition": "test", "prompt_mode": "0shot"},
        notes="Gold is the full instance dict, not a string answer."),
    "naturalplan_trip": TableTask(
        key="naturalplan_trip", label="NaturalPlan Trip",
        baseline="natural_plan/trip/20260504.074537.workflow",
        expected_cases=100, split="trip",
        configure={"shuffle_seed": 42, "n": 100},
        loader="natural_plan", gold="instance_digest",
        scorer="eval_trip_single replays the itinerary against cities and durations",
        export="NaturalPlanTrip",
        validation={"split": "trip", "configure": {"shuffle_seed": 42, "n": 50},
                    "extra": {"partition": "valid", "prompt_mode": "0shot"}},
        aflow_benchmark="benchmarks.naturalplan_trip.NaturalPlanTripBenchmark",
        adapter_status="written: NaturalPlanTripBenchmark; needs registration",
        extra={"partition": "test", "prompt_mode": "0shot"},
        notes="Gold is the full instance dict, not a string answer."),

    "rulearena_nba": TableTask(
        key="rulearena_nba", label="RuleArena NBA",
        baseline="rulearena/nba/20260430.023413.workflow",
        expected_cases=46, split="test",
        configure={"shuffle_seed": 137},
        loader="rulearena", gold="bool",
        # RuleArenaEvaluator branches on the gold type. All 46 NBA golds are
        # booleans (34 true, 12 false), so this column is an exact yes/no match
        # and the 1% tolerance path is never taken.
        scorer="exact boolean match; every NBA gold is a boolean",
        export="RuleArenaNBA",
        # All 42 valid cases, which is what the optimizer used; there is no
        # n in its config, so this set is 42 rather than 50.
        validation={"split": "valid", "configure": {"shuffle_seed": 137},
                    "extra": {"domain": "nba"}},
        aflow_benchmark="benchmarks.rulearena_nba.RuleArenaNBABenchmark",
        adapter_status="written: RuleArenaNBABenchmark; needs registration",
        extra={"domain": "nba"},
        notes="46 cases is the smallest column; one case moves it 2.2 points."),

    # One saved 1040-case run supplies both MedCalc columns, so one search and
    # one test pass covers them and the partition happens afterwards.
    "medcalc_formulas": TableTask(
        key="medcalc_formulas", label="MedCalc Formulas",
        baseline="medcalc/formulas/20260425.233811.workflow",
        expected_cases=660, split="test",
        configure={"shuffle_seed": 42},
        loader="medcalc", gold="float",
        scorer="calculate_accuracy within the per-case lower and upper limits",
        export="MedCalcTest",
        # The MedCalc validation set, confirmed by its author, is 275 cases
        # drawn from the train split of ncbi/MedCalc-Bench-v1.2 by
        # stratified_sample at seed 42; filtering those by category afterwards
        # gives 205 formulas and 64 rules. Reproduced exactly here.
        #
        # AFlow selects on 50 cases for every task, so rather than draw an
        # independent 50, which is not a subset of the official set, the 50 are
        # drawn from those 275 by the same function and seed. Our validation is
        # therefore a subset of the official one: 70% formulas and 28% rules
        # against the 275's 75/23, and closer to the test split's 63/37.
        # Drawn from train, so no overlap with the 1040 reported test cases.
        validation={"split": "train", "configure": {},
                    "official_n": 275, "stratified": 50},
        aflow_benchmark="benchmarks.medcalc.MedCalcBenchmark",
        adapter_status="written: MedCalcBenchmark; needs registration",
        category_filter=("dosage", "lab test", "physical"),
        notes="Shares one 1040-case run and test pass with MedCalc Rules."),
    "medcalc_rules": TableTask(
        key="medcalc_rules", label="MedCalc Rules",
        baseline="medcalc/rules/20260425.233811.workflow",
        expected_cases=380, split="test",
        configure={"shuffle_seed": 42},
        loader="medcalc", gold="float",
        scorer="calculate_accuracy within the per-case lower and upper limits",
        export="MedCalcTest",
        validation={"split": "train", "configure": {},
                    "official_n": 275, "stratified": 50},
        aflow_benchmark="benchmarks.medcalc.MedCalcBenchmark",
        adapter_status="written: MedCalcBenchmark; needs registration",
        category_filter=("diagnosis", "risk", "severity"),
        notes="Shares one 1040-case run and test pass with MedCalc Formulas."),
}

# The seven columns this handoff covers. MuSR Object is wired already.
PENDING = [key for key in TASKS if key != "musr_object"]


def read_baseline(task: TableTask) -> list[dict]:
    """The saved rows that define this table column."""
    if not task.baseline_csv.is_file():
        raise FileNotFoundError(f"{task.key}: missing baseline {task.baseline_csv}")
    with task.baseline_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != task.expected_cases:
        raise ValueError(f"{task.key}: baseline has {len(rows)} cases, "
                         f"the table column needs {task.expected_cases}")
    return rows


def _as_float(value) -> float:
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return 1.0 if text.lower() == "true" else 0.0
    return float(text)


def baseline_cell(task: TableTask) -> dict:
    """Reproduce the printed table column from the saved rows.

    hero_table.py reports mean(correct) and mean(cost) * 100, so recomputing
    both here is what makes an AFlow number comparable to the column.
    """
    rows = read_baseline(task)
    correct = [_as_float(row["correct"]) for row in rows]
    cost = [_as_float(row["cost"]) for row in rows]
    return {
        "task": task.key,
        "cases": len(rows),
        "accuracy": sum(correct) / len(correct),
        "usd_per_100": sum(cost) / len(cost) * 100,
        "case_names": [row["case_name"] for row in rows],
    }


def canonical_instance(value) -> str:
    """One canonical text form for an instance dict, whatever it arrived as.

    The saved CSV holds a Python dict repr, while the export holds JSON, so the
    two never match as text even when they carry the same instance. Both are
    parsed back into a dict and re-rendered the same way before comparison.
    """
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            value = ast.literal_eval(text)
    return json.dumps(value, sort_keys=True, default=str)


def instance_digest(value) -> str:
    """A stable digest of an instance dict, independent of how it was rendered."""
    return hashlib.sha256(canonical_instance(value).encode("utf-8")).hexdigest()


def baseline_gold(task: TableTask) -> list:
    """Gold answers from the baseline, in the comparable form for this task."""
    rows = read_baseline(task)
    if task.gold == "int":
        return [int(float(row["expected_output"])) for row in rows]
    if task.gold == "float":
        return [float(row["expected_output"]) for row in rows]
    if task.gold == "bool":
        # A boolean gold is saved as 1.0 or 0.0, so compare it as a number.
        return [_as_float(row["expected_output"]) == 1.0 for row in rows]
    if task.gold == "instance_digest":
        # The instance is a Python dict repr in the CSV, so compare a digest of
        # its text rather than parsing it back into a structure.
        return [instance_digest(row["expected_output"]) for row in rows]
    raise ValueError(f"{task.key}: unknown gold form {task.gold!r}")


def _gold_matches(task: TableTask, exported, baseline) -> bool:
    if exported is None:
        return False
    if task.gold == "int":
        return int(float(exported)) == baseline
    if task.gold == "float":
        return abs(float(exported) - baseline) <= 1e-9
    if task.gold == "bool":
        return (_as_float(exported) == 1.0) == baseline
    if task.gold == "instance_digest":
        return instance_digest(exported) == baseline
    raise ValueError(f"{task.key}: unknown gold form {task.gold!r}")


def check_case_alignment(task: TableTask, exported: list[dict]) -> tuple[list[str], list[str]]:
    """Compare an export against the table column. Returns (blockers, warnings).

    Cases are matched by name rather than by position. What has to hold is that
    both sides grade the same cases against the same gold answers; a different
    ordering of that same set changes neither mean accuracy nor mean cost.
    MedCalc is the case in point: its saved order is not reproducible from the
    saved config, while its case set reproduces exactly. Order is still
    reported, because an unexpected reordering elsewhere is worth seeing.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    baseline = read_baseline(task)

    if len(exported) != len(baseline):
        blockers.append(f"exported {len(exported)} cases, "
                        f"the table column has {len(baseline)}")
        return blockers, warnings

    def short_name(row) -> str:
        return str(row.get("case_name", "")).split("/")[-1]

    exported_names = [short_name(row) for row in exported]
    table_names = [row["case_name"] for row in baseline]

    missing = sorted(set(table_names) - set(exported_names))
    extra = sorted(set(exported_names) - set(table_names))
    if missing or extra:
        if missing:
            blockers.append(f"{len(missing)} table cases are not in the export, "
                            f"first few: {missing[:5]}")
        if extra:
            blockers.append(f"{len(extra)} exported cases are not in the table, "
                            f"first few: {extra[:5]}")
        return blockers, warnings

    gold_by_name = dict(zip(table_names, baseline_gold(task)))
    for name, row in zip(exported_names, exported):
        if not _gold_matches(task, row.get("target"), gold_by_name[name]):
            # NaturalPlan gold is a whole instance, so print a stub of it.
            shown = str(row.get("target"))
            if len(shown) > 120:
                shown = shown[:120] + f"... ({len(shown)} chars)"
            blockers.append(f"gold answer differs for {name}: exported {shown}")
            break

    if exported_names != table_names:
        moved = sum(1 for a, b in zip(exported_names, table_names) if a != b)
        warnings.append(f"same {len(table_names)} cases as the table but in a "
                        f"different order ({moved} positions differ); accuracy "
                        f"and mean cost are unaffected")
    return blockers, warnings


def model_family(name: str) -> str:
    """A comparable key for a model name, independent of who serves it.

    `together_ai/deepseek-ai/DeepSeek-V3.1` and `deepseek/deepseek-chat-v3.1`
    are the same weights reached through different providers, and the naming
    only differs because each provider labels its own route.
    """
    leaf = name.split("/")[-1].lower()
    return leaf.replace("-chat-", "-").replace("_", "-")


def check_model_match(task: TableTask, aflow_model: str = AFLOW_EXECUTOR_MODEL,
                      aflow_host: str = AFLOW_EXECUTOR_HOST) -> list[str]:
    """Report how the AFlow executor differs from the column it will sit beside.

    Neither case stops a run. They exist so a difference is stated every time
    rather than discovered once it is already in the table.
    """
    if model_family(task.baseline_model) != model_family(aflow_model):
        return [f"{task.label}: the table column was produced with "
                f"{task.baseline_model}, the AFlow run uses {aflow_model}; "
                f"these are different models and the column would not be comparable"]
    if aflow_host != TABLE_BASELINE_HOST:
        return [f"{task.label}: same model as the column ({aflow_model}), but "
                f"served by {aflow_host} where the column used "
                f"{TABLE_BASELINE_HOST}; state the host in the paper"]
    return []


def export_group(name: str) -> list[TableTask]:
    """The table columns sharing one AFlow dataset, search and test pass."""
    return [task for task in TASKS.values() if task.export == name]


def dataset_path(task: TableTask, aflow_dir: Path) -> Path:
    return Path(aflow_dir) / "data" / "datasets" / f"{task.export.lower()}_table_test.jsonl"


def write_jsonl(rows: list[dict], path: Path) -> str:
    """Write the export and record its checksum next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest
