"""Partition NaturalPlan data into train/valid/test splits (100 each).

The TEST split is defined as the 100 cases that were actually used for
the paper's baseline experiments (shuffle_seed=42 + n=100 on the full
dataset). We extract their case_names from an existing results CSV so
that the test set is EXACTLY the set that was evaluated — no re-runs
needed.

TRAIN and VALID are each 100 cases, freshly stratified-sampled from the
pool of cases NOT in test (and disjoint from each other), with seeds
42 and 43 respectively.

The previous 50-example splits are preserved as `{task}_{split}_50.json`
so past experiments keyed on those files (via dataset.partition=train_50
etc.) still work.

Usage:
    cd benchmarks/natural_plan
    uv run python data/partition.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent
_BENCHMARK_DIR = _DATA_DIR.parent
_SECRETAGENT_ROOT = _BENCHMARK_DIR.parent.parent

sys.path.insert(0, str(_SECRETAGENT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from secretagent.dataset import Dataset, Case


# ── Task configs ──

TASKS = {
    'calendar': {
        'data_file': 'calendar_scheduling.json',
        'strata_key': lambda inst: f"({inst['num_people']},{inst['num_days']})",
        'prompt_field': 'prompt_0shot',
        # Any results CSV from the paper's N=100 seed=42 runs — all
        # strategies on a given subtask evaluated the same 100 case names.
        'test_case_source_csv': _BENCHMARK_DIR / 'results/20260420.201651.workflow/results.csv',
    },
    'meeting': {
        'data_file': 'meeting_planning.json',
        'strata_key': lambda inst: str(inst['num_people']),
        'prompt_field': 'prompt_0shot',
        'test_case_source_csv': _BENCHMARK_DIR / 'results/20260421.021958.structured_baseline/results.csv',
    },
    'trip': {
        'data_file': 'trip_planning.json',
        'strata_key': lambda inst: str(inst['num_cities']),
        'prompt_field': 'prompt_0shot',
        'test_case_source_csv': _BENCHMARK_DIR / 'results/20260421.052749.workflow/results.csv',
    },
}

NON_TEST_SPLITS = {
    'train': {'seed': 42, 'n': 100},
    'valid': {'seed': 43, 'n': 100},
}


def stratified_sample(
    data: dict[str, dict],
    strata_key: Callable[[dict], str],
    n: int,
    seed: int,
) -> dict[str, dict]:
    """Pick exactly `n` examples, distributed across strata as evenly as
    possible (round-robin across shuffled strata). Falls back to random
    if n > len(data)."""
    import random
    strata: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for k, inst in data.items():
        strata[strata_key(inst)].append((k, inst))

    rng = random.Random(seed)
    for items in strata.values():
        rng.shuffle(items)

    stratum_keys = sorted(strata.keys())
    picks: list[tuple[str, dict]] = []
    while len(picks) < n and any(strata[k] for k in stratum_keys):
        for sk in stratum_keys:
            if strata[sk] and len(picks) < n:
                picks.append(strata[sk].pop(0))
    return dict(picks[:n])


def make_cases(data: dict[str, dict], prompt_field: str) -> list[Case]:
    cases = []
    for key, inst in data.items():
        prompt = inst.get(prompt_field, inst.get('prompt_5shot', ''))
        cases.append(Case(
            name=key,
            input_args=(prompt,),
            expected_output=inst,
        ))
    return cases


def make_cases_in_order(data: dict[str, dict], ordered_keys: list[str], prompt_field: str) -> list[Case]:
    cases = []
    for key in ordered_keys:
        if key not in data:
            raise KeyError(f'case {key} not found in data')
        inst = data[key]
        prompt = inst.get(prompt_field, inst.get('prompt_5shot', ''))
        cases.append(Case(
            name=key,
            input_args=(prompt,),
            expected_output=inst,
        ))
    return cases


def save_dataset(filepath: Path, task: str, split: str, cases: list[Case]):
    dataset = Dataset(
        name=f'naturalplan_{task}',
        split=f'{task}_{split}',
        cases=cases,
    )
    with open(filepath, 'w') as fp:
        fp.write(dataset.model_dump_json(indent=2))
    print(f'  {filepath.name}: {len(cases)} cases')


def partition_task(task: str, cfg: dict):
    data_path = _DATA_DIR / cfg['data_file']
    with open(data_path) as f:
        all_data = json.load(f)

    print(f'{task}: {len(all_data)} total examples')

    # --- TEST: the 100 cases actually evaluated in the paper baseline ---
    test_csv_path = cfg['test_case_source_csv']
    if not test_csv_path.exists():
        raise FileNotFoundError(
            f'Expected paper test-set source CSV at {test_csv_path}; '
            f'if you moved results, update TASKS[{task!r}]["test_case_source_csv"].'
        )
    test_names = pd.read_csv(test_csv_path)['case_name'].tolist()
    missing = [n for n in test_names if n not in all_data]
    if missing:
        raise ValueError(f'{task}: {len(missing)} test case names not in full data')

    test_cases = make_cases_in_order(all_data, test_names, cfg['prompt_field'])
    save_dataset(_DATA_DIR / f'{task}_test.json', task, 'test', test_cases)

    # --- TRAIN and VALID: 100 each, disjoint from test and each other ---
    test_set = set(test_names)
    used_keys: set[str] = set()
    for split_name, split_cfg in NON_TEST_SPLITS.items():
        available = {k: v for k, v in all_data.items()
                     if k not in test_set and k not in used_keys}
        sampled = stratified_sample(
            available, cfg['strata_key'], split_cfg['n'], split_cfg['seed'],
        )
        used_keys.update(sampled.keys())
        cases = make_cases(sampled, cfg['prompt_field'])
        save_dataset(_DATA_DIR / f'{task}_{split_name}.json', task, split_name, cases)

    # Sanity: confirm disjointness
    overlap_tt = test_set & used_keys
    assert not overlap_tt, f'{task}: test/train-or-valid overlap {overlap_tt}'


if __name__ == '__main__':
    for task, cfg in TASKS.items():
        partition_task(task, cfg)
    print('\nDone. train/valid/test splits created (100 each, all disjoint).')
    print('  test = the exact 100 cases used in paper baselines')
    print('  train, valid = 100 each, stratified from non-test pool')
    print('  Old 50-example splits preserved as *_50.json')
