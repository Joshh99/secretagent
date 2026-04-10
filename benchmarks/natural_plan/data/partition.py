"""Partition NaturalPlan data into train/valid/test splits.

Uses stratified sampling (same strata as expt.py) to produce balanced splits.
Train uses seed=42 (same 50 IDs as existing experiments).
Valid uses seed=43, test uses seed=44.

Usage:
    cd benchmarks/natural_plan
    uv run python data/partition.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

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
    },
    'meeting': {
        'data_file': 'meeting_planning.json',
        'strata_key': lambda inst: str(inst['num_people']),
        'prompt_field': 'prompt_0shot',
    },
    'trip': {
        'data_file': 'trip_planning.json',
        'strata_key': lambda inst: str(inst['num_cities']),
        'prompt_field': 'prompt_0shot',
    },
}

SPLITS = {
    'train': {'seed': 42, 'n': 50},
    'valid': {'seed': 43, 'n': 50},
    'test':  {'seed': 44, 'n': 50},
}


def stratified_sample(
    data: dict[str, dict],
    strata_key: Callable[[dict], str],
    n: int,
    seed: int,
) -> dict[str, dict]:
    """Stratified sampling: pick n/num_strata examples from each stratum."""
    import random
    strata: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, instance in data.items():
        s = strata_key(instance)
        strata[s].append((key, instance))
    num_strata = len(strata)
    per_stratum = max(1, n // num_strata)
    rng = random.Random(seed)
    sampled = {}
    for s_key in sorted(strata.keys()):
        items = strata[s_key]
        rng.shuffle(items)
        for k, v in items[:per_stratum]:
            sampled[k] = v
    return sampled


def make_cases(data: dict[str, dict], task: str, split: str, prompt_field: str) -> list[Case]:
    """Convert raw data dict to list of Cases."""
    cases = []
    for key, inst in data.items():
        prompt = inst.get(prompt_field, inst.get('prompt_5shot', ''))
        cases.append(Case(
            name=key,
            input_args=(prompt,),
            expected_output=inst,
        ))
    return cases


def save_dataset(filepath: Path, task: str, split: str, cases: list[Case]):
    """Save dataset as JSON."""
    dataset = Dataset(
        name=f'naturalplan_{task}',
        split=f'{task}_{split}',
        cases=cases,
    )
    with open(filepath, 'w') as fp:
        fp.write(dataset.model_dump_json(indent=2))
    print(f'  {filepath.name}: {len(cases)} cases')


def partition_task(task: str, cfg: dict):
    """Partition one task into train/valid/test."""
    data_path = _DATA_DIR / cfg['data_file']
    with open(data_path) as f:
        all_data = json.load(f)

    print(f'{task}: {len(all_data)} total examples')

    # Sample each split, ensuring no overlap
    used_keys: set[str] = set()
    for split_name, split_cfg in SPLITS.items():
        # Remove already-used keys before sampling
        available = {k: v for k, v in all_data.items() if k not in used_keys}
        sampled = stratified_sample(
            available, cfg['strata_key'], split_cfg['n'], split_cfg['seed'],
        )
        used_keys.update(sampled.keys())

        cases = make_cases(sampled, task, split_name, cfg['prompt_field'])
        out_path = _DATA_DIR / f'{task}_{split_name}.json'
        save_dataset(out_path, task, split_name, cases)


if __name__ == '__main__':
    for task, cfg in TASKS.items():
        partition_task(task, cfg)
    print('\nDone. Train/valid/test splits created.')
