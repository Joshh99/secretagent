"""DesignBench benchmark experiment.

Example CLI commands:

    # run with default config
    uv run python benchmarks/designbench/expt.py run --config-file conf/conf.yaml

    # run first 10 examples
    uv run python benchmarks/designbench/expt.py run --config-file conf/conf.yaml dataset.n=10

    # change model
    uv run python benchmarks/designbench/expt.py run --config-file conf/conf.yaml llm.model=gpt-4o-mini

    # skip visual evaluation (generation only)
    uv run python benchmarks/designbench/expt.py run --config-file conf/conf.yaml benchmark.skip_eval=true
"""

import json
import sys
import base64
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from tqdm import tqdm

# Allow running from any directory: add project src/ and this benchmark dir to path
_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from secretagent import config, savefile, record
from secretagent.core import implement_via_config
from secretagent.dataset import Dataset, Case
from secretagent.evaluate import Evaluator
import secretagent.implement_vlm

import ptools
from eval_util import render_to_screenshot, evaluate_visual

FRAMEWORK_TO_EXT = {
    'vanilla': 'html',
    'react': 'jsx',
    'vue': 'vue',
    'angular': 'html',
}


def _configure_designbench_imports() -> None:
    """Add DesignBench evaluator package to sys.path."""
    cfg_root = config.get('designbench.root')
    if cfg_root:
        root = Path(cfg_root)
    else:
        # Default: sibling repository at ../DesignBench relative to secretagent root
        root = _PROJECT_ROOT.parent / 'DesignBench'

    evaluator_dir = root / 'code' / 'evaluator'
    code_dir = root / 'code'
    sys.path.insert(0, str(evaluator_dir))
    sys.path.insert(0, str(code_dir))


def _set_designbench_generate_prompt(framework: str) -> None:
    """Set generate_code docstring to the original DesignBench system prompt."""
    try:
        from prompt.generation_prompt import get_design_generation_prompt  # type: ignore
        from utils import Framework  # type: ignore
    except ImportError:
        return
    system_prompt, _ = get_design_generation_prompt(Framework(framework))
    ptools.generate_code.doc = system_prompt


def _encode_image_base64(path: Path) -> str:
    """Encode an image file for multimodal model input."""
    return base64.b64encode(path.read_bytes()).decode('ascii')


def _find_dataset_reference_image(item_dir: Path, item_id: str) -> Path | None:
    """Find an existing reference image shipped with DesignBench data."""
    candidates = [
        item_dir / f'{item_id}.png',
        item_dir / f'{item_id}_p.png',
        item_dir / f'{item_id}.jpg',
        item_dir / f'{item_id}.jpeg',
        item_dir / f'{item_id}.webp',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_dataset(framework: str, max_reference_chars: int | None) -> Dataset:
    """Load DesignBench generation data for one framework."""
    data_root = _BENCHMARK_DIR / 'data' / 'generation' / framework
    if not data_root.exists():
        raise FileNotFoundError(f'Framework directory not found: {data_root}')

    cases: list[Case] = []
    for item_dir in sorted(
        data_root.iterdir(),
        key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name),
    ):
        if not item_dir.is_dir():
            continue

        item_id = item_dir.name
        html_path = item_dir / f'{item_id}.html'
        meta_path = item_dir / f'{item_id}.json'
        if not html_path.exists() or not meta_path.exists():
            continue

        with open(meta_path) as f:
            metadata = json.load(f)
        reference_html = html_path.read_text(encoding='utf-8', errors='ignore')
        if max_reference_chars and max_reference_chars > 0:
            reference_html = reference_html[:max_reference_chars]

        expected = {
            'id': item_id,
            'framework': framework,
            'reference_html_path': str(html_path),
        }
        input_kw = {}
        reference_image = _find_dataset_reference_image(item_dir, item_id)
        if reference_image is not None:
            encoded_image = _encode_image_base64(reference_image)
            expected['reference_image_path'] = str(reference_image)
            input_kw = {'images': {'reference': encoded_image}}
        cases.append(Case(
            name=f'{framework}.{item_id}',
            input_args=(reference_html, framework, metadata),
            input_kw=input_kw,
            expected_output=expected,
        ))

    return Dataset(name='designbench', split=framework, cases=cases)


class DesignBenchEvaluator(Evaluator):
    """Evaluator for DesignBench visual similarity."""

    def __init__(self, output_framework: str, skip_eval: bool):
        self.output_framework = output_framework
        self.skip_eval = skip_eval
        self.artifacts_dir: Path | None = None
        self.visual_eval_enabled = not skip_eval
        self.visual_eval_skip_reason: str | None = None

    def compare_predictions(self, predicted_output: Any, expected_output: Any) -> dict[str, Any]:
        if self.artifacts_dir is None:
            raise RuntimeError('Evaluator artifacts_dir not initialized')

        item_id = str(expected_output['id'])
        framework = str(expected_output['framework'])
        reference_image_path = expected_output.get('reference_image_path')
        ext = FRAMEWORK_TO_EXT.get(self.output_framework, FRAMEWORK_TO_EXT.get(framework, 'html'))

        code_path = self.artifacts_dir / f'{item_id}.{ext}'
        raw_path = self.artifacts_dir / f'{item_id}_raw.txt'
        generated_png = self.artifacts_dir / f'{item_id}_generated.png'
        metrics_path = self.artifacts_dir / f'{item_id}_metrics.json'

        raw_text = str(predicted_output)
        raw_path.write_text(raw_text, encoding='utf-8')
        if raw_text.startswith('**exception raised**'):
            result = {
                'error': raw_text,
                'code_path': str(code_path),
                'raw_path': str(raw_path),
            }
            metrics_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            return result

        code_path.write_text(raw_text, encoding='utf-8')
        reference_png = Path(reference_image_path) if reference_image_path else None
        result: dict[str, Any] = {
            'code_path': str(code_path),
            'raw_path': str(raw_path),
            'reference_screenshot': str(reference_png) if reference_png else None,
            'generated_screenshot': str(generated_png),
        }

        if not self.visual_eval_enabled:
            result['eval_skipped'] = True
            if self.visual_eval_skip_reason:
                result['eval_skip_reason'] = self.visual_eval_skip_reason
            metrics_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            return result

        try:
            if reference_png is None or not reference_png.exists():
                result['render_failed'] = True
                result['eval_error'] = 'missing reference_image_path for visual comparison'
                metrics_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
                return result

            rendered = render_to_screenshot(
                code_path=str(code_path),
                save_path=str(generated_png),
                framework=self.output_framework,
            )
            if rendered and generated_png.exists():
                result.update(evaluate_visual(str(reference_png), str(generated_png)))
            else:
                result['render_failed'] = True
        except ImportError as ex:
            # Disable visual eval after first dependency failure.
            self.visual_eval_enabled = False
            self.visual_eval_skip_reason = f'{type(ex).__name__}: {ex}'
            result['eval_skipped'] = True
            result['eval_skip_reason'] = self.visual_eval_skip_reason
        except Exception as ex:
            result['eval_error'] = f'{type(ex).__name__}: {ex}'

        metrics_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result

    def measure(self, example: Case, interface) -> dict[str, Any]:
        """Measure one case, forwarding args and kwargs from Dataset.Case."""
        input_args = tuple(example.input_args or ())
        input_kw = dict(example.input_kw or {})
        with record.recorder() as records:
            try:
                predicted_output = interface(*input_args, **input_kw)
            except Exception as ex:
                predicted_output = f'**exception raised**: {ex}'
        llm_usage_stats = self.aggregate_usage_stats(records)
        metrics = self.compare_predictions(predicted_output, example.expected_output)
        return dict(
            predicted_output=predicted_output,
            expected_output=example.expected_output,
            **metrics,
            **llm_usage_stats,
        )

    def evaluate(self, dataset: Dataset, interface) -> Path:
        expt_name = config.get('evaluate.expt_name')
        result_dir = config.require('evaluate.result_dir')
        csv_path, jsonl_path = savefile.filename_list(
            result_dir, ['results.csv', 'results.jsonl'], file_under=expt_name)
        self.artifacts_dir = csv_path.parent / 'artifacts'
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        results = []
        with open(jsonl_path, 'w') as fp:
            for row in tqdm(self.measurements(dataset, interface)):
                row.update(expt_name=expt_name)
                fp.write(json.dumps(row) + '\n')
                csv_row = dict(row)
                if 'predicted_output' in csv_row:
                    csv_row['predicted_output_path'] = csv_row.get('code_path')
                    del csv_row['predicted_output']
                results.append(csv_row)

        df = pd.DataFrame(results).set_index('case_name')
        df.to_csv(csv_path)
        print(f'saved in {csv_path}')
        return csv_path


app = typer.Typer()


@app.callback()
def callback():
    """DesignBench benchmark."""


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(
    ctx: typer.Context,
    config_file: str = typer.Option(..., help="Config YAML file"),
):
    """Run DesignBench evaluation. Extra args are config overrides in dot notation."""
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = _BENCHMARK_DIR / cfg_path

    config.configure(yaml_file=str(cfg_path), dotlist=ctx.args)
    config.set_root(_BENCHMARK_DIR)
    _configure_designbench_imports()

    framework = config.require('dataset.framework')
    _set_designbench_generate_prompt(framework)
    implement_via_config(ptools, config.require('ptools'))
    dataset = load_dataset(
        framework=framework,
        max_reference_chars=config.get('dataset.max_reference_chars'),
    )
    dataset = dataset.configure(
        n=config.get('dataset.n'),
    )
    print('dataset:', dataset.summary())
    entry_point = config.require('evaluate.entry_point')
    interface = getattr(ptools, entry_point)

    evaluator = DesignBenchEvaluator(
        output_framework=config.get('benchmark.output_framework') or framework,
        skip_eval=bool(config.get('benchmark.skip_eval')),
    )
    csv_path = evaluator.evaluate(dataset, interface)

    df = pd.read_csv(csv_path)
    if 'clip_similarity' in df.columns:
        scored = df['clip_similarity'].notna()
        n_scored = int(scored.sum())
        if n_scored > 0:
            print(f"Avg CLIP: {df.loc[scored, 'clip_similarity'].mean():.4f}")
            for _, row in df.loc[scored, ['case_name', 'clip_similarity']].iterrows():
                print(f"CLIP {row['case_name']}: {float(row['clip_similarity']):.4f}")
            print(f"Evaluated: {n_scored}/{len(df)}")
        else:
            print('CLIP unavailable: no scored rows in results.')
    else:
        print('CLIP unavailable: clip_similarity missing from results.')
    if 'eval_error' in df.columns:
        errors = [e for e in df['eval_error'].dropna().unique() if str(e).strip()]
        for err in errors:
            print(f"eval_error: {err}")


if __name__ == '__main__':
    app()
