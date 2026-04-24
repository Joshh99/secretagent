"""Orchestration Learner: supervisor-driven pipeline hill climbing.

A supervisor LLM iteratively improves a pipeline by analyzing failures,
proposing code changes, and evaluating them. Hill climbs on accuracy,
rolls back regressions, tracks train/eval curves.

Usage (from benchmark directory, e.g. benchmarks/medcalc/):

    uv run python -m secretagent.cli.orchestration_learner \
        --config-file conf/workflow.yaml \
        --n-train 110 --n-eval 110 \
        --max-iterations 10

    # With custom instructions and model switching:
    uv run python -m secretagent.cli.orchestration_learner \
        --config-file conf/workflow.yaml \
        --custom-instructions "Focus on scoring system calculators" \
        --model-change models.json \
        --supervisor-model gemini/gemini-3.1-pro-preview

    # Generate HTML report for an existing run:
    uv run python -m secretagent.cli.orchestration_learner view results/TIMESTAMP.orch_learner
"""

import json
import os
import sys
import inspect
import random
import shutil
from datetime import datetime
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

import typer

_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}
app = typer.Typer(pretty_exceptions_enable=False)


def _load_custom_instructions(value: str) -> str:
    """Load custom instructions from text or @filepath."""
    if not value:
        return ''
    if value.startswith('@'):
        path = Path(value[1:])
        if path.exists():
            return path.read_text()
        print(f'Warning: instruction file {path} not found, using as text')
    return value


def _load_model_choices(path_str: str) -> str:
    """Load model choices from JSON and format as table."""
    if not path_str:
        return ''
    path = Path(path_str)
    if not path.exists():
        print(f'Warning: model file {path} not found')
        return ''
    models = json.loads(path.read_text())
    lines = ['| Model | Input $/1M | Output $/1M |',
             '|-------|-----------|------------|']
    for m in models:
        name = m.get('name', m.get('model', '?'))
        cin = m.get('input_cost', m.get('cost_in', '?'))
        cout = m.get('output_cost', m.get('cost_out', '?'))
        lines.append(f'| {name} | ${cin} | ${cout} |')
    return '\n'.join(lines)


def _infer_ptools_module_name(
    benchmark_dir: Path,
    split: str,
    requested: str = '',
) -> str:
    """Infer the ptools module for benchmarks with per-task modules."""
    if requested:
        return requested
    if (benchmark_dir / 'ptools.py').exists():
        return 'ptools'
    if benchmark_dir.name == 'musr':
        if split.startswith('murder_mysteries'):
            return 'ptools_murder'
        if split.startswith('object_placements'):
            return 'ptools_object'
        if split.startswith('team_allocation'):
            return 'ptools_team'
    if benchmark_dir.name == 'natural_plan':
        for task in ('calendar', 'meeting', 'trip'):
            if split == task or split.startswith(f'{task}_'):
                return f'ptools_{task}'
    raise FileNotFoundError(
        f'Could not infer ptools module in {benchmark_dir}. '
        'Pass --ptools-module explicitly.'
    )


def _canonical_task_name(benchmark_dir: Path, split: str) -> str:
    """Map split aliases like meeting_train to evaluator task names."""
    if benchmark_dir.name == 'natural_plan':
        for task in ('calendar', 'meeting', 'trip'):
            if split == task or split.startswith(f'{task}_'):
                return task
    return split


def _copy_seed_resources(benchmark_dir: Path, scratch_dir: Path) -> None:
    """Copy small module-local resources needed by scratch ptools modules."""
    for dirname in ('prompt_templates',):
        src = benchmark_dir / dirname
        dst = scratch_dir / dirname
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)


def _module_interfaces(module):
    """Return Interface objects exposed by a module, avoiding global registry drift."""
    from secretagent.core import Interface

    interfaces = []
    seen = set()
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Interface) and id(obj) not in seen:
            interfaces.append(obj)
            seen.add(id(obj))
    return interfaces


def _dataset_from_cases(dataset, cases: list, suffix: str):
    """Return a dataset copy with a replacement case list."""
    return dataset.model_copy(update={
        'name': f'{dataset.name}{suffix}',
        'cases': list(cases),
    })


def _load_dataset_for_split(load_dataset, split: str):
    """Call benchmark load_dataset with supported dataset.* config keys."""
    from secretagent import config

    sig = inspect.signature(load_dataset)
    kwargs = {}
    actual_split = split
    cfg_map = {
        'prompt_mode': 'dataset.prompt_mode',
        'stratified': 'dataset.stratified',
        'sample_n': 'dataset.sample_n',
        'sample_seed': 'dataset.sample_seed',
        'partition': 'dataset.partition',
    }
    for param, cfg_key in cfg_map.items():
        if param in sig.parameters:
            value = config.get(cfg_key)
            if value is not None:
                kwargs[param] = value

    # NaturalPlan load_dataset(task, partition=...) uses task names
    # (meeting/trip/calendar) rather than split filenames. Accept convenient
    # split aliases like meeting_train and trip_valid in the generic CLI.
    if 'partition' in sig.parameters:
        for task in ('calendar', 'meeting', 'trip'):
            prefix = f'{task}_'
            if split.startswith(prefix):
                actual_split = task
                kwargs['partition'] = split[len(prefix):]
                break

    return load_dataset(actual_split, **kwargs)


def _plain_sample(cases: list, n: int, seed: int | None) -> list:
    """Deterministically sample cases without replacement."""
    sampled = list(cases)
    if seed is not None:
        random.Random(seed).shuffle(sampled)
    return sampled[:min(n, len(sampled))]


def _case_stratified_sample_fn(evaluator_module):
    """Return stratified_sample if it accepts (cases, n, seed)."""
    sampler = getattr(evaluator_module, 'stratified_sample', None)
    if sampler is None:
        return None
    params = list(inspect.signature(sampler).parameters)
    if len(params) >= 2 and params[1] == 'n':
        return sampler
    return None


def _resolve_train_eval_datasets(
    evaluator_module,
    load_dataset,
    train_split: str,
    eval_split: str,
    n_train: int,
    n_eval: int,
    seed: int | None,
):
    """Load train/eval datasets, keeping same-split samples disjoint."""
    full_dataset = _load_dataset_for_split(load_dataset, train_split)
    case_stratified_sample = _case_stratified_sample_fn(evaluator_module)

    if n_eval > 0 and hasattr(evaluator_module, 'stratified_split'):
        from expt import stratified_split
        train_cases, eval_cases = stratified_split(
            full_dataset.cases, n_train, n_eval, seed=seed or 42)
        train_dataset = _dataset_from_cases(full_dataset, train_cases, '_train')
        eval_dataset = _dataset_from_cases(full_dataset, eval_cases, '_eval')
        return train_dataset, eval_dataset, f'disjoint stratified split from {train_split}'

    if n_eval > 0 and train_split == eval_split:
        cases = list(full_dataset.cases)
        if seed is not None:
            random.Random(seed).shuffle(cases)
        train_cases = cases[:min(n_train, len(cases))]
        eval_cases = cases[len(train_cases):len(train_cases) + n_eval]
        train_dataset = _dataset_from_cases(full_dataset, train_cases, '_train')
        eval_dataset = _dataset_from_cases(full_dataset, eval_cases, '_eval')
        return train_dataset, eval_dataset, f'disjoint plain split from {train_split}'

    if case_stratified_sample is not None:
        train_cases = case_stratified_sample(
            full_dataset.cases, n_train, seed=seed or 42)
        train_dataset = _dataset_from_cases(full_dataset, train_cases, '_train')
    else:
        train_dataset = _dataset_from_cases(
            full_dataset, _plain_sample(full_dataset.cases, n_train, seed), '_train')

    eval_dataset = None
    if n_eval > 0:
        eval_full = _load_dataset_for_split(load_dataset, eval_split)
        if case_stratified_sample is not None:
            eval_cases = case_stratified_sample(
                eval_full.cases, n_eval, seed=seed or 42)
        else:
            eval_cases = _plain_sample(eval_full.cases, n_eval, seed)
        eval_dataset = _dataset_from_cases(eval_full, eval_cases, '_eval')

    return train_dataset, eval_dataset, 'separate splits'


def _seed_task_description(tools_cfg, entry_point_name: str,
                           override: str, entry_interface) -> str:
    """Choose task description for first workflow generation."""
    if override:
        return _load_custom_instructions(override)
    entry_cfg = tools_cfg.get(entry_point_name) if hasattr(tools_cfg, 'get') else None
    if entry_cfg and entry_cfg.get('task_description'):
        return str(entry_cfg.get('task_description'))
    return (entry_interface.doc or '').strip()


def _rename_def_signature(signature: str, new_name: str) -> str:
    """Rename a one-line Python def signature."""
    return 'def ' + new_name + '(' + signature.split('(', 1)[1]


def _seed_orchestrated_workflow(
    *,
    evolved_path: Path,
    ptools_module,
    ptools_module_name: str,
    tools_cfg,
    entry_point_name: str,
    task_description_override: str,
    model: str,
) -> str:
    """Generate an initial workflow from ptools and append it to the module."""
    from secretagent import config
    from secretagent.core import implement_via_config
    from secretagent.orchestrate.catalog import PtoolCatalog
    from secretagent.orchestrate.composer import compose
    from secretagent.orchestrate.pipeline import (
        Pipeline, _entry_signature_from_interface,
    )

    entry_interface = getattr(ptools_module, entry_point_name)
    tools_only_cfg = {
        name: cfg for name, cfg in tools_cfg.items()
        if name != entry_point_name
    }
    allowed_tool_names = set(tools_only_cfg)
    implement_via_config(ptools_module, tools_only_cfg)

    tool_interfaces = [
        iface for iface in _module_interfaces(ptools_module)
        if iface.name in allowed_tool_names and iface.implementation is not None
    ]
    catalog = PtoolCatalog.from_interfaces(tool_interfaces)
    if not catalog.ptools:
        raise ValueError(
            f'No implemented tools available to seed {entry_point_name}. '
            'Check the ptools config or pass a config with non-entry tools.'
        )

    entry_signature = _entry_signature_from_interface(entry_interface)
    task_description = _seed_task_description(
        tools_cfg, entry_point_name, task_description_override, entry_interface)

    print(f'[seed] composing initial workflow for {entry_point_name} '
          f'from {len(catalog)} tools with {model}')
    code = compose(task_description, catalog, entry_signature, model=model)

    # Compile-check before writing it into the ptools file.
    namespace = {iface.name: iface for iface in tool_interfaces}
    Pipeline(code, entry_signature, namespace)

    seed_fn_name = f'{entry_point_name}_orchestrated_seed'
    seed_signature = _rename_def_signature(entry_signature, seed_fn_name)
    seed_pipeline = Pipeline(code, seed_signature, namespace)
    seed_source = (
        '\n\n# --- Auto-generated by orchestration_learner --seed-orchestrate ---\n'
        f'{seed_pipeline.source}\n'
    )
    evolved_path.write_text(evolved_path.read_text().rstrip() + seed_source)
    config.configure(dotlist=[
        f'ptools.{entry_point_name}.method=direct',
        f'ptools.{entry_point_name}.fn={ptools_module_name}.{seed_fn_name}',
    ])
    print(f'[seed] wrote {seed_fn_name} to {evolved_path}')
    return seed_fn_name


def _load_local_json_dataset(benchmark_dir: Path, split: str):
    """Fallback loader for benchmarks storing Dataset JSON in data/{split}.json."""
    from secretagent.dataset import Dataset

    dataset_path = benchmark_dir / 'data' / f'{split}.json'
    if not dataset_path.exists():
        raise FileNotFoundError(
            f'No load_dataset() in expt.py and no local dataset at {dataset_path}'
        )
    return Dataset.model_validate_json(dataset_path.read_text())


def _find_evaluator_cls(evaluator_module, benchmark_dir: Path):
    """Find a benchmark-specific Evaluator class from expt.py or evaluator.py."""
    from secretagent.evaluate import Evaluator

    modules = [evaluator_module]
    try:
        import importlib
        modules.append(importlib.import_module('evaluator'))
    except ImportError:
        pass

    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Evaluator) and obj is not Evaluator:
                return obj

    from secretagent.evaluate import ExactMatchEvaluator
    return ExactMatchEvaluator


def _generate_html_report(report, output_dir: Path):
    """Generate a self-contained HTML report with full iteration visibility."""
    import html as html_mod

    iter_dir = output_dir / 'iterations'
    iterations_data = []

    for rec in report.iterations:
        iter_name = f'iter_{rec.iteration:03d}' if rec.iteration > 0 else 'iter_000_baseline'
        idir = iter_dir / iter_name
        entry = {
            'iteration': rec.iteration,
            'train_accuracy': rec.train_accuracy,
            'eval_accuracy': rec.eval_accuracy,
            'train_cost': rec.train_cost,
            'supervisor_cost': rec.supervisor_cost,
            'kept': rec.kept,
            'reasoning': rec.reasoning or '',
            'config_overrides': rec.config_overrides,
        }
        # Load artifacts if they exist
        for fname in ('profiling_summary.txt', 'failure_traces.txt',
                      'supervisor_prompt.txt', 'supervisor_response.txt',
                      'outcome.txt', 'iteration_history.txt'):
            fpath = idir / fname
            if fpath.exists():
                entry[fname.replace('.txt', '')] = fpath.read_text()

        # Compute concise diff
        before = idir / 'ptools_before.py'
        after = idir / 'ptools_after.py'
        if before.exists() and after.exists():
            import difflib
            b_lines = before.read_text().splitlines(keepends=True)
            a_lines = after.read_text().splitlines(keepends=True)
            diff = list(difflib.unified_diff(b_lines, a_lines,
                                             fromfile='before', tofile='after',
                                             n=3))
            entry['diff'] = ''.join(diff) if diff else '(no changes)'
        iterations_data.append(entry)

    # Build JSON data for the page
    _page_data = {
        'best_train_accuracy': report.best_train_accuracy,
        'best_iteration': report.best_iteration,
        'final_eval_accuracy': report.final_eval_accuracy,
        'total_supervisor_cost': report.total_supervisor_cost,
        'iterations': iterations_data,
    }

    def esc(s):
        return html_mod.escape(str(s)) if s else ''

    # --- Build HTML ---
    # Chart data
    iters_json = json.dumps(iterations_data, default=str)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Orchestration Learner Report — {output_dir.name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; line-height: 1.5; }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; }}
  h2 {{ color: #58a6ff; margin: 24px 0 12px; font-size: 1.2em; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px; margin: 16px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; text-align: center; }}
  .card .value {{ font-size: 2em; font-weight: bold; color: #58a6ff; }}
  .card .label {{ font-size: 0.85em; color: #8b949e; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                      padding: 16px; margin: 16px 0; }}
  canvas {{ width: 100% !important; height: 300px !important; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }}
  th {{ background: #161b22; color: #58a6ff; font-size: 0.85em; }}
  tr:hover {{ background: #161b22; }}
  .kept {{ color: #3fb950; font-weight: bold; }}
  .rollback {{ color: #f85149; }}
  .baseline {{ color: #8b949e; }}
  .iter-detail {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                  margin: 12px 0; overflow: hidden; }}
  .iter-header {{ padding: 12px 16px; cursor: pointer; display: flex;
                  justify-content: space-between; align-items: center;
                  background: #161b22; border-bottom: 1px solid #21262d; }}
  .iter-header:hover {{ background: #1c2128; }}
  .iter-header .arrow {{ transition: transform 0.2s; color: #8b949e; }}
  .iter-header.open .arrow {{ transform: rotate(90deg); }}
  .iter-body {{ display: none; padding: 16px; }}
  .iter-body.open {{ display: block; }}
  .section {{ margin: 12px 0; }}
  .section-title {{ font-weight: bold; color: #58a6ff; margin-bottom: 4px;
                    cursor: pointer; user-select: none; }}
  .section-title:hover {{ text-decoration: underline; }}
  .section-content {{ display: none; background: #0d1117; border: 1px solid #21262d;
                      border-radius: 4px; padding: 12px; margin-top: 4px;
                      max-height: 600px; overflow: auto; }}
  .section-content.open {{ display: block; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em;
         font-family: "Fira Code", "Cascadia Code", monospace; }}
  .diff-add {{ color: #3fb950; }}
  .diff-del {{ color: #f85149; }}
  .diff-hdr {{ color: #58a6ff; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
          font-size: 0.75em; font-weight: bold; }}
  .tag-kept {{ background: #0f2d1a; color: #3fb950; border: 1px solid #238636; }}
  .tag-roll {{ background: #2d1117; color: #f85149; border: 1px solid #da3633; }}
  .tag-base {{ background: #1c1e23; color: #8b949e; border: 1px solid #30363d; }}
  .acc-bar {{ height: 6px; border-radius: 3px; margin-top: 4px; }}
</style>
</head>
<body>

<h1>🎛️ Orchestration Learner Report</h1>
<p style="color:#8b949e">{esc(output_dir.name)}</p>

<div class="summary">
  <div class="card">
    <div class="value">{report.best_train_accuracy:.1%}</div>
    <div class="label">Best Train Accuracy (iter {report.best_iteration})</div>
  </div>
  <div class="card">
    <div class="value">{f"{report.final_eval_accuracy:.1%}" if report.final_eval_accuracy is not None else "—"}</div>
    <div class="label">Final Eval Accuracy</div>
  </div>
  <div class="card">
    <div class="value">{len(report.iterations)}</div>
    <div class="label">Iterations</div>
  </div>
  <div class="card">
    <div class="value">${report.total_supervisor_cost:.2f}</div>
    <div class="label">Supervisor Cost</div>
  </div>
</div>

<h2>Accuracy Curve</h2>
<div class="chart-container">
  <canvas id="accChart"></canvas>
</div>

<h2>Iteration Log</h2>
<table>
  <thead>
    <tr><th>Iter</th><th>Train</th><th>Fail</th><th>TO</th><th>Eval</th><th>Cost/case</th><th>Sup $</th><th>Status</th></tr>
  </thead>
  <tbody id="iterTable"></tbody>
</table>

<h2>Iteration Details</h2>
<div id="iterDetails"></div>

<script>
const DATA = {iters_json};

// --- Populate table ---
const tbody = document.getElementById('iterTable');
DATA.forEach(d => {{
  const status = d.kept ? (d.iteration === 0 ? 'BASELINE' : 'KEPT') : 'ROLLBACK';
  const cls = d.kept ? (d.iteration === 0 ? 'baseline' : 'kept') : 'rollback';
  const evalStr = d.eval_accuracy !== null ? (d.eval_accuracy * 100).toFixed(1) + '%' : '—';
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{d.iteration}}</td>
    <td>${{(d.train_accuracy * 100).toFixed(1)}}%</td>
    <td>${{d.train_failures || 0}}</td>
    <td>${{d.train_timeouts || 0}}</td>
    <td>${{evalStr}}</td>
    <td>${{d.train_cost ? '$' + d.train_cost.toFixed(4) : '—'}}</td>
    <td>${{d.supervisor_cost ? '$' + d.supervisor_cost.toFixed(4) : '—'}}</td>
    <td class="${{cls}}">${{status}}</td>`;
  tbody.appendChild(tr);
}});

// --- Populate details ---
const details = document.getElementById('iterDetails');
function makeSection(title, content, startOpen) {{
  if (!content || content === '(no changes)') return '';
  const id = 'sec_' + Math.random().toString(36).substr(2);
  const openCls = startOpen ? 'open' : '';
  // Colorize diffs
  let escaped = content.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  if (title.includes('Diff')) {{
    escaped = escaped.split('\\n').map(line => {{
      if (line.startsWith('+') && !line.startsWith('+++')) return `<span class="diff-add">${{line}}</span>`;
      if (line.startsWith('-') && !line.startsWith('---')) return `<span class="diff-del">${{line}}</span>`;
      if (line.startsWith('@@')) return `<span class="diff-hdr">${{line}}</span>`;
      return line;
    }}).join('\\n');
  }}
  return `<div class="section">
    <div class="section-title" onclick="document.getElementById('${{id}}').classList.toggle('open');
      this.textContent = this.textContent.startsWith('▸') ?
        '▾' + this.textContent.slice(1) : '▸' + this.textContent.slice(1);">
      ${{startOpen ? '▾' : '▸'}} ${{title}} (${{content.length > 1000 ? (content.length/1024).toFixed(0) + 'KB' : content.length + ' chars'}})
    </div>
    <div id="${{id}}" class="section-content ${{openCls}}"><pre>${{escaped}}</pre></div>
  </div>`;
}}

DATA.forEach(d => {{
  const status = d.kept ? (d.iteration === 0 ? 'BASELINE' : 'KEPT') : 'ROLLBACK';
  const tagCls = d.kept ? (d.iteration === 0 ? 'tag-base' : 'tag-kept') : 'tag-roll';
  const evalStr = d.eval_accuracy !== null ? ` | eval: ${{(d.eval_accuracy*100).toFixed(1)}}%` : '';
  const div = document.createElement('div');
  div.className = 'iter-detail';
  div.innerHTML = `
    <div class="iter-header" onclick="this.classList.toggle('open');
      this.nextElementSibling.classList.toggle('open');">
      <span><strong>Iteration ${{d.iteration}}</strong> — train: ${{(d.train_accuracy*100).toFixed(1)}}%${{evalStr}}
        <span class="tag ${{tagCls}}">${{status}}</span></span>
      <span class="arrow">▸</span>
    </div>
    <div class="iter-body">
      ${{d.reasoning ? '<div style="margin-bottom:12px"><strong>Reasoning:</strong><br>' +
        d.reasoning.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>') + '</div>' : ''}}
      ${{makeSection('Code Diff', d.diff, true)}}
      ${{makeSection('Profiling Summary', d.profiling_summary, false)}}
      ${{makeSection('Failure Traces', d.failure_traces, false)}}
      ${{makeSection('Iteration History (sent to supervisor)', d.iteration_history, false)}}
      ${{makeSection('Supervisor Prompt (full)', d.supervisor_prompt, false)}}
      ${{makeSection('Supervisor Response (full)', d.supervisor_response, false)}}
      ${{makeSection('Outcome', d.outcome, false)}}
    </div>`;
  details.appendChild(div);
}});

// --- Chart (simple canvas) ---
const canvas = document.getElementById('accChart');
const ctx = canvas.getContext('2d');
function drawChart() {{
  const W = canvas.width = canvas.offsetWidth;
  const H = canvas.height = 300;
  const pad = {{ t: 30, r: 20, b: 40, l: 55 }};
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, W, H);

  const n = DATA.length;
  if (n < 2) return;
  const allAcc = [...DATA.map(d => d.train_accuracy),
                   ...DATA.filter(d=>d.eval_accuracy!==null).map(d=>d.eval_accuracy)];
  const maxAcc = Math.min(1.0, Math.max(...allAcc) + 0.05);
  const minAcc = Math.max(0, Math.min(...allAcc) - 0.05);

  function x(i) {{ return pad.l + (i / (n - 1)) * cw; }}
  function y(v) {{ return pad.t + (1 - (v - minAcc) / (maxAcc - minAcc)) * ch; }}

  // Grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  const step = (maxAcc - minAcc) > 0.3 ? 0.1 : 0.05;
  for (let v = Math.ceil(minAcc/step)*step; v <= maxAcc; v += step) {{
    ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(W-pad.r, y(v)); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((v*100).toFixed(0)+'%', pad.l-8, y(v)+4);
  }}
  // X axis labels
  ctx.textAlign = 'center';
  DATA.forEach((d,i) => {{ ctx.fillText(d.iteration, x(i), H-pad.b+18); }});
  ctx.fillText('Iteration', W/2, H-5);

  // Train line
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2;
  ctx.beginPath();
  DATA.forEach((d,i) => {{ i === 0 ? ctx.moveTo(x(i), y(d.train_accuracy)) : ctx.lineTo(x(i), y(d.train_accuracy)); }});
  ctx.stroke();

  // Eval line
  const evalData = DATA.filter(d => d.eval_accuracy !== null);
  if (evalData.length > 1) {{
    ctx.strokeStyle = '#f0883e'; ctx.lineWidth = 2; ctx.setLineDash([5,5]);
    ctx.beginPath();
    evalData.forEach((d,i) => {{
      const xi = x(DATA.indexOf(d));
      i === 0 ? ctx.moveTo(xi, y(d.eval_accuracy)) : ctx.lineTo(xi, y(d.eval_accuracy));
    }});
    ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Points
  DATA.forEach((d,i) => {{
    ctx.fillStyle = d.kept ? '#3fb950' : '#f85149';
    ctx.beginPath(); ctx.arc(x(i), y(d.train_accuracy), 5, 0, Math.PI*2); ctx.fill();
    if (d.eval_accuracy !== null) {{
      ctx.fillStyle = '#f0883e';
      ctx.beginPath(); ctx.arc(x(DATA.indexOf(d)), y(d.eval_accuracy), 4, 0, Math.PI*2); ctx.fill();
    }}
  }});

  // Legend
  ctx.font = '12px sans-serif';
  const lx = pad.l + 10, ly = pad.t + 10;
  ctx.fillStyle = '#58a6ff'; ctx.fillRect(lx, ly, 16, 3); ctx.fillText('Train', lx+22, ly+5);
  if (evalData.length > 0) {{
    ctx.fillStyle = '#f0883e'; ctx.fillRect(lx, ly+16, 16, 3); ctx.fillText('Eval', lx+22, ly+21);
  }}
  ctx.fillStyle = '#3fb950'; ctx.beginPath(); ctx.arc(lx+100, ly+3, 4, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#c9d1d9'; ctx.fillText('Kept', lx+110, ly+5);
  ctx.fillStyle = '#f85149'; ctx.beginPath(); ctx.arc(lx+150, ly+3, 4, 0, Math.PI*2); ctx.fill();
  ctx.fillText('Rollback', lx+160, ly+5);
}}
drawChart();
window.addEventListener('resize', drawChart);
</script>

</body>
</html>'''

    report_path = output_dir / 'report.html'
    report_path.write_text(html_content)
    print(f'[report] saved {report_path}')


def _generate_plots(report, output_dir: Path):
    """Generate accuracy/cost plots from iteration data."""
    try:
        # /// script
        # requires-python = ">=3.11"
        # dependencies = ["matplotlib"]
        # ///
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plots] matplotlib not available, skipping plots')
        return

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    iters = report.iterations
    xs = [r.iteration for r in iters]
    train_acc = [r.train_accuracy for r in iters]
    train_cost = [r.train_cost for r in iters]
    kept = [r.kept for r in iters]

    # --- Accuracy over iterations (train + eval) ---
    eval_acc = [r.eval_accuracy for r in iters]
    has_eval = any(e is not None for e in eval_acc)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, train_acc, 'b-o', label='Train accuracy', markersize=4)

    if has_eval:
        eval_xs = [x for x, e in zip(xs, eval_acc) if e is not None]
        eval_ys = [e for e in eval_acc if e is not None]
        ax.plot(eval_xs, eval_ys, 'r-s', label='Eval accuracy', markersize=4)

    # Mark kept vs rolled back
    for x, acc, k in zip(xs, train_acc, kept):
        if k:
            ax.plot(x, acc, 'go', markersize=8, zorder=5)
        elif x > 0:
            ax.plot(x, acc, 'rx', markersize=8, zorder=5)

    # Best accuracy line
    best_acc = max(train_acc)
    ax.axhline(y=best_acc, color='green', linestyle='--', alpha=0.5,
               label=f'Best train: {best_acc:.1%}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Orchestration Learner: Train & Eval Accuracy over Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'accuracy_over_iterations.png', dpi=150)
    plt.close(fig)
    print(f'[plots] saved {plots_dir / "accuracy_over_iterations.png"}')

    # --- Cost over iterations ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(xs, train_cost, 'r-o', label='Avg cost/case', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Avg cost per case ($)')
    ax1.set_title('Orchestration Learner: Cost over Iterations')

    # Cumulative supervisor cost
    sup_costs = [r.supervisor_cost for r in iters]
    cum_sup = []
    total = 0.0
    for sc in sup_costs:
        total += sc
        cum_sup.append(total)
    ax2 = ax1.twinx()
    ax2.plot(xs, cum_sup, 'b--', label='Cumulative supervisor cost', alpha=0.7)
    ax2.set_ylabel('Cumulative supervisor cost ($)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'cost_over_iterations.png', dpi=150)
    plt.close(fig)
    print(f'[plots] saved {plots_dir / "cost_over_iterations.png"}')

    # --- Accuracy vs Cost scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in iters:
        color = 'green' if r.kept else 'red'
        marker = 'o' if r.kept else 'x'
        ax.plot(r.train_cost, r.train_accuracy, marker=marker, color=color,
                markersize=8)
        ax.annotate(str(r.iteration), (r.train_cost, r.train_accuracy),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Avg cost per case ($)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Orchestration Learner: Accuracy vs Cost')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'accuracy_vs_cost.png', dpi=150)
    plt.close(fig)
    print(f'[plots] saved {plots_dir / "accuracy_vs_cost.png"}')


def _on_iteration(output_dir: Path):
    """Regenerate HTML report after each iteration (auto-refresh in browser)."""
    from secretagent.orchestrate.improve import SupervisorReport
    report_path = output_dir / 'report.json'
    if report_path.exists():
        try:
            report = SupervisorReport.model_validate_json(report_path.read_text())
            _generate_html_report(report, output_dir)
        except Exception:
            pass


@app.command('run', context_settings=_EXTRA_ARGS)
def run(
    ctx: typer.Context,
    config_file: str = typer.Option('', help='Starting config YAML'),
    benchmark: str = typer.Option(
        '', help='Benchmark name from orchestrate/benchmarks.yaml',
    ),
    n_train: int = typer.Option(110, help='Training set size'),
    n_eval: int = typer.Option(110, help='Eval set size'),
    max_iterations: int = typer.Option(10, help='Max improvement iterations'),
    target_accuracy: float = typer.Option(None, help='Stop when reached'),
    supervisor_model: str = typer.Option(
        'gemini/gemini-3.1-pro-preview', help='Supervisor LLM model',
    ),
    custom_instructions: str = typer.Option(
        '', help='Extra instructions (text or @filepath)',
    ),
    model_change: str = typer.Option(
        '', help='JSON file with model choices for supervisor',
    ),
    train_split: str = typer.Option('', help='Dataset split for training'),
    eval_split: str = typer.Option('', help='Dataset split for evaluation'),
    ptools_module: str = typer.Option(
        '', help='Ptools module name, e.g. ptools_murder or ptools_meeting',
    ),
    seed_orchestrate: bool = typer.Option(
        False,
        help='Generate an initial workflow from ptools before supervisor iteration',
    ),
    orchestrate_task_description: str = typer.Option(
        '',
        help='Task description text, or @file, for --seed-orchestrate',
    ),
    debug: bool = typer.Option(False, help='Full transparency: echo supervisor I/O'),
    resume: str = typer.Option('', help='Resume from a previous .orch_learner run directory'),
):
    """Run supervisor-driven pipeline improvement on a benchmark.

    Iteratively improves a pipeline by calling a supervisor LLM that sees
    profiling data and failure traces. Hill climbs on accuracy, rolls back
    regressions. Reports best config on a held-out eval set.

    Run from a benchmark directory (e.g. benchmarks/medcalc/).
    Extra args are dotlist config overrides.
    """
    benchmark_dir = Path.cwd()

    # Add cwd and project src/ to path so we can import secretagent first.
    sys.path.insert(0, str(benchmark_dir))
    project_root = benchmark_dir
    while project_root != project_root.parent:
        if (project_root / 'src').exists():
            sys.path.insert(0, str(project_root / 'src'))
            break
        project_root = project_root.parent

    from secretagent import config
    from secretagent.core import implement_via_config
    from secretagent.orchestrate.catalog import PtoolCatalog
    from secretagent.orchestrate.improve import improve_with_supervisor

    adapter = None
    adapter_train_split = None
    adapter_eval_split = None
    if benchmark:
        from secretagent.orchestrate.benchmark_adapter import BenchmarkAdapter
        adapter = BenchmarkAdapter(benchmark)
        adapter.setup_sys_path()
        benchmark_dir = adapter.benchmark_dir
        if str(benchmark_dir) not in sys.path:
            sys.path.insert(0, str(benchmark_dir))
        os.chdir(benchmark_dir)
        if not config_file:
            config_file = str(adapter.config_file)
        if not ptools_module:
            ptools_module = adapter.spec['ptools_module']
        adapter_train_split = adapter.spec.get('train', {}).get('split')
        adapter_eval_split = adapter.spec.get('eval', {}).get('split')

    if not config_file:
        print('Error: --config-file is required unless --benchmark is provided')
        raise typer.Exit(1)

    timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')

    # --- Load config ---
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = benchmark_dir / cfg_path
    config.configure(yaml_file=str(cfg_path), dotlist=ctx.args)
    config.set_root(benchmark_dir)

    if adapter is not None:
        adapter_overrides = []
        extra_keys = [arg.split('=', 1)[0] for arg in ctx.args if '=' in arg]
        if 'evaluate.entry_point' not in extra_keys:
            adapter_overrides.append(
                f'evaluate.entry_point={adapter.entry_point_name}')
        if 'dataset.shuffle_seed' not in extra_keys:
            adapter_overrides.append(
                f'dataset.shuffle_seed={adapter.shuffle_seed}')
        agent_model = adapter.spec.get('agent_model')
        if agent_model and 'llm.model' not in extra_keys:
            adapter_overrides.append(f'llm.model={agent_model}')
        if adapter_overrides:
            config.configure(dotlist=adapter_overrides)

    # Enable caching by default
    if config.get('cachier.enable_caching') is None:
        config.configure(cachier=dict(enable_caching=True))

    # Debug mode: full transparency into supervisor I/O
    if debug:
        config.configure(echo=dict(orchestrate_llm=True))

    configured_split = config.get('dataset.split')
    train_split = train_split or adapter_train_split or configured_split or 'train'
    eval_split = eval_split or adapter_eval_split or (
        'test' if train_split == 'train' else train_split)

    ptools_module_name = _infer_ptools_module_name(
        benchmark_dir, train_split, ptools_module)

    # --- Create/load evolved ptools module ---
    base_path = benchmark_dir / f'{ptools_module_name}.py'
    if not base_path.exists():
        print(f'Error: base ptools module not found: {base_path}')
        raise typer.Exit(1)

    if seed_orchestrate and not resume:
        scratch_dir = benchmark_dir / '.orchestration_learner'
        scratch_dir.mkdir(exist_ok=True)
        _copy_seed_resources(benchmark_dir, scratch_dir)
        evolved_path = scratch_dir / f'{ptools_module_name}_{timestamp}_seed.py'
        shutil.copy2(base_path, evolved_path)
        print(f'Created seed module {evolved_path.relative_to(benchmark_dir)} '
              f'from {base_path.name}')
    else:
        evolved_path = benchmark_dir / f'{ptools_module_name}_evolved.py'
        if not evolved_path.exists():
            shutil.copy2(base_path, evolved_path)
            print(f'Created {evolved_path.name} from {base_path.name}')
        else:
            print(f'Using existing {evolved_path.name}')

    # Import evolved source under the original module name so dotted config
    # paths like ptools.workflow or ptools_meeting.meeting_workflow resolve
    # to the evolving module rather than the immutable base file.
    import importlib.util
    spec = importlib.util.spec_from_file_location(ptools_module_name, str(evolved_path))
    ptools_module_obj = importlib.util.module_from_spec(spec)
    sys.modules[ptools_module_name] = ptools_module_obj
    spec.loader.exec_module(ptools_module_obj)

    if adapter is not None:
        evaluator_module = None
        load_dataset = None
        evaluator = adapter.get_evaluator()
    else:
        # Try to import benchmark-specific evaluator and dataset loader
        try:
            evaluator_module = __import__('expt')
        except ImportError as e:
            print(f'Error: could not import expt module from {benchmark_dir}: {e}')
            raise typer.Exit(1)
        load_dataset = getattr(evaluator_module, 'load_dataset', None)
        if load_dataset is None:
            load_dataset = lambda split: _load_local_json_dataset(benchmark_dir, split)

        # Get evaluator class
        evaluator_cls = _find_evaluator_cls(evaluator_module, benchmark_dir)

        evaluator_sig = inspect.signature(evaluator_cls)
        if 'task' in evaluator_sig.parameters:
            evaluator = evaluator_cls(_canonical_task_name(benchmark_dir, train_split))
        else:
            evaluator = evaluator_cls()

    entry_point_name = config.get('evaluate.entry_point', 'calculate_medical_value')
    if not hasattr(ptools_module_obj, entry_point_name):
        print(f'Error: entry point {entry_point_name!r} not found in '
              f'{ptools_module_name}')
        raise typer.Exit(1)

    if seed_orchestrate and not resume:
        seed_model = config.get('orchestrate.model') or supervisor_model
        _seed_orchestrated_workflow(
            evolved_path=evolved_path,
            ptools_module=ptools_module_obj,
            ptools_module_name=ptools_module_name,
            tools_cfg=config.require('ptools'),
            entry_point_name=entry_point_name,
            task_description_override=orchestrate_task_description,
            model=seed_model,
        )
        spec = importlib.util.spec_from_file_location(
            ptools_module_name, str(evolved_path))
        spec.loader.exec_module(ptools_module_obj)

    # --- Bind interfaces ---
    implement_via_config(ptools_module_obj, config.require('ptools'))
    if adapter is not None:
        adapter.run_setup_hook(ptools_module_obj)
    entry_interface = getattr(ptools_module_obj, entry_point_name)

    # --- Load datasets (disjoint train/eval from same split) ---
    print('\n=== Loading datasets ===')
    seed = config.get('dataset.shuffle_seed', 42)
    if adapter is not None:
        train_dataset, eval_dataset = adapter.load_train_eval(n_train, n_eval)
        split_note = f'adapter:{benchmark}'
    else:
        train_dataset, eval_dataset, split_note = _resolve_train_eval_datasets(
            evaluator_module, load_dataset, train_split, eval_split,
            n_train, n_eval, seed,
        )
    print(f'Train: {len(train_dataset.cases)} cases')
    if eval_dataset:
        print(f'Eval: {len(eval_dataset.cases)} cases')
    print(f'Sampling: {split_note}')

    # --- Build catalog ---
    allowed_tool_names = set(config.require('ptools')) - {entry_point_name}
    tool_interfaces = [
        iface for iface in _module_interfaces(ptools_module_obj)
        if iface.name in allowed_tool_names and iface.implementation is not None
    ]
    catalog = PtoolCatalog.from_interfaces(
        tool_interfaces,
    )

    # --- Load optional args ---
    instructions = _load_custom_instructions(custom_instructions)
    model_choices_text = _load_model_choices(model_change)

    # --- Resume state (loaded before output_dir so we can log it) ---
    resume_iterations = None
    resume_best_accuracy = None
    resume_best_eval_accuracy = None
    resume_supervisor_cost = 0.0

    if resume:
        from secretagent.orchestrate.improve import SupervisorReport
        resume_dir = Path(resume)
        prev_report_path = resume_dir / 'report.json'
        if not prev_report_path.exists():
            print(f'Error: {prev_report_path} not found')
            raise typer.Exit(1)
        prev_report = SupervisorReport.model_validate_json(prev_report_path.read_text())
        resume_iterations = prev_report.iterations
        resume_best_accuracy = prev_report.best_train_accuracy
        # Best eval from final eval or from per-iteration evals
        resume_best_eval_accuracy = prev_report.final_eval_accuracy
        if resume_best_eval_accuracy is None:
            kept_evals = [r.eval_accuracy for r in prev_report.iterations
                          if r.kept and r.eval_accuracy is not None]
            if kept_evals:
                resume_best_eval_accuracy = max(kept_evals)
        resume_supervisor_cost = prev_report.total_supervisor_cost

        # Copy best ptools_evolved.py from previous run as starting point
        prev_evolved = resume_dir / 'ptools_evolved.py'
        if prev_evolved.exists():
            evolved_path.write_text(prev_evolved.read_text())
            # Reload the module so it picks up the resumed code
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                ptools_module_name, str(evolved_path))
            spec.loader.exec_module(ptools_module_obj)
            implement_via_config(ptools_module_obj, config.require('ptools'))
            entry_interface = getattr(ptools_module_obj, entry_point_name)
            print(f'Loaded ptools_evolved.py from {resume_dir.name}')

        last_iter = resume_iterations[-1].iteration if resume_iterations else 0
        print(f'Resuming from iteration {last_iter} '
              f'(best train: {resume_best_accuracy:.1%}, '
              f'supervisor cost: ${resume_supervisor_cost:.4f})')

    # --- Output directory (results/orchestration_learner/TIMESTAMP) ---
    results_base = benchmark_dir / 'results' / 'orchestration_learner'
    output_dir = results_base / f'{timestamp}.orch_learner'

    # Set evaluate.result_dir so per-iteration eval results also go in the subfolder
    config.configure(evaluate=dict(result_dir=str(results_base)))
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / 'config.yaml')
    run_metadata = {
        'benchmark': benchmark_dir.name,
        'benchmark_dir': str(benchmark_dir),
        'config_file': str(cfg_path),
        'entry_point': entry_point_name,
        'ptools_module': ptools_module_name,
        'seed_orchestrate': seed_orchestrate,
        'results_base': str(results_base),
        'resume_from': resume or None,
        'train_split': train_split,
        'eval_split': eval_split,
        'n_train': len(train_dataset.cases),
        'n_eval': len(eval_dataset.cases) if eval_dataset else 0,
        'max_iterations': max_iterations,
        'target_accuracy': target_accuracy,
        'supervisor_model': supervisor_model,
    }
    (output_dir / 'run_metadata.json').write_text(
        json.dumps(run_metadata, indent=2)
    )

    # --- Print setup summary ---
    print('\n=== Orchestration Learner ===')
    print(f'Benchmark: {benchmark_dir.name}')
    print(f'Config: {cfg_path.name}')
    print(f'Ptools module: {ptools_module_name} ({evolved_path.name})')
    print(f'Entry point: {entry_point_name}')
    print(f'Supervisor: {supervisor_model}')
    print(f'Train: {len(train_dataset.cases)} cases, Eval: {len(eval_dataset.cases) if eval_dataset else 0} cases')
    print(f'Max iterations: {max_iterations}')
    if target_accuracy:
        print(f'Target accuracy: {target_accuracy:.1%}')
    if instructions:
        print(f'Custom instructions: {instructions[:100]}...')
    if model_choices_text:
        print('Model choices loaded')
    if resume:
        print(f'Resuming from: {resume}')
    print(f'Output: {output_dir}')

    # --- Run improvement loop ---
    report = improve_with_supervisor(
        entry_interface=entry_interface,
        tool_interfaces=tool_interfaces,
        catalog=catalog,
        evaluator=evaluator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        supervisor_model=supervisor_model,
        max_iterations=max_iterations,
        target_accuracy=target_accuracy,
        custom_instructions=instructions,
        model_choices=model_choices_text,
        output_dir=output_dir,
        ptools_module=ptools_module_obj,
        resume_iterations=resume_iterations,
        resume_best_accuracy=resume_best_accuracy,
        resume_best_eval_accuracy=resume_best_eval_accuracy,
        resume_supervisor_cost=resume_supervisor_cost,
        on_iteration_complete=_on_iteration,
    )

    # --- Final eval on held-out set ---
    # Re-get entry_interface from the (possibly reloaded) ptools module
    entry_interface = getattr(ptools_module_obj, entry_point_name)
    if eval_dataset:
        print(f'\n=== Final Evaluation on Held-Out Set ({len(eval_dataset.cases)} cases) ===')
        final_dir = output_dir / 'final_eval'
        final_dir.mkdir(exist_ok=True)
        with config.configuration(evaluate=dict(
            expt_name='rc_final_eval',
            result_dir=str(final_dir),
            record_details=True,
        )):
            csv_path = evaluator.evaluate(eval_dataset, entry_interface)
        import pandas as pd
        df = pd.read_csv(csv_path)
        eval_acc = df['correct'].mean()
        eval_cost = df.get('cost', pd.Series([0])).mean()
        report.final_eval_accuracy = eval_acc
        print(f'Final eval accuracy: {eval_acc:.1%}')
        print(f'Final eval avg cost: ${eval_cost:.4f}')

        # Save updated report
        (output_dir / 'report.json').write_text(
            report.model_dump_json(indent=2)
        )

    # --- Generate plots and HTML report ---
    print('\n=== Generating Plots & Report ===')
    _generate_plots(report, output_dir)
    _generate_html_report(report, output_dir)

    # --- Final summary ---
    print(f'\n{"=" * 60}')
    print('=== Orchestration Learner Complete ===')
    print(f'{"=" * 60}')
    print(f'Best train accuracy: {report.best_train_accuracy:.1%} '
          f'(iteration {report.best_iteration})')
    if report.final_eval_accuracy is not None:
        print(f'Final eval accuracy: {report.final_eval_accuracy:.1%}')
    print(f'Total supervisor cost: ${report.total_supervisor_cost:.4f}')
    print(f'Output saved to: {output_dir}')

    # Print iteration summary table
    has_eval = any(r.eval_accuracy is not None for r in report.iterations)
    eval_hdr = f'  {"Eval":>8}' if has_eval else ''
    print('\nIteration log:')
    print(f'  {"Iter":>4}  {"Train":>8}  {"Fail":>4}  {"TO":>3}{eval_hdr}  {"Sup $":>7}  {"Status":>10}')
    for r in report.iterations:
        status = 'KEPT' if r.kept else ('BASELINE' if r.iteration == 0 else 'ROLLBACK')
        eval_col = f'  {r.eval_accuracy:>7.1%}' if has_eval and r.eval_accuracy is not None else (f'  {"—":>8}' if has_eval else '')
        to_str = f'{r.train_timeouts:>3}' if r.train_timeouts else '  0'
        print(f'  {r.iteration:>4}  {r.train_accuracy:>7.1%}  {r.train_failures:>4}  {to_str}{eval_col}  '
              f'${r.supervisor_cost:>6.4f}  {status:>10}')


@app.command('view')
def view(
    run_dir: str = typer.Argument(..., help='Path to a remote_control output directory'),
):
    """Generate HTML report for an existing run directory."""
    from secretagent.orchestrate.improve import SupervisorReport

    output_dir = Path(run_dir)
    report_json = output_dir / 'report.json'
    if not report_json.exists():
        print(f'Error: {report_json} not found')
        raise typer.Exit(1)

    report = SupervisorReport.model_validate_json(report_json.read_text())
    _generate_html_report(report, output_dir)
    print(f'Open: {output_dir / "report.html"}')


if __name__ == '__main__':
    app()
