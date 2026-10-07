"""Build the anonymous supplementary zip from the repository.

Usage (from anywhere, with any Python 3.9 or newer):
  python build_supplement.py --repo REPO --out OUT_FOLDER [--appendix appendix.pdf]

What it does, in order. It stops at the first problem.
  1. Copies only the files on the allow list: the framework, task code and configs, the table
     scripts, the tests, the AFlow folder and the packaging files.
  2. For every table entry in tables/index.json, writes a small copy of the saved result file:
     one row per question with its ID, whether it was right, and its cost. Question text and
     model answers are left out. It checks that each small copy gives the same accuracy and cost
     as the full file.
  3. Replaces names, usernames, personal paths, repository links and our internal code revision.
  4. Scans every file again for those strings and for API keys. Stops if anything is found.
  5. Writes OUT_FOLDER/supplement/ and OUT_FOLDER/supplement.zip and stops if the zip is over 24 MB.
"""
import argparse
import csv
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BYTES = 24 * 1000 * 1000
csv.field_size_limit(2**31 - 1)

ROOT_FILES = ["pyproject.toml", "uv.lock", "LICENSE"]
SCRIPTS = ["hero_table.py", "learning_table.py", "table2_shortcut_recompute.py", "optimize_plot.py", "nsga2_eval_counts.py",
           # the AFlow runner and the collectors, which the tests also cover
           "aflow_table_tasks.py", "run_aflow_table_task.py", "run_aflow_table_musr.py", "collect_aflow_results.py",
           # the AFlow setup and launch scripts that benchmarks/COMMON/aflow-rebuttal/README.md tells readers to run
           "prepare_aflow.py", "export_aflow_datasets.py", "run_aflow_cell.py", "make_aflow_seed_prompts.py",
           "check_aflow_cost.py"]
# Internal notes that are not documentation for readers.
SKIP_FILES = {"src/secretagent/learn/inducer_results.md"}
# Tests that need the full saved data or a local AFlow checkout, neither of which is in the zip.
SKIP_TESTS = {"tests/test_aflow_table_adapters.py", "tests/test_aflow_table_tasks.py", "tests/test_aflow_prompt_repair.py",
              "tests/test_collect_table_runs.py", "tests/test_modo_control.py"}
# The AFlow adapters, scorers, selection rule and patch live here; only its results/ folder is left out.
AFLOW_INFRA = "benchmarks/COMMON/aflow-rebuttal/"
DOCS = ["docs/CLI.md", "docs/CONFIG_KEYS.md"]
# Folders under benchmarks/ that hold logs, old results or personal scratch work.
SKIP_BENCHMARK_DIRS = {"COMMON", "jerry", "codedistill_logs_v2", "results"}
# Any path containing one of these folder names is results, logs or data, not code.
SKIP_SEGMENT = re.compile(r"^(results?|test_results\w*|recordings?\w*|learned\w*|llm_cache|logs?|archive\w*|cache\w*|"
                          r"rollouts?|figures?|plots?|__pycache__|_?train_dirs|training\w*|outputs?|runs?|workspace\w*|scratch\w*)$", re.I)
CODE_EXT = {".py", ".yaml", ".yml", ".txt", ".json", ".toml", ".cfg", ".ini", ".sh", ".j2", ".jinja", ".prompt"}
MAX_CODE_FILE = 300 * 1000

# What the zip must not contain. Each pattern is replaced, then the whole zip is scanned again.
REPLACE = [
    # Only links to our own accounts; public upstream projects (benchmarks, CLIP) stay.
    (re.compile(r"https?://(www\.)?github\.com/(wwcohen|joshh99)[^\s\"'<>)\]]*", re.I), "<repository-url>"),
    # Real personal paths only. Made-up example paths in the tests (/home/user/...) stay.
    (re.compile(r"(/mnt/[a-z]/|/home/(?!user/)[^/\s\"']+/|/Users/(?!user/)[^/\s\"']+/|[A-Za-z]:[\\/]+Users[\\/]+)[^\s\"',;)\]}]*", re.I), "<path>"),
    (re.compile(r"\b(wwcohen|joshh99)\b", re.I), "anonymous"),
    (re.compile(r"\b(aditya|cassie|jerry|suman|joshua|josh|momo|william|cohen|carnegie[ _-]?mellon|cmu)\b", re.I), "anonymous"),
    (re.compile(r"\bLex\b"), "anonymous"),
    (re.compile(r"6ba57557[0-9a-f]*"), "<revision>"),
]
SECRETS = re.compile(r"(sk-or-v1-[A-Za-z0-9]{10,}|sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{32,}|AIza[0-9A-Za-z_-]{30,}|"
                     r"Bearer\s+[A-Za-z0-9._-]{20,}|(api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9._-]{16,}['\"])", re.I)
FORBIDDEN = [p for p, _ in REPLACE] + [SECRETS]


def is_code(rel):
    parts = rel.split("/")
    infra = rel.startswith(AFLOW_INFRA)
    if parts[0] == "benchmarks" and not infra and (len(parts) < 3 or parts[1] in SKIP_BENCHMARK_DIRS):
        return False
    if any(SKIP_SEGMENT.match(p) for p in parts[1:-1]):
        return False
    if "data" in parts[1:-1] and not rel.endswith(".py"):
        return False
    name = parts[-1]
    # Dataset samples and logs: benchmark text and run output, not code.
    if re.search(r"(^data[_.]|train|valid|test_set|^log|_log|traces?\.)", name, re.I) and not name.endswith(".py"):
        return False
    if name.endswith(".md"):
        return name.lower() == "readme.md" or (infra and "install" in name.lower())
    if infra and (name.endswith(".patch") or name.endswith(".redacted")):
        return True
    return Path(name).suffix.lower() in CODE_EXT


def code_files(repo):
    out = [f for f in ROOT_FILES + DOCS if (repo / f).is_file()]
    out += [f"scripts/{s}" for s in SCRIPTS if (repo / "scripts" / s).is_file()]
    for top in ("src/secretagent", "tests", "benchmarks", "examples"):
        for p in sorted((repo / top).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(repo).as_posix()
                if rel in SKIP_TESTS or rel in SKIP_FILES:
                    continue
                if (top != "benchmarks" or is_code(rel)) and p.stat().st_size <= MAX_CODE_FILE:
                    out.append(rel)
    return sorted(set(out))


def slim(src, dst):
    """Write only question ID, correct and cost. Returns (accuracy, mean cost of recorded, total cost, n)."""
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        c = str(r.get("correct", "")).strip()
        correct = "" if c in ("", "nan", "None") else ("1" if c in ("1", "1.0", "True", "true") else "0")
        k = str(r.get("cost", "")).strip()
        cost = "" if k in ("", "nan", "NaN", "None") else k
        out.append({"case_name": r["case_name"], "correct": correct, "cost": cost})
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_name", "correct", "cost"])
        w.writeheader()
        w.writerows(out)
    acc = [float(r["correct"]) for r in out if r["correct"] != ""]
    costs = [float(r["cost"]) for r in out if r["cost"] != ""]
    return (sum(acc) / len(acc) if acc else None, sum(costs) / len(costs) * 100 if costs else None, len(out))


def scrub_text(text):
    for pattern, repl in REPLACE:
        text = pattern.sub(repl, text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--appendix", type=Path, help="the appendix PDF from the paper build")
    a = ap.parse_args()
    repo = a.repo.resolve()
    target = a.out / "supplement"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    # 1. code
    files = code_files(repo)
    for rel in files:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dst)
    aflow_src = repo / "submission" / "aflow"
    shutil.copytree(aflow_src, target / "aflow", ignore=shutil.ignore_patterns("outputs", "__pycache__"))
    for name in ("rebuild_tables.py", "README.md"):
        shutil.copy2(HERE / name, target / name)
    (target / "tables").mkdir()
    shutil.copy2(HERE / "expected_tables.json", target / "tables" / "expected_tables.json")
    summaries = sorted((repo / "benchmarks" / "COMMON" / "optimize-results").rglob("nsga2_summary*.csv"))
    for p in summaries:
        dst = target / p.relative_to(repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    if a.appendix:
        shutil.copy2(a.appendix, target / "appendix.pdf")
    print(f"copied {len(files)} code files, {len(summaries)} search summaries")

    # 2. small copies of the saved results, checked against the index
    index = json.loads((HERE / "index.json").read_text(encoding="utf-8"))
    done = {}
    for e in index["entries"]:
        rel = e["file"]
        if rel not in done:
            done[rel] = slim(repo / rel, target / rel)
            config = (repo / rel).parent / "config.yaml"
            if config.is_file():
                shutil.copy2(config, target / Path(rel).parent / "config.yaml")
        acc, cost, n = done[rel]
        same = n == e["n"] and (acc is None) == (e["correct"] is None) and (cost is None) == (e["cost100"] is None)
        same = same and (acc is None or abs(acc - e["correct"]) < 1e-9) and (cost is None or abs(cost - e["cost100"]) < 1e-9)
        if not same:
            raise SystemExit(f"small copy of {rel} does not match the full file")
    for e in index["entries"]:
        e.pop("sha256_note", None)
    (target / "tables" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {len(done)} small result files; all match the full files")

    # 3 and 4. scrub, then scan everything
    problems = []
    for p in sorted(target.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(target).as_posix()
        if p.suffix.lower() in (".pdf", ".png", ".jpg"):
            raw = p.read_bytes().decode("latin-1")
            for pattern in FORBIDDEN:
                m = pattern.search(raw)
                if m:
                    problems.append(f"{rel}: '{m.group(0)[:40]}' inside a binary file; fix it at the source")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: not a text file and not on the binary list")
            continue
        new = scrub_text(text)
        if new != text:
            p.write_text(new, encoding="utf-8")
        for pattern in FORBIDDEN:
            m = pattern.search(new)
            if m:
                problems.append(f"{rel}: '{m.group(0)[:40]}' is still there after scrubbing")
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"{len(problems)} problem(s); nothing was zipped")

    # 5. zip and size
    zpath = a.out / "supplement.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(target.rglob("*")):
            if p.is_file():
                z.write(p, Path("supplement") / p.relative_to(target))
    size = zpath.stat().st_size
    print(f"zip: {zpath} ({size / 1e6:.2f} MB, {sum(1 for _ in target.rglob('*') if _.is_file())} files)")
    if size > MAX_BYTES:
        raise SystemExit(f"the zip is {size / 1e6:.2f} MB; the limit we set is 24 MB")


if __name__ == "__main__":
    main()
