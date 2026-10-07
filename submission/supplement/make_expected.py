"""Read the numbers printed in the paper's tables, so the rebuild can be checked against them.

Usage: python make_expected.py --hero-dir PAPER/results --learning FINAL-TABLES.tex --out expected_tables.json

Main results tables: results/hero-compact.tex, results/hero.tex and results/hero-cost.tex.
Learning tables: the tab:learning and tab:learning-cost blocks of the given .tex file.
Each table becomes {row name: [cell, ...]}, where a cell is the printed number as a string
("0.84") or "--". Bold, daggers and stars are dropped; they do not change the number.
"""
import argparse
import json
import re
from pathlib import Path

NUMBER = re.compile(r"-?\d+\.\d+")


def cells_of(line):
    line = line.split("%")[0] if not line.lstrip().startswith("%") else ""
    line = line.replace(r"\midrule", "").strip()
    if line.endswith(r"\\"):
        line = line[:-2]
    if "&" not in line:
        return None
    parts = [p.strip() for p in line.split("&")]
    name, rest = parts[0], parts[1:]
    out = []
    for p in rest:
        m = NUMBER.search(p)
        out.append(m.group(0) if m else "--")
    return name, out


def clean_name(name):
    name = re.sub(r"\\new\{|\\makecell\[l\]\{|\\textrm\{|\}", "", name)
    return re.sub(r"\s+", " ", name.replace(r"\\", " ")).strip()


def parse_block(text):
    rows = {}
    body = text.split(r"\midrule", 1)[1] if r"\midrule" in text else text
    body = body.split(r"\bottomrule")[0]
    for raw in body.splitlines():
        got = cells_of(raw)
        if got:
            name, vals = got
            rows[clean_name(name)] = vals
    return rows


def table_block(tex, label):
    """The table environment that holds \\label{label}."""
    i = tex.index(r"\label{" + label + "}")
    start = tex.rfind(r"\begin{table", 0, i)
    end = tex.index(r"\end{table", i)
    return tex[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero-dir", required=True, type=Path)
    ap.add_argument("--learning", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = {}
    for name in ("hero-compact", "hero", "hero-cost"):
        out[name] = parse_block((a.hero_dir / f"{name}.tex").read_text(encoding="utf-8"))
    tex = a.learning.read_text(encoding="utf-8")
    out["learning"] = parse_block(table_block(tex, "tab:learning"))
    out["learning-cost"] = parse_block(table_block(tex, "tab:learning-cost"))
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(f"{k}: {len(v)} rows")


if __name__ == "__main__":
    main()
