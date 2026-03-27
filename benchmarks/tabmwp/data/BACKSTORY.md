# TabMWP Data Provenance

## Source

**Dataset:** TabMWP (Tabular Math Word Problems)
**Paper:** Lu et al., "Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning," ICLR 2023.
**arXiv:** https://arxiv.org/abs/2209.14610
**Repository:** https://github.com/lupantech/PromptPG
**License:** CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)

## Download

Data was downloaded from the PromptPG GitHub repository using `download.py` in this directory.

```bash
uv run benchmarks/tabmwp/data/download.py
```

## Files

| File | Description | Examples |
|------|-------------|---------|
| `problems_train.json` | Training split | 23,059 |
| `problems_dev.json` | Validation split | 7,686 |
| `problems_test.json` | Test split | 7,686 |
| `problems_dev1k.json` | 1K dev subset | 1,000 |
| `problems_test1k.json` | 1K test subset | 1,000 |
| `splits.json` | Split assignments | — |

## Format

Each JSON file is a dict keyed by string example IDs. Each example contains:
- `question`: the math word problem (text)
- `table`: pipe-delimited table (text)
- `table_for_pd`: dict ready for `pandas.DataFrame()`
- `choices`: list of options (multi-choice) or null (free-text)
- `answer`: gold answer (string)
- `solution`: step-by-step reasoning trace (text)
- `ques_type`: "free_text" or "multi_choice"
- `ans_type`: "integer_number", "decimal_number", "extractive_text", "boolean_text", "other_text"
- `grade`: integer 1-8
- `table_title`, `row_num`, `column_num`, `unit`, `split`

## Citation

```bibtex
@inproceedings{lu2023dynamic,
    title={Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning},
    author={Lu, Pan and Qiu, Liang and Chang, Kai-Wei and Wu, Ying Nian and Zhu, Song-Chun and Rajpurohit, Tanmay and Clark, Peter and Kalyan, Ashwin},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2023}
}
```
