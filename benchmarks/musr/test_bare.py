"""Quick test: bare prompt (no instructions) with choice matching."""

import ast
import json
import re
import sys
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

from secretagent import config
from secretagent.llm_util import llm

def match_choice(text, choices):
    """Try to match LLM output to a choice index.

    Searches from the end of the text for the last mentioned choice.
    """
    text_lower = text.strip().lower()
    # Find the last occurrence of each choice in the full text
    last_pos = {i: text_lower.rfind(c.lower()) for i, c in enumerate(choices)}
    # Filter to choices that appear at all
    found = {i: pos for i, pos in last_pos.items() if pos >= 0}
    if found:
        # Return the choice that appears latest in the text
        return max(found, key=found.get)
    # Try bare index on last line
    last_line = text.strip().splitlines()[-1].strip()
    if re.fullmatch(r'\d+', last_line):
        return int(last_line)
    return None

def main():
    config.configure(cachier=dict(enable_caching=False))
    model = sys.argv[1] if len(sys.argv) > 1 else "together_ai/deepseek-ai/DeepSeek-V3"
    split = sys.argv[2] if len(sys.argv) > 2 else "murder_mysteries"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 75

    data_file = _BENCHMARK_DIR / 'data' / f'{split}.json'
    with open(data_file) as f:
        examples = json.load(f)['examples'][:n]

    correct = 0
    for i, ex in enumerate(examples):
        choices = ex['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)

        prompt = f"{ex['narrative']}\n\n{ex['question']}\n\n{choices}"
        output, stats = llm(prompt, model)

        predicted = match_choice(output, choices)
        expected = ex['answer_index']
        is_correct = (predicted == expected)
        correct += is_correct

        print(f"[{i}] predicted={predicted} expected={expected} {'OK' if is_correct else 'WRONG'}")
        print(f"    LLM output (last 200 chars): ...{output[-200:]}")
        print()

    print(f"\nAccuracy: {correct}/{n} ({correct/n:.0%})")

if __name__ == '__main__':
    main()
