"""Tests for RLMFactory (Recursive Language Model REPL)."""

import pytest
from omegaconf import OmegaConf

from conftest import needs_gemini_key
from secretagent import config, record
from secretagent.core import interface, all_factories, _INTERFACES
from secretagent.implement.rlm import (
    RLMFactory, _extract_code_blocks, _strip_code_blocks, _load_template,
)


@pytest.fixture(autouse=True)
def reset_config():
    config.GLOBAL_CONFIG = OmegaConf.create()
    yield
    config.GLOBAL_CONFIG = OmegaConf.create()


# --- factory registration ---

def test_rlm_factory_registered():
    factory_names = [name for name, _ in all_factories()]
    assert 'rlm' in factory_names


# --- prompt generation ---

def test_create_prompt_includes_stub_and_context_info():
    @interface
    def find_word(text: str) -> str:
        """Find the secret word hidden in the text."""

    template = _load_template('rlm_system.txt')
    prompt = template.substitute(
        stub_src=find_word.src,
        context_info="type=str, length=100 chars",
        thoughts="",
    )
    assert 'Find the secret word' in prompt
    assert 'type=str, length=100 chars' in prompt
    _INTERFACES.remove(find_word)


def test_create_prompt_includes_llm_query_instructions():
    @interface
    def analyze(text: str) -> str:
        """Analyze the text."""

    template = _load_template('rlm_system.txt')
    prompt = template.substitute(
        stub_src=analyze.src,
        context_info="type=str",
        thoughts="",
    )
    assert 'llm_query' in prompt
    assert 'FINAL' in prompt
    assert 'FINAL_VAR' in prompt
    _INTERFACES.remove(analyze)


# --- parsing helpers ---

def test_final_answer_parsing():
    """Test regex extraction for FINAL(some answer)."""
    import re
    text = "After analysis, I found the answer.\nFINAL(42)\n"
    text_clean = _strip_code_blocks(text)
    match = re.search(r'FINAL\((.+?)\)', text_clean, re.MULTILINE)
    assert match is not None
    assert match.group(1) == '42'


def test_final_var_parsing():
    """Test regex extraction for FINAL_VAR(my_var)."""
    import re
    text = "The answer is stored in result.\nFINAL_VAR(result)\n"
    text_clean = _strip_code_blocks(text)
    match = re.search(r'FINAL_VAR\((.+?)\)', text_clean, re.MULTILINE)
    assert match is not None
    assert match.group(1) == 'result'


def test_code_block_extraction():
    """Test extracting ```repl blocks from LLM output."""
    text = """Let me check the context.
```repl
x = context.split()
print(len(x))
```
That gives us the word count.
```python
answer = x[0]
FINAL(answer)
```
"""
    blocks = _extract_code_blocks(text)
    assert len(blocks) == 2
    assert 'context.split()' in blocks[0]
    assert 'FINAL(answer)' in blocks[1]


def test_code_block_extraction_no_blocks():
    blocks = _extract_code_blocks("Just some text with no code.")
    assert blocks == []


# --- integration tests (need GEMINI_API_KEY) ---

GEMINI_TEST_MODEL = 'gemini/gemini-3.1-flash-lite-preview'


@needs_gemini_key
def test_rlm_simple_string_lookup():
    """RLM should find a hidden word in a text."""
    @interface
    def find_secret(text: str) -> str:
        """Find and return the single word enclosed in triple asterisks (***word***) in the text."""

    find_secret.implement_via('rlm', llm={'model': GEMINI_TEST_MODEL})

    # build a text with a hidden word
    filler = "The quick brown fox jumps over the lazy dog. " * 20
    text = filler + "***PINEAPPLE***" + filler

    result = find_secret(text)
    assert 'PINEAPPLE' in result.upper()
    _INTERFACES.remove(find_secret)


@needs_gemini_key
def test_rlm_with_llm_query():
    """RLM should be able to use llm_query for sub-tasks."""
    @interface
    def count_animals(text: str) -> str:
        """Count how many distinct animal names appear in the text. Return just the number."""

    count_animals.implement_via('rlm', llm={'model': GEMINI_TEST_MODEL})

    text = "The cat sat on the mat. A dog chased a bird. The cat and the dog played."

    with record.recorder() as rollout:
        result = count_animals(text)

    # should find 3 animals: cat, dog, bird
    assert '3' in str(result)
    _INTERFACES.remove(count_animals)


@needs_gemini_key
def test_rlm_records_trajectory():
    """RLM should record trajectory with turn count and LLM call count."""
    @interface
    def simple_task(text: str) -> str:
        """Return the first word of the text."""

    simple_task.implement_via('rlm', llm={'model': GEMINI_TEST_MODEL})

    with record.recorder() as rollout:
        simple_task("hello world")

    rlm_entries = [r for r in rollout if r['func'] == 'simple_task']
    assert len(rlm_entries) == 1
    step_info = rlm_entries[0]['step_info']
    assert 'turns' in step_info
    assert 'n_llm_calls' in step_info
    assert step_info['turns'] >= 1
    assert step_info['n_llm_calls'] >= 1
    _INTERFACES.remove(simple_task)


@needs_gemini_key
def test_rlm_handles_error_gracefully():
    """If generated code raises an exception, the LLM should see the error and retry."""
    @interface
    def compute(x: str) -> str:
        """Parse the number from the text and return its square. Return just the number."""

    compute.implement_via('rlm', max_turns=5, llm={'model': GEMINI_TEST_MODEL})

    result = compute("the number is 7")
    assert '49' in str(result)
    _INTERFACES.remove(compute)
