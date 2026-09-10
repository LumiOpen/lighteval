"""Unit tests for the EvalPlus (HumanEval+/MBPP+) execution grader.

Pure-stdlib (multiprocessing + signal); no optional deps. Exercises code
extraction and the execute-and-grade path that the sandbox wrapper drives.
"""

from lighteval.tasks.tasks.evalplus import execution


# A tiny HumanEval-style problem: implement add(a, b).
TEST_HARNESS = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(0, 0) == 0\n"
ENTRY_POINT = "add"
CORRECT = "def add(a, b):\n    return a + b"
WRONG = "def add(a, b):\n    return a - b"


def _grade(code: str) -> bool:
    sample = execution.make_sample(TEST_HARNESS, ENTRY_POINT)
    result = execution.check_correctness(sample, code, timeout=10)
    return bool(result) and all(x == 1 for x in result)


def test_correct_solution_passes():
    assert _grade(CORRECT) is True


def test_wrong_solution_fails():
    assert _grade(WRONG) is False


def test_exception_fails():
    assert _grade("def add(a, b):\n    raise ValueError('boom')") is False


def test_timeout_fails():
    assert _grade("def add(a, b):\n    while True:\n        pass") is False


def test_missing_function_fails():
    assert _grade("x = 1") is False


def test_extract_code_prefers_python_fence():
    text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert execution.extract_code(text) == "def add(a, b):\n    return a + b"


def test_extract_code_post_reasoning_answer():
    # With reasoning_parser=qwen3 the completion is the final answer only.
    text = "The function adds the two inputs.\n```python\ndef add(a, b):\n    return a + b\n```"
    assert _grade(execution.extract_code(text)) is True


def test_extract_code_falls_back_to_raw_text():
    assert execution.extract_code("def add(a, b):\n    return a + b").startswith("def add")


def test_extract_code_empty():
    assert execution.extract_code("") == ""
