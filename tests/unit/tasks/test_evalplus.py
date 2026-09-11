"""Unit tests for the EvalPlus (HumanEval+/MBPP+) execution grader.

Pure-stdlib (multiprocessing + signal); no optional deps. Exercises code
extraction and the execute-and-grade path that the sandbox wrapper drives.
"""

from lighteval.tasks.tasks.evalplus import execution


# A tiny HumanEval-style problem: implement add(a, b). HumanEval+ tests define a
# ``check(candidate)`` function that must be invoked with the entry point.
TEST_HARNESS = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(0, 0) == 0\n"
ENTRY_POINT = "add"
CORRECT = "def add(a, b):\n    return a + b"
WRONG = "def add(a, b):\n    return a - b"

# The same problem in MBPP+ style: assertions call the target function directly
# at module scope, with no ``check`` wrapper. Appending ``check(add)`` here would
# raise NameError and fail every problem (the bug this guards against).
MBPP_HARNESS = "assert add(2, 3) == 5\nassert add(0, 0) == 0\n"


def _grade(code: str, harness: str = TEST_HARNESS, entry: str = ENTRY_POINT) -> bool:
    sample = execution.make_sample(harness, entry)
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


def test_mbpp_style_direct_assert_correct_passes():
    # MBPP+ harness: module-level asserts, no check() wrapper. Regression test
    # for the bug where check(entry_point) was appended unconditionally.
    assert _grade(CORRECT, harness=MBPP_HARNESS) is True


def test_mbpp_style_direct_assert_wrong_fails():
    assert _grade(WRONG, harness=MBPP_HARNESS) is False


def test_mbpp_harness_does_not_append_check_call():
    sample = execution.make_sample(MBPP_HARNESS, ENTRY_POINT)
    program = execution._build_program(sample, CORRECT)
    assert "check(add)" not in program
    assert "assert add(2, 3) == 5" in program


def test_humaneval_harness_appends_check_call():
    sample = execution.make_sample(TEST_HARNESS, ENTRY_POINT)
    program = execution._build_program(sample, CORRECT)
    assert "check(add)" in program


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
