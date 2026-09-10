"""
name:
EvalPlus (HumanEval+ / MBPP+)

dataset:
ezosa/evalplus

abstract:
HumanEval+ and MBPP+ (EvalPlus) code-generation benchmarks evaluated as native
inspect_ai tasks. The model is prompted (instruct-style) to implement a Python
function; the code is extracted from its final answer and graded by *executing*
it against EvalPlus's augmented unit tests. pass@1 is estimated over N samples
via the wrapper's epochs (mean epoch-reducer); no judge model.

Reasoning is left ON: the checkpoint chat template opens `<think>`, vLLM's
`reasoning_parser=qwen3` separates the reasoning trace, and the scorer extracts
code from the post-reasoning answer.

Code execution is unsafe and MUST run under the TensorWave sandbox
(`eval_lighteval_sandboxed.sbatch` + `sandbox_grading.py`), which wraps this
task's `execution.check_correctness` with a per-sample network/user namespace.
The task names are registered in `sandboxed_evals.SANDBOXED_TASK_PREFIXES`
(`humaneval`, `mbpp`) and its grader in `GRADER_PATCHES`.

Exposes: humaneval_plus, mbpp_plus

languages:
english

tags:
code-generation, code, humaneval, mbpp, evalplus, sandboxed
"""

import asyncio
import os

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, accuracy, scorer, stderr
from inspect_ai.solver import generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.evalplus import execution


# Wall-clock ceiling (seconds) for executing one problem's generated code.
_EXEC_TIMEOUT = 20


def _humaneval_message(record):
    return (
        "Complete the following Python function. Return the complete function, "
        "including the signature, in a single ```python code block.\n\n"
        f"```python\n{record['prompt']}\n```"
    )


def _mbpp_message(record):
    example = record.get("example") or ""
    example_block = f"\n\nYour function must pass this test:\n```python\n{example}\n```" if example else ""
    return (
        f"{record['prompt']}\n\n"
        f"Write a single Python function named `{record['entry_point']}` that solves this. "
        f"Return the complete function in a single ```python code block.{example_block}"
    )


def record_to_sample(record):
    kind = record["kind"]
    message = _humaneval_message(record) if kind == "humaneval" else _mbpp_message(record)
    return Sample(
        input=[ChatMessageUser(content=message)],
        metadata={
            "task_id": record["task_id"],
            "kind": kind,
            "entry_point": record["entry_point"],
            "test": record["test"],
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def code_exec_scorer():
    async def score(state, target):
        md = state.metadata
        code = execution.extract_code(state.output.completion or "")
        if not code:
            return Score(value=0.0, explanation="no code block in answer")
        sample = execution.make_sample(md["test"], md["entry_point"])
        # Reference the grader via the module attribute so the sandbox monkey-patch
        # (sandbox_grading.install) applies when running sandboxed.
        result = await asyncio.to_thread(execution.check_correctness, sample, code, _EXEC_TIMEOUT)
        passed = bool(result) and all(x == 1 for x in result)
        return Score(
            value=1.0 if passed else 0.0,
            answer=code,
            explanation=f"{md['task_id']}: tests {'passed' if passed else 'failed'} ({result})",
        )

    return score


# One multi-config HF dataset (split="train"), config per benchmark. Override
# with EVALPLUS_HF_REPO to point at a mirror (e.g. under an org).
_HF_REPO = os.environ.get("EVALPLUS_HF_REPO", "ezosa/evalplus")

_TASKS = ["humaneval_plus", "mbpp_plus"]


def _make_config(name, subset, repo):
    return LightevalTaskConfig(
        name=name,
        prompt_function=None,
        hf_repo=repo,
        hf_subset=subset,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        metrics=[],
        generation_size=32768,
        stop_sequence=[],
        version=1,
        sample_fields=record_to_sample,
        solver=[generate(cache=True)],
        scorer=code_exec_scorer(),
    )


TASKS_TABLE = [_make_config(name, name, _HF_REPO) for name in _TASKS]
