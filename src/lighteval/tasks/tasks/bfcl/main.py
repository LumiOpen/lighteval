"""
name:
BFCL non-live (function calling)

dataset:
ezosa/bfcl

abstract:
Berkeley Function Calling Leaderboard (BFCL) non-live categories evaluated as
native inspect_ai tasks. The model is prompted in BFCL's canonical python-list
format ("[func(arg=val), ...]"); the response is decoded with BFCL's own Python
AST parser and graded with BFCL's real `ast_checker` (name / count /
required-params / type / value matching). Programmatic scoring — no judge model.

Exposes one task per non-live category plus a merged `bfcl_nonlive` average:
  bfcl_simple_python, bfcl_multiple, bfcl_parallel, bfcl_parallel_multiple,
  bfcl_simple_java, bfcl_simple_javascript, bfcl_irrelevance, bfcl_nonlive

Requires BFCL's `ast_checker` from bfcl-eval. It pins numpy==1.26.4 (conflicts with
lighteval's numpy>=2) but the modules used here are numpy-free, so install it without
deps: `pip install --no-deps bfcl-eval` (or set `BFCL_EVAL_ROOT` to a gorilla
`berkeley-function-call-leaderboard` checkout).
Java/JavaScript categories are evaluated in python-list mode (BFCL's language-
specific type coercion is still applied by `ast_checker`); the model is not
prompted in Java/JS syntax.

languages:
english

tags:
function-calling, tool-use, agentic, bfcl
"""

import json
import os

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, accuracy, scorer, stderr
from inspect_ai.solver import generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.bfcl.core import build_system_prompt, grade


_ROLE_TO_MSG = {
    "system": ChatMessageSystem,
    "assistant": ChatMessageAssistant,
    "user": ChatMessageUser,
}


def record_to_sample(record):
    functions = json.loads(record["function"])
    question = json.loads(record["question"])
    gt_raw = record.get("ground_truth")
    ground_truth = json.loads(gt_raw) if gt_raw else None

    system_prompt = build_system_prompt(functions)
    turn0 = question[0] if question else []

    messages = []
    if turn0 and turn0[0].get("role") == "system":
        merged = system_prompt + "\n\n" + turn0[0].get("content", "")
        messages.append(ChatMessageSystem(content=merged))
        rest = turn0[1:]
    else:
        messages.append(ChatMessageSystem(content=system_prompt))
        rest = turn0
    for msg in rest:
        cls = _ROLE_TO_MSG.get(msg.get("role"), ChatMessageUser)
        messages.append(cls(content=msg.get("content", "")))

    return Sample(
        input=messages,
        metadata={
            "id": record["id"],
            "category": record["category"],
            "function": functions,
            "ground_truth": ground_truth,
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def bfcl_ast_scorer():
    async def score(state, target):
        md = state.metadata
        value, explanation = grade(
            state.output.completion or "",
            md["category"],
            md["function"],
            md["ground_truth"],
        )
        return Score(value=value, explanation=explanation)

    return score


# Data hosted on the HF Hub as one multi-config dataset (split="train"): the
# non-live config group (`simple_python`, ..., `irrelevance`, merged `nonlive`)
# and the live config group (`live_simple`, ..., `live_irrelevance`, merged
# `live`). Override with BFCL_HF_REPO to point at a mirror (e.g. under an org).
_HF_REPO = os.environ.get("BFCL_HF_REPO", "ezosa/bfcl")

_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "simple_java",
    "simple_javascript",
    "irrelevance",
]
# Live (real user-contributed) categories: 4 AST + relevance/irrelevance.
_LIVE_CATEGORIES = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
]


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
        generation_size=4096,
        stop_sequence=[],
        version=1,
        sample_fields=record_to_sample,
        solver=[generate(cache=True)],
        scorer=bfcl_ast_scorer(),
    )


TASKS_TABLE = (
    [_make_config(f"bfcl_{cat}", cat, _HF_REPO) for cat in _CATEGORIES]
    + [_make_config("bfcl_nonlive", "nonlive", _HF_REPO)]
    + [_make_config(f"bfcl_{cat}", cat, _HF_REPO) for cat in _LIVE_CATEGORIES]
    + [_make_config("bfcl_live", "live", _HF_REPO)]
)
