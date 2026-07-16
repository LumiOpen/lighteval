"""
name:
MT-Bench (inspect_ai)

dataset:
lighteval/mt-bench

abstract:
inspect_ai-compatible version of MT-Bench. Uses the scorer model server
(SCORER_MODEL_BASE_URL / SCORER_MODEL_PATH env vars) as the judge.
Supports 2-turn conversation; scores each turn independently.

languages:
english

tags:
conversational, generation, multi-turn
"""

import os
import re

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, generate, solver

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.mt_bench.judge_prompt_templates import (
    flow_judge_prompt_mt_bench_with_ref,
    flow_judge_prompt_mt_bench_without_ref,
)


def _scorer_endpoint_config(task_name: str = "mt_bench_inspect"):
    base_url = os.environ.get("SCORER_MODEL_BASE_URL")
    if not base_url:
        raise RuntimeError(f"SCORER_MODEL_BASE_URL must be set for {task_name}.")

    model_name = os.environ.get("SCORER_MODEL_PATH")
    if not model_name:
        raise RuntimeError(f"SCORER_MODEL_PATH must be set for {task_name}.")

    return base_url, model_name, os.environ.get("VLLM_API_KEY", "inspectai")


def _get_scorer_model(task_name: str = "mt_bench_inspect"):
    base_url, model_name, api_key = _scorer_endpoint_config(task_name)
    return get_model(
        f"openai-api/scorer/{model_name}",
        config=GenerateConfig(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        base_url=base_url,
        api_key=api_key,
    )


def _parse_judge_score(text: str) -> int:
    match = re.search(r"<score>\s*(\d)\s*</score>", text)
    return int(match.group(1)) if match else 0


def record_to_sample(record):
    return Sample(
        input=record["turns"][0],
        metadata={
            "turns": record["turns"],
            "reference": record.get("reference", []),
            "category": record.get("category", ""),
            "question_id": record.get("question_id", ""),
        },
    )


@solver
def require_scorer_endpoint(task_name: str = "mt_bench_inspect"):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _scorer_endpoint_config(task_name)
        return state

    return solve


@solver
def append_second_turn():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        turns = state.metadata["turns"]
        if len(turns) > 1:
            state.messages.append(ChatMessageUser(content=turns[1]))
        return state

    return solve


@scorer(metrics={"turn_1": [mean(), stderr()], "turn_2": [mean(), stderr()]})
def mt_bench_scorer():
    judge = None

    async def score(state: TaskState, target) -> Score:
        nonlocal judge
        if judge is None:
            judge = _get_scorer_model()

        turns = state.metadata["turns"]
        references = state.metadata.get("reference", [])
        assistant_messages = [m for m in state.messages if m.role == "assistant"]

        scores = {}
        for i, question in enumerate(turns):
            if i >= len(assistant_messages):
                scores[f"turn_{i + 1}"] = 0
                continue

            message = assistant_messages[i]
            raw = message.text if hasattr(message, "text") else str(message.content)
            answer = raw
            ref = references[i] if references and i < len(references) else None

            if ref:
                messages = flow_judge_prompt_mt_bench_with_ref(question, [], answer, ref)
            else:
                messages = flow_judge_prompt_mt_bench_without_ref(question, [], answer, None)

            judge_input = [ChatMessageUser(content=m["content"]) for m in messages if m["role"] == "user"]
            output = await judge.generate(
                input=judge_input,
                config=GenerateConfig(temperature=0, max_tokens=512),
            )
            scores[f"turn_{i + 1}"] = _parse_judge_score(output.completion)

        if "turn_2" not in scores:
            scores["turn_2"] = 0

        return Score(
            value=scores,
            explanation=f"category={state.metadata.get('category', '')} id={state.metadata.get('question_id', '')}",
        )

    return score


mt_bench_inspect = LightevalTaskConfig(
    name="mt_bench_inspect",
    prompt_function=None,
    hf_repo="lighteval/mt-bench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metrics=[],
    generation_size=1024,
    stop_sequence=[],
    sample_fields=record_to_sample,
    solver=[require_scorer_endpoint(), generate(cache=True), append_second_turn(), generate(cache=True)],
    scorer=mt_bench_scorer(),
)


TASKS_TABLE = [mt_bench_inspect]
