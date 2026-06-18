"""MT-Bench (English) — inspect-ai backend, true 2-turn.

Mirrors mt_bench_fi/main_inspect.py but for the English MT-Bench (lighteval/mt-bench),
using the English judge rubric. Multi-turn solver generates turn 1, appends the
turn-2 question to history, generates turn 2; two per-turn scorers judge each turn
via a separate scorer vLLM server (SCORER_MODEL_BASE_URL).
"""

import os
import re

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, solver

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


_SCORE_RE = re.compile(r"<score>\s*(\d+)\s*</score>")


def _judge_prompt(question: str, answer: str) -> str:
    return f"""# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large language model.

You will be provided with the inputs and output of the task, as well as the evaluation criteria and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation criteria and scoring rubric provided.

# INPUT
<inputs>
{question}
</inputs>

# OUTPUT
<output>
{answer}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
<evaluation_criteria>
How well the response answers the question?
</evaluation_criteria>

<scoring_rubric>
- Score 1: The response completely fails to answer the question.
- Score 2: The response barely answers the question.
- Score 3: The response partially answers the question.
- Score 4: The response mostly answers the question.
- Score 5: The response completely answers the question.
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria.
2. Review the inputs and output.
3. Compare output to score descriptions.
4. Pay attention to small details that might impact the score.
5. Write verbal feedback justifying your evaluation.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."""


def _get_scorer_model():
    base_url = os.environ.get("SCORER_MODEL_BASE_URL")
    if not base_url:
        raise RuntimeError("SCORER_MODEL_BASE_URL not set; need a judge vLLM server.")
    model_name = os.environ.get("SCORER_MODEL_PATH", "scorer")
    return get_model(
        f"openai-api/scorer/{model_name}",
        config=GenerateConfig(
            max_tokens=2048,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        base_url=base_url,
        api_key=os.environ.get("VLLM_API_KEY", "inspectai"),
    )


@solver
def mt_bench_two_turn():
    async def solve(state: TaskState, generate_fn):
        state = await generate_fn(state)
        turn2 = state.metadata.get("turn_2_query")
        if turn2:
            state.messages.append(ChatMessageUser(content=turn2))
            state = await generate_fn(state)
        return state

    return solve


async def _judge_turn(judge_model, question: str, answer: str):
    if not answer:
        return 0.0, ""
    result = await judge_model.generate(_judge_prompt(question, answer))
    text = result.completion or ""
    m = _SCORE_RE.search(text)
    raw = int(m.group(1)) if m else 0
    return (raw / 5.0 if raw else 0.0), text


def _assistant_texts(state: TaskState):
    return [m.text for m in state.messages if m.role == "assistant"]


@scorer(metrics=[mean(), stderr()])
def mt_bench_turn1():
    judge_model = _get_scorer_model()

    async def score(state: TaskState, target: Target):
        q1 = state.metadata.get("turn_1_query", state.input_text)
        answers = _assistant_texts(state)
        a1 = answers[0] if answers else ""
        val, expl = await _judge_turn(judge_model, q1, a1)
        return Score(value=val, answer=a1[:200], explanation=expl)

    return score


@scorer(metrics=[mean(), stderr()])
def mt_bench_turn2():
    judge_model = _get_scorer_model()

    async def score(state: TaskState, target: Target):
        q2 = state.metadata.get("turn_2_query")
        answers = _assistant_texts(state)
        if not q2 or len(answers) < 2:
            return Score(value=0.0, answer="", explanation="no turn-2 response")
        a1 = answers[0]
        a2 = answers[1]
        ctx_question = f"[Turn 1 question and answer for context]\nQ1: {q2}\nPrior answer: {a1}\n\n[Turn 2 question]\n{q2}"
        val, expl = await _judge_turn(judge_model, ctx_question, a2)
        return Score(value=val, answer=a2[:200], explanation=expl)

    return score


def record_to_sample(record):
    turns = record["turns"]
    md = {
        "category": record.get("category"),
        "question_id": record.get("question_id"),
        "turn_1_query": turns[0],
        "turn_2_query": turns[1] if len(turns) > 1 else None,
    }
    return Sample(input=turns[0], target="", metadata=md)


def _mt_bench_prompt(line, task_name: str = ""):
    return Doc(task_name=task_name, query=f"{line['turns'][0]}", choices=[], gold_index=[])


mt_bench_inspect = LightevalTaskConfig(
    name="mt_bench_inspect",
    prompt_function=_mt_bench_prompt,
    hf_repo="lighteval/mt-bench",
    hf_subset=None,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[],
    version=1,
    sample_fields=record_to_sample,
    solver=[mt_bench_two_turn()],
    scorer=[mt_bench_turn1(), mt_bench_turn2()],
)


TASKS_TABLE = [mt_bench_inspect]
