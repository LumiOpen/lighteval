"""
name:
MT-Bench Finnish (inspect_ai)

dataset:
LumiOpen/mtbench_multi

abstract:
inspect_ai-compatible Finnish MT-Bench task. Uses the scorer model server
(SCORER_MODEL_BASE_URL / SCORER_MODEL_PATH env vars) as the judge.
Supports 2-turn conversation; scores each turn independently.

languages:
finnish

tags:
conversational, generation, multi-turn
"""

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, mean, scorer, stderr
from inspect_ai.solver import generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.mt_bench.judge_prompt_templates import (
    flow_judge_prompt_mt_bench_with_ref,
    flow_judge_prompt_mt_bench_without_ref,
)
from lighteval.tasks.tasks.mt_bench.main_inspect import (
    _get_scorer_model,
    _parse_judge_score,
    append_second_turn,
    require_scorer_endpoint,
)
from lighteval.utils.language_detection import LanguageDetectionResult, detect_expected_language


TASK_NAME = "mt_bench_fi_inspect"
EXPECTED_LANGUAGE = "fi"
FINNISH_JUDGE_NOTE = "Note: The task is in Finnish. Evaluate the response based on the Finnish language content."


def _add_finnish_note(messages):
    noted_messages = []
    for message in messages:
        if message["role"] == "user":
            noted_messages.append(
                {**message, "content": message["content"].replace("# INPUT", f"{FINNISH_JUDGE_NOTE}\n\n# INPUT", 1)}
            )
        else:
            noted_messages.append(message)
    return noted_messages


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


def _language_explanation(language_results: dict[str, LanguageDetectionResult]) -> str:
    parts = []
    for turn, result in language_results.items():
        detected = result.detected_language or "unknown"
        confidence = "" if result.confidence is None else f":{result.confidence:.3f}"
        parts.append(f"{turn}_language={detected}{confidence}")
    return " ".join(parts)


@scorer(metrics={"turn_1": [mean(), stderr()], "turn_2": [mean(), stderr()]})
def mt_bench_fi_scorer():
    judge = None

    async def score(state, target) -> Score:
        nonlocal judge

        turns = state.metadata["turns"]
        references = state.metadata.get("reference", [])
        assistant_messages = [m for m in state.messages if m.role == "assistant"]

        scores = {}
        language_results = {}
        for i, question in enumerate(turns):
            turn_key = f"turn_{i + 1}"
            if i >= len(assistant_messages):
                scores[turn_key] = 0
                continue

            message = assistant_messages[i]
            raw = message.text if hasattr(message, "text") else str(message.content)
            answer = raw
            ref = references[i] if references and i < len(references) else None
            language_result = detect_expected_language(answer, EXPECTED_LANGUAGE)
            language_results[turn_key] = language_result
            if not language_result.is_expected:
                scores[turn_key] = 0
                continue

            if ref:
                messages = flow_judge_prompt_mt_bench_with_ref(question, [], answer, ref)
            else:
                messages = flow_judge_prompt_mt_bench_without_ref(question, [], answer, None)

            judge_input = [
                ChatMessageUser(content=m["content"]) for m in _add_finnish_note(messages) if m["role"] == "user"
            ]
            if judge is None:
                judge = _get_scorer_model(TASK_NAME)
            output = await judge.generate(
                input=judge_input,
                config=GenerateConfig(temperature=0, max_tokens=512),
            )
            scores[turn_key] = _parse_judge_score(output.completion)

        if "turn_2" not in scores:
            scores["turn_2"] = 0

        return Score(
            value=scores,
            explanation=(
                f"category={state.metadata.get('category', '')} id={state.metadata.get('question_id', '')} "
                f"{_language_explanation(language_results)}"
            ),
        )

    return score


mt_bench_fi_inspect = LightevalTaskConfig(
    name=TASK_NAME,
    prompt_function=None,
    hf_repo="LumiOpen/mtbench_multi",
    hf_subset="fi",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="",
    few_shots_select="random",
    metrics=[],
    generation_size=1024,
    stop_sequence=[],
    sample_fields=record_to_sample,
    solver=[require_scorer_endpoint(TASK_NAME), generate(cache=True), append_second_turn(), generate(cache=True)],
    scorer=mt_bench_fi_scorer(),
)


TASKS_TABLE = [mt_bench_fi_inspect]
