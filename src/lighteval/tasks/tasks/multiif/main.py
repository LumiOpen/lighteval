"""
name:
Multi-IF

dataset:
facebook/Multi-IF

abstract:
Three-turn multilingual instruction following with generated conversation history.
The headline score averages strict/loose prompt/instruction accuracy over turns.

languages:
english, chinese, french, hindi, italian, portuguese, russian, spanish

tags:
instruction-following, multi-turn

paper:
https://arxiv.org/abs/2410.15553
"""

import json
from statistics import mean

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

from lighteval.tasks.lighteval_task import LightevalTaskConfig


DATASET_REVISION = "0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa"


def _decode(value):
    return json.loads(value) if isinstance(value, str) else value


def record_to_sample(record):
    turns = []
    for index in range(1, 4):
        prompt = record[f"turn_{index}_prompt"]
        if prompt in (None, "", "None"):
            continue
        prompt = _decode(prompt)
        instructions = _decode(record[f"turn_{index}_instruction_id_list"])
        kwargs = [_decode(item) for item in _decode(record[f"turn_{index}_kwargs"])]
        if prompt["role"] != "user" or not instructions or len(instructions) != len(kwargs):
            raise ValueError(f"Invalid Multi-IF turn {index} for {record['key']}")
        turns.append({"index": index, "prompt": prompt["content"], "instructions": instructions, "kwargs": kwargs})
    if not turns or turns[0]["index"] != 1:
        raise ValueError(f"Missing first Multi-IF turn for {record['key']}")
    return Sample(
        id=record["key"],
        input=turns[0]["prompt"],
        metadata={"turns": turns, "language": record["language"]},
    )


@solver
def multiif_conversation():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for position, turn in enumerate(state.metadata["turns"]):
            if position:
                state.messages.append(ChatMessageUser(content=turn["prompt"]))
            # Inspect appends the generated assistant message to the conversation.
            state = await generate(state)
        return state

    return solve


def _check_response(response, turn):
    # Keep the benchmark's own multilingual checks; IFEval's checks differ.
    from lighteval.tasks.tasks.multiif.instructions import INSTRUCTION_DICT

    lines = response.split("\n")
    trimmed = ["\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    candidates = [response, response.replace("*", ""), *trimmed, *(text.replace("*", "") for text in trimmed)]
    strict, loose = [], []
    for mode, results in (("strict", strict), ("loose", loose)):
        for instruction_id, kwargs in zip(turn["instructions"], turn["kwargs"]):
            instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
            instruction.build_description(**kwargs)
            if mode == "strict":
                results.append(bool(response and instruction.check_following(response)))
            else:
                results.append(any(text.strip() and instruction.check_following(text) for text in candidates))
    return {"index": turn["index"], "strict": strict, "loose": loose}


def aggregate_turn_scores(turn_scores):
    """Pool instructions within each turn, then macro-average the turn metrics.

    A conversation missing turn 3 contributes only to turns 1 and 2. Instruction
    accuracy weights each instruction equally, not each conversation equally.
    """
    result = {}
    components = []
    for index in range(1, 4):
        rows = [turn for turns in turn_scores for turn in turns if turn["index"] == index]
        if not rows:
            continue
        turn_components = []
        for mode in ("strict", "loose"):
            flags = [flag for row in rows for flag in row[mode]]
            prompt_accuracy = mean(int(all(row[mode])) for row in rows)
            instruction_accuracy = sum(flags) / len(flags)
            result[f"turn_{index}_prompt_level_{mode}_acc"] = prompt_accuracy
            result[f"turn_{index}_inst_level_{mode}_acc"] = instruction_accuracy
            turn_components.extend([prompt_accuracy, instruction_accuracy])
        result[f"turn_{index}_overall"] = mean(turn_components)
        components.extend(turn_components)
    # Inspect uses the first metric as its headline.
    return {"overall": mean(components) if components else 0.0, **result}


@metric
def multiif_accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> dict[str, float]:
        return aggregate_turn_scores([sample.score.metadata["turn_scores"] for sample in scores])

    return compute


@scorer(metrics=[multiif_accuracy()])
def multiif_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        messages = [message for message in state.messages if message.role == "assistant"]
        turns = state.metadata["turns"]
        if len(messages) != len(turns):
            raise ValueError(f"Expected {len(turns)} Multi-IF responses, got {len(messages)}")
        turn_scores = [_check_response(message.text.strip(), turn) for message, turn in zip(messages, turns)]
        return Score(
            value=aggregate_turn_scores([turn_scores])["overall"],
            metadata={"turn_scores": turn_scores},
            explanation=f"language={state.metadata['language']}; turns={len(turns)}",
        )

    return score


def _task(name, language=None):
    return LightevalTaskConfig(
        name=name,
        prompt_function=None,
        hf_repo="facebook/Multi-IF",
        hf_subset="default",
        hf_revision=DATASET_REVISION,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="",
        metrics=[],
        generation_size=1024,
        stop_sequence=[],
        sample_fields=record_to_sample,
        filter=(lambda sample: sample.metadata["language"] == language) if language else None,
        solver=[multiif_conversation()],
        scorer=multiif_scorer(),
        version="0.1",
    )


# Each language is scored independently. The registry expands `multiif` into
# these language tasks; English-only evaluation uses `multiif-en`.
TASKS_TABLE = [
    _task(f"multiif-{code}", language)
    for code, language in {
        "en": "English",
        "zh": "Chinese",
        "fr": "French",
        "ru": "Russian",
        "hi": "Hindi",
        "it": "Italian",
        "pt": "Portuguese",
        "es": "Spanish",
    }.items()
]
