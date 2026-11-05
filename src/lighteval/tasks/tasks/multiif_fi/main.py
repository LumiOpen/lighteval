"""
name:
Multi-IF Finnish (multiif-fi)

dataset:
ezosa/multiif-fi

abstract:
Three-turn Finnish instruction-following, parallel to multiif-en. The prompts and the
verifiable constraint kwargs (keywords, end-phrases, first-words, section-splitters,
forbidden-words) are translated to Finnish; a universal "respond in Finnish"
(language:response_language -> fi) constraint is added to the base records, while the 46
"respond in language X" records keep their original target language. Scored with a
Finnish-adapted verifier (multiif_fi.instructions): langdetect==fi case checks and
simplemma-lemmatised keyword matching so inflected Finnish forms verify correctly.

The Finnish data is a constraint-aware translation of the English split of
facebook/Multi-IF (909 conversations).

languages:
finnish

tags:
instruction-following, multi-turn
"""

from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from lighteval.tasks.lighteval_task import LightevalTaskConfig

# Reuse the language-agnostic pieces from the English task.
from lighteval.tasks.tasks.multiif.main import (
    aggregate_turn_scores,
    multiif_accuracy,
    multiif_conversation,
    record_to_sample,
)


def _check_response(response, turn):
    # Finnish-adapted verifier (langdetect==fi case checks + lemmatised keyword matching).
    from lighteval.tasks.tasks.multiif_fi.instructions import INSTRUCTION_DICT

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


@scorer(metrics=[multiif_accuracy()])
def multiif_fi_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        messages = [m for m in state.messages if m.role == "assistant"]
        turns = state.metadata["turns"]
        if len(messages) != len(turns):
            raise ValueError(f"Expected {len(turns)} Multi-IF responses, got {len(messages)}")
        turn_scores = [_check_response(m.text.strip(), turn) for m, turn in zip(messages, turns)]
        return Score(
            value=aggregate_turn_scores([turn_scores])["overall"],
            metadata={"turn_scores": turn_scores},
            explanation=f"language={state.metadata['language']}; turns={len(turns)}",
        )

    return score


multiif_fi = LightevalTaskConfig(
    name="multiif-fi",
    prompt_function=None,
    hf_repo="ezosa/multiif-fi",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    metrics=[],
    generation_size=1024,
    stop_sequence=[],
    sample_fields=record_to_sample,
    filter=(lambda sample: sample.metadata["language"] == "Finnish"),
    solver=[multiif_conversation()],
    scorer=multiif_fi_scorer(),
    version="0.1",
)

TASKS_TABLE = [multiif_fi]
