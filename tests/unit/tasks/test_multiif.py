import asyncio
import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import SampleScore, Score, Target

from lighteval.tasks.registry import Registry
from lighteval.tasks.tasks.multiif.main import (
    TASKS_TABLE,
    _check_response,
    aggregate_turn_scores,
    multiif_accuracy,
    multiif_conversation,
    multiif_scorer,
    record_to_sample,
)


def _record():
    record = {"key": "example:en", "language": "English", "responses": "unused reference"}
    for index in range(1, 4):
        record[f"turn_{index}_prompt"] = json.dumps({"role": "user", "content": f"turn {index}"})
        record[f"turn_{index}_instruction_id_list"] = '["change_case:english_lowercase"]'
        record[f"turn_{index}_kwargs"] = '["{}"]'
    return record


def test_generated_history_and_missing_turn():
    record = _record()
    record["turn_3_prompt"] = "None"
    sample = record_to_sample(record)
    state = SimpleNamespace(metadata=sample.metadata, messages=[ChatMessageUser(content=sample.input)])
    histories = []

    async def generate(state):
        histories.append([(message.role, message.text) for message in state.messages])
        state.messages.append(ChatMessageAssistant(content=f"generated answer {len(histories)}"))
        return state

    asyncio.run(multiif_conversation()(state, generate))
    assert histories == [
        [("user", "turn 1")],
        [("user", "turn 1"), ("assistant", "generated answer 1"), ("user", "turn 2")],
    ]
    assert sample.id == "example:en"


def test_nested_kwargs_and_cumulative_constraints():
    record = _record()
    record["turn_2_instruction_id_list"] = '["change_case:english_lowercase", "startend:end_checker"]'
    record["turn_2_kwargs"] = json.dumps(["{}", json.dumps({"end_phrase": "done"})])
    turn = record_to_sample(record).metadata["turns"][1]
    assert _check_response("this is an english response and it is done", turn)["strict"] == [True, True]
    assert _check_response("This is an english response and it is done", turn)["strict"] == [False, True]


def test_loose_checks_preserve_original_whitespace_and_markdown_rules():
    turn = {"index": 1, "instructions": ["startend:quotation"], "kwargs": [{}]}
    result = _check_response('preface\n"answer"\nsuffix', turn)
    assert result["strict"] == [False]
    assert result["loose"] == [True]


def test_instruction_weighting_and_turn_macro_average():
    first = [{"index": 1, "strict": [True], "loose": [True]}, {"index": 2, "strict": [False], "loose": [False]}]
    second = [{"index": 1, "strict": [False, False, False], "loose": [False, False, False]}]
    result = aggregate_turn_scores([first, second])
    assert result["turn_1_prompt_level_strict_acc"] == 0.5
    assert result["turn_1_inst_level_strict_acc"] == 0.25
    assert result["turn_1_overall"] == 0.375
    assert result["overall"] == 0.1875
    scores = [SampleScore(score=Score(value=1, metadata={"turn_scores": turns})) for turns in [first, second]]
    assert multiif_accuracy()(scores) == result


def test_scorer_scores_every_generated_turn_and_rejects_missing_answers():
    sample = record_to_sample(_record())
    state = SimpleNamespace(
        metadata=sample.metadata,
        messages=[
            ChatMessageAssistant(content="lowercase"),
            ChatMessageAssistant(content="UPPERCASE"),
            ChatMessageAssistant(content="lowercase again"),
        ],
    )
    score = asyncio.run(multiif_scorer()(state, Target("")))
    assert score.value == pytest.approx(2 / 3)
    assert len(score.metadata["turn_scores"]) == 3
    state.messages.pop()
    with pytest.raises(ValueError, match="Expected 3"):
        asyncio.run(multiif_scorer()(state, Target("")))


def test_language_variants_and_bad_input():
    tasks = {task.name: task for task in TASKS_TABLE}
    sample = record_to_sample(_record())
    assert tasks["multiif-en"].filter(sample)
    assert not tasks["multiif-fr"].filter(sample)
    assert all(task.filter is not None for task in TASKS_TABLE)
    record = _record()
    record["turn_1_kwargs"] = "[]"
    with pytest.raises(ValueError, match="Invalid Multi-IF turn"):
        record_to_sample(record)


@pytest.mark.parametrize("task_name", ["multiif-en", "multiif-es"])
def test_language_selection_and_group_expansion(task_name):
    registry = object.__new__(Registry)
    assert registry._get_full_task_list_from_input_string(task_name) == [task_name]
    assert set(registry._get_full_task_list_from_input_string("multiif")) == {
        "multiif-en",
        "multiif-zh",
        "multiif-fr",
        "multiif-ru",
        "multiif-hi",
        "multiif-it",
        "multiif-pt",
        "multiif-es",
    }
