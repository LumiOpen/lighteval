# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import asyncio
import importlib
import sys
import types

import pytest


@pytest.fixture()
def main_inspect(monkeypatch):
    registry_info_attr = "__registry_info__"
    registry_params_attr = "__registry_params__"

    def install_module(name):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    inspect_ai = install_module("inspect_ai")
    inspect_ai.__path__ = []
    inspect_ai.Epochs = lambda *args, **kwargs: (args, kwargs)
    inspect_ai.Task = lambda *args, **kwargs: (args, kwargs)
    inspect_ai.eval_set = lambda *args, **kwargs: None
    inspect_ai.task = lambda fn: fn

    inspect_util = install_module("inspect_ai._util")
    inspect_util.__path__ = []
    registry = install_module("inspect_ai._util.registry")

    def has_registry_params(obj):
        return hasattr(obj, registry_params_attr)

    def is_registry_object(obj, type=None):
        info = getattr(obj, registry_info_attr, None)
        return info is not None and (type is None or info.type == type)

    def registry_info(obj):
        return getattr(obj, registry_info_attr)

    def registry_params(obj):
        return getattr(obj, registry_params_attr)

    def set_registry_info(obj, info):
        setattr(obj, registry_info_attr, info)
        return obj

    def set_registry_params(obj, params):
        setattr(obj, registry_params_attr, params)
        return obj

    registry.has_registry_params = has_registry_params
    registry.is_registry_object = is_registry_object
    registry.registry_info = registry_info
    registry.registry_params = registry_params
    registry.set_registry_info = set_registry_info
    registry.set_registry_params = set_registry_params

    dataset = install_module("inspect_ai.dataset")
    dataset.hf_dataset = lambda *args, **kwargs: []

    log = install_module("inspect_ai.log")
    log.bundle_log_dir = lambda *args, **kwargs: None

    scorer = install_module("inspect_ai.scorer")
    scorer.exact = lambda: None

    solver = install_module("inspect_ai.solver")
    solver.generate = lambda *args, **kwargs: "generate"
    solver.solver = lambda fn: fn
    solver.system_message = lambda message: ("system", message)

    pytablewriter = install_module("pytablewriter")
    pytablewriter.MarkdownTableWriter = type(
        "MarkdownTableWriter",
        (),
        {"__init__": lambda self: None, "dumps": lambda self: ""},
    )

    models = install_module("lighteval.models")
    models.__path__ = []
    abstract_model = install_module("lighteval.models.abstract_model")
    abstract_model.InspectAIModelConfig = type(
        "InspectAIModelConfig",
        (),
        {"_parse_args": staticmethod(lambda model_args: model_args)},
    )

    tasks = install_module("lighteval.tasks")
    tasks.__path__ = []
    lighteval_task = install_module("lighteval.tasks.lighteval_task")
    lighteval_task.LightevalTaskConfig = type("LightevalTaskConfig", (), {})

    sys.modules.pop("lighteval.main_inspect", None)
    module = importlib.import_module("lighteval.main_inspect")
    yield module
    sys.modules.pop("lighteval.main_inspect", None)


class Message:
    def __init__(self, role, text):
        self.role = role
        self.content = text

    @property
    def text(self):
        return self.content

    @text.setter
    def text(self, value):
        self.content = value


class Choice:
    def __init__(self, message):
        self.message = message


class Output:
    def __init__(self, completion, choices):
        self.completion = completion
        self.choices = choices


class Score:
    def __init__(self, value, answer=None, explanation=None, metadata=None):
        self.value = value
        self.answer = answer
        self.explanation = explanation
        self.metadata = metadata


def test_reasoning_postprocessor_strips_assistant_turns_and_output_choices(main_inspect):
    state = types.SimpleNamespace(
        messages=[
            Message("user", "Question </think> stays untouched"),
            Message("assistant", "First turn reasoning </think> First answer"),
            Message("assistant", "<think>Second turn reasoning</think>Second answer"),
        ],
        output=Output(
            completion="Final reasoning </think> Final answer",
            choices=[
                Choice(Message("assistant", "<think>Choice reasoning</think>Choice answer")),
                Choice(Message("assistant", "Choice two reasoning </think> Choice two answer")),
            ],
        ),
    )

    postprocessor = main_inspect.reasoning_tag_postprocessor(tag_pairs=[("<think>", "</think>")])

    asyncio.run(postprocessor(state, generate_fn=None))

    assert state.messages[0].text == "Question </think> stays untouched"
    assert state.messages[1].text == " First answer"
    assert state.messages[2].text == "Second answer"
    assert state.output.completion == " Final answer"
    assert state.output.choices[0].message.text == "Choice answer"
    assert state.output.choices[1].message.text == " Choice two answer"


def test_reasoning_scorer_wrapper_strips_returned_score_fields_and_preserves_registry(main_inspect):
    async def scorer(state, target):
        return Score(
            value={"judge": "<think>judge reasoning</think>C", "nested": ["raw judge </think> D"]},
            answer="answer reasoning </think> answer",
            explanation="<think>judge rationale</think>judge answer",
            metadata={"raw": "metadata reasoning </think> metadata"},
        )

    scorer.__registry_info__ = types.SimpleNamespace(
        type="scorer",
        name="custom_scorer",
        metadata={"metrics": ["accuracy"]},
    )
    scorer.__registry_params__ = {"ignore_case": True}

    wrapped = main_inspect._wrap_reasoning_tag_scorer(scorer, tag_pairs=[("<think>", "</think>")])
    score = asyncio.run(wrapped(state=None, target=None))

    assert score.value == {"judge": "C", "nested": [" D"]}
    assert score.answer == " answer"
    assert score.explanation == "judge answer"
    assert score.metadata == {"raw": " metadata"}
    assert wrapped.__registry_info__ is scorer.__registry_info__
    assert wrapped.__registry_params__ == {"ignore_case": True}


def test_inspect_task_passes_pinned_revision_to_evaluation_and_fewshot_datasets(main_inspect, monkeypatch):
    calls = []

    def hf_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(main_inspect, "hf_dataset", hf_dataset)
    config = types.SimpleNamespace(
        name="revision_test",
        sample_fields=lambda record: record,
        hf_repo="org/dataset",
        hf_subset="default",
        hf_revision="pinned-revision",
        evaluation_splits=["test"],
        filter=None,
        solver=None,
        scorer=None,
        num_fewshots=1,
        sample_to_fewshot=lambda sample: str(sample),
    )

    main_inspect.get_inspect_ai_task(config)

    assert len(calls) == 2
    assert calls[0][1]["revision"] == "pinned-revision"
    assert calls[1][1]["revision"] == "pinned-revision"
