"""
name:
mGPQA

dataset:
LumiOpen/mGPQA

abstract:
Multilingual GPQA Diamond is a translated version of GPQA Diamond,
a dataset of expert-written multiple-choice questions in biology,
physics, and chemistry designed to test graduate-level reasoning.
Currently contains automatic Finnish translations, with manual
translations forthcoming.

languages:
finnish

tags:
biology, chemistry, graduate-level, multiple-choice, physics, qa, reasoning, science, multilingual

paper:
https://arxiv.org/abs/2311.12022
"""

import random
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def record_to_sample_fi(record):
    gold_index = random.randint(0, 3)
    choices = [
        record["Incorrect Answer 1_fi"],
        record["Incorrect Answer 2_fi"],
        record["Incorrect Answer 3_fi"],
    ]
    choices.insert(gold_index, record["Correct Answer_fi"])
    return Sample(
        input=record["Question_fi"].strip(),
        choices=choices,
        target=ascii_uppercase[gold_index],
    )


def mgpqa_prompt_fi(line, task_name: str = None):
    GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
    gold_index = random.randint(0, 3)
    choices = [
        line["Incorrect Answer 1_fi"],
        line["Incorrect Answer 2_fi"],
        line["Incorrect Answer 3_fi"],
    ]
    choices.insert(gold_index, line["Correct Answer_fi"])

    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        Question=line["Question_fi"].strip(),
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[:len(choices)],
        gold_index=gold_index,
        instruction=query,
    )


mgpqa_diamond_fi = LightevalTaskConfig(
    name="mgpqa:fi_diamond",
    prompt_function=mgpqa_prompt_fi,
    sample_fields=record_to_sample_fi,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="LumiOpen/mGPQA",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 1})],
    stop_sequence=[],
    version=1,
)

TASKS_TABLE = [
    mgpqa_diamond_fi,
]
