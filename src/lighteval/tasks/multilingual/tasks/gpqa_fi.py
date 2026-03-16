"""
name:
GPQA-FI

dataset:
LumiOpen/GPQA-FI

abstract:
Finnish translation of the GPQA Diamond benchmark (Graduate-Level Google-Proof
Q&A). Contains 198 expert-written multiple-choice questions in biology,
physics, and chemistry, professionally post-edited from machine translations
by native Finnish speakers.

languages:
finnish

tags:
knowledge, multiple-choice, qa, science, multilingual

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

random.seed(42)


def record_to_sample(record):
    gold_index = random.randint(0, 3)
    choices = [record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
    choices.insert(gold_index, record["Correct Answer"])
    return Sample(
        input=record["Question"].strip(),
        choices=choices,
        target=ascii_uppercase[gold_index],
    )


def gpqa_fi_prompt(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    instruction = "Vastaa seuraavaan monivalintakysymykseen. Vastauksesi viimeisen rivin tulee olla muotoa: 'Vastaus: $KIRJAIN' (ilman lainausmerkkejä), jossa KIRJAIN on A, B, C tai D. Ajattele vaihe vaiheelta ennen vastaamista.\n\n"

    query = f"Kysymys: {line['Question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, choices)])
    query += "Vastaus: "
    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        choices=ascii_uppercase[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


def gpqa_fi_instruct_prompt(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    instruction = "Vastaa seuraavaan monivalintakysymykseen. Vastauksesi viimeisen rivin tulee olla muotoa: 'Vastaus: $KIRJAIN' (ilman lainausmerkkejä), jossa KIRJAIN on A, B, C tai D. Ajattele vaihe vaiheelta ennen vastaamista."
    query_template = "{Instruction}\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
        Question=line["Question"].strip(),
        Instruction=instruction,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase)[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


# Log-likelihood based evaluation (matches French GPQA pattern)
gpqa_fi = LightevalTaskConfig(
    name="gpqa-fi",
    prompt_function=gpqa_fi_prompt,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="LumiOpen/GPQA-FI",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

# Instruct / generative evaluation (matches English GPQA diamond pattern)
gpqa_fi_diamond = LightevalTaskConfig(
    name="gpqa-fi:diamond",
    prompt_function=gpqa_fi_instruct_prompt,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="LumiOpen/GPQA-FI",
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

TASKS_TABLE = [gpqa_fi, gpqa_fi_diamond]
