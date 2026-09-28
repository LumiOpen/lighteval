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

import numpy as np
from inspect_ai.dataset import Sample
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics import Metrics, multichoice_scorer
from lighteval.metrics.metrics_sample import PassAtK
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


GPQA_FI_INSTRUCTION = "Vastaa seuraavaan monivalintakysymykseen. Vastauksesi viimeisen rivin tulee olla muotoa: 'Vastaus: $KIRJAIN' (ilman lainausmerkkejä), jossa KIRJAIN on A, B, C tai D. Ajattele vaihe vaiheelta ennen vastaamista."

# inspect-ai backend: `multiple_choice` renders {choices} as "A) ...\nB) ...", matching gpqa_fi_instruct_prompt.
# Its built-in `choice` scorer only parses English "ANSWER: X", so pair it with a Finnish-aware scorer.
GPQA_FI_INSPECT_TEMPLATE = GPQA_FI_INSTRUCTION + "\n\n{question}\n\n{choices}"
GPQA_FI_INSPECT_SOLVER = [multiple_choice(template=GPQA_FI_INSPECT_TEMPLATE, cache=True)]
GPQA_FI_INSPECT_SCORER = multichoice_scorer(language=Language.FINNISH)

# Native backend: Finnish counterpart of Metrics.gpqa_instruct_pass_at_k (which extracts with English anchors).
GPQA_FI_PASS_AT_1 = SampleLevelMetric(
    metric_name="gpqa_pass@k",
    sample_level_fn=PassAtK(
        sample_scoring_function=MultilingualExtractiveMatchMetric(
            language=Language.FINNISH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            precision=6,
        ),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)(sample_params={"k": 1})


random.seed(42)


def record_to_sample(record):
    gold_index = random.randint(0, 3)
    choices = [record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
    choices.insert(gold_index, record["Correct Answer"])
    return Sample(
        input=record["Question"].strip(),
        choices=[choice.strip() for choice in choices],
        target=ascii_uppercase[gold_index],
    )


def gpqa_fi_prompt(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])

    instruction = GPQA_FI_INSTRUCTION + "\n\n"

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
    instruction = GPQA_FI_INSTRUCTION
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
    solver=GPQA_FI_INSPECT_SOLVER,
    scorer=GPQA_FI_INSPECT_SCORER,
    hf_repo="LumiOpen/GPQA-FI",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=1,
)

# Instruct / generative evaluation (matches English GPQA diamond pattern)
gpqa_fi_diamond = LightevalTaskConfig(
    name="gpqa-fi:diamond",
    prompt_function=gpqa_fi_instruct_prompt,
    sample_fields=record_to_sample,
    solver=GPQA_FI_INSPECT_SOLVER,
    scorer=GPQA_FI_INSPECT_SCORER,
    hf_repo="LumiOpen/GPQA-FI",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[GPQA_FI_PASS_AT_1],
    stop_sequence=[],
    version=2,
)

TASKS_TABLE = [gpqa_fi, gpqa_fi_diamond]
