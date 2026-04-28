"""
name:
mAIME2025, mAIME2026

dataset:
LumiOpen/mAIME2025, LumiOpen/mAIME2026

abstract:
The Multilingual AIME (mAIME) datasets are multilingual versions of the
AIME (American Invitational Mathematics Examination) problems,
professionally translated into European languages. Each dataset contains
all 30 problems from AIME I and AIME II for the respective year, translated
and human-reviewed by native speakers to preserve mathematical accuracy and
LaTeX formatting.

languages:
danish, finnish

tags:
math, multilingual, reasoning

paper:
https://maa.org/aime-thresholds-are-available/
"""

from textwrap import dedent

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Per-language prompt templates
MATH_PROMPT_TEMPLATES = {
    "en": dedent("""
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{prompt}
"""),
    "fi": dedent("""
Ratkaise seuraava matemaattinen tehtävä tehokkaasti ja selkeästi.
Vastauksesi viimeisen rivin tulee olla seuraavassa muodossa:
'Näin ollen lopullinen vastaus on: $\\boxed{{ANSWER}}$. Toivottavasti se on oikein'
(ilman lainausmerkkejä), jossa ANSWER on pelkästään lopullinen luku tai lauseke,
joka ratkaisee tehtävän. Ajattele vaiheittain ennen vastaamista.

{prompt}
"""),
    "da": dedent("""
Løs følgende matematiske problem korrekt og effektivt.
Den sidste linje i dit svar skal være i følgende format:
'Derfor er det endelige svar: $\\boxed{{ANSWER}}$. Jeg håber, det er korrekt'
(uden anførselstegn), hvor ANSWER kun er den endelige løsning. Tænk trin for trin, før du svarer.

{prompt}
"""),
}


def record_to_sample(record):
    return Sample(input=record["question"], target=record["solution"])


def record_to_sample_en(record):
    return Sample(input=record["problem"], target=record["answer"])


def _maime_prompt_fn(lang: str):
    template = MATH_PROMPT_TEMPLATES[lang]

    def maime_prompt(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=template.format(prompt=line["question"]),
            choices=[line["solution"]],
            gold_index=0,
        )

    return maime_prompt


def _maime_prompt_fn_en(lang: str):
    template = MATH_PROMPT_TEMPLATES[lang]

    def maime_prompt_en(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=template.format(prompt=line["problem"]),
            choices=[line["answer"]],
            gold_index=0,
        )

    return maime_prompt_en


# English tasks — AIME 2025
maime25_en = LightevalTaskConfig(
    name="maime25:en",
    prompt_function=_maime_prompt_fn_en("en"),
    sample_fields=record_to_sample_en,
    solver=[prompt_template(MATH_PROMPT_TEMPLATES["en"]), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime25_en_avg = LightevalTaskConfig(
    name="maime25_avg:en",
    prompt_function=_maime_prompt_fn_en("en"),
    sample_fields=record_to_sample_en,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_en_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:en",
    prompt_function=_maime_prompt_fn_en("en"),
    sample_fields=record_to_sample_en,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

# Danish tasks
maime25_da = LightevalTaskConfig(
    name="maime25:da",
    prompt_function=_maime_prompt_fn("da"),
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATES["da"]), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime25_da_avg = LightevalTaskConfig(
    name="maime25_avg:da",
    prompt_function=_maime_prompt_fn("da"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_da_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:da",
    prompt_function=_maime_prompt_fn("da"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

# Finnish tasks
maime25_fi = LightevalTaskConfig(
    name="maime25:fi",
    prompt_function=_maime_prompt_fn("fi"),
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATES["fi"]), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime25_fi_avg = LightevalTaskConfig(
    name="maime25_avg:fi",
    prompt_function=_maime_prompt_fn("fi"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_fi_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:fi",
    prompt_function=_maime_prompt_fn("fi"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

# English tasks — AIME 2026
maime26_en = LightevalTaskConfig(
    name="maime26:en",
    prompt_function=_maime_prompt_fn_en("en"),
    sample_fields=record_to_sample_en,
    solver=[prompt_template(MATH_PROMPT_TEMPLATES["en"]), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="math-ai/aime26",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime26_en_avg = LightevalTaskConfig(
    name="maime26_avg:en",
    prompt_function=_maime_prompt_fn_en("en"),
    sample_fields=record_to_sample_en,
    hf_repo="math-ai/aime26",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime26_en_gpassk = LightevalTaskConfig(
    name="maime26_gpassk:en",
    prompt_function=_maime_prompt_fn_en("en"),
    sample_fields=record_to_sample_en,
    hf_repo="math-ai/aime26",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

# Danish tasks — AIME 2026
maime26_da = LightevalTaskConfig(
    name="maime26:da",
    prompt_function=_maime_prompt_fn("da"),
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATES["da"]), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime26_da_avg = LightevalTaskConfig(
    name="maime26_avg:da",
    prompt_function=_maime_prompt_fn("da"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime26_da_gpassk = LightevalTaskConfig(
    name="maime26_gpassk:da",
    prompt_function=_maime_prompt_fn("da"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

# Finnish tasks — AIME 2026
maime26_fi = LightevalTaskConfig(
    name="maime26:fi",
    prompt_function=_maime_prompt_fn("fi"),
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATES["fi"]), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime26_fi_avg = LightevalTaskConfig(
    name="maime26_avg:fi",
    prompt_function=_maime_prompt_fn("fi"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime26_fi_gpassk = LightevalTaskConfig(
    name="maime26_gpassk:fi",
    prompt_function=_maime_prompt_fn("fi"),
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

TASKS_TABLE = [
    maime25_en,
    maime25_en_avg,
    maime25_en_gpassk,
    maime25_da,
    maime25_da_avg,
    maime25_da_gpassk,
    maime25_fi,
    maime25_fi_avg,
    maime25_fi_gpassk,
    maime26_en,
    maime26_en_avg,
    maime26_en_gpassk,
    maime26_da,
    maime26_da_avg,
    maime26_da_gpassk,
    maime26_fi,
    maime26_fi_avg,
    maime26_fi_gpassk,
]
