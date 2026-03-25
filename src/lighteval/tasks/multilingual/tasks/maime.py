"""
name:
mAIME2025

dataset:
LumiOpen/mAIME2025

abstract:
The Multilingual AIME 2025 (mAIME2025) is a multilingual version of the
2025 AIME (American Invitational Mathematics Examination) problems,
professionally translated into European languages. This dataset contains
all 30 problems from AIME I and AIME II 2025, translated and
human-reviewed by native speakers to preserve mathematical accuracy and
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

TASKS_TABLE = [
    maime25_da,
    maime25_da_avg,
    maime25_da_gpassk,
    maime25_fi,
    maime25_fi_avg,
    maime25_fi_gpassk,
]
