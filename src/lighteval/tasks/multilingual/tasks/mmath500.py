"""
name:
mMATH-500

dataset:
LumiOpen/MATH-500_mt

abstract:
Multilingual MATH-500 benchmark, a subset of 500 problems from the MATH
benchmark that OpenAI created in their Let's Verify Step by Step paper.
Contains the original English problems and Finnish translations produced
with Claude Opus 4.5. Supports configurable scorer model via env vars.

languages:
english, finnish

tags:
math, reasoning, multilingual

paper:
https://arxiv.org/abs/2305.20050
"""

import os

from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


MATH_QUERY_TEMPLATES = {
    "en": """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip(),
    "fi": """
Ratkaise seuraava tehtävä. Vastauksesi viimeisen rivin TÄYTYY olla seuraavassa muodossa:
"ANSWER: $ANSWER" (ilman lainausmerkkejä), jossa $ANSWER on lopullinen vastaus. Ajattele vaiheittain ennen vastaamista.

{prompt}
""".strip(),
}


def _get_scorer_model():
    """Resolve the scorer model from environment variables if available.

    When SCORER_MODEL_BASE_URL is set (e.g. by the eval harness which starts
    a dedicated scorer vLLM server), connect to it via the OpenAI-compatible
    API. Otherwise return None so model_graded_fact uses the eval model.
    """
    base_url = os.environ.get("SCORER_MODEL_BASE_URL")
    if base_url:
        model_name = os.environ.get("SCORER_MODEL_PATH", "Qwen/Qwen3.5-9B")
        return get_model(
            f"openai-api/scorer/{model_name}",
            config=GenerateConfig(
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "inspectai"),
        )
    return None


def _mmath500_prompt_fn(lang: str):
    template = MATH_QUERY_TEMPLATES[lang]

    def mmath500_prompt(line, task_name: str = None):
        query = template.format(prompt=line["problem"])
        return Doc(
            task_name=task_name,
            query=query,
            choices=[f"ANSWER: {line['answer']}"],
            gold_index=0,
        )

    return mmath500_prompt


def record_to_sample(record):
    query = record["problem"]
    target = record["answer"]
    return Sample(input=query, target=target)


mmath500_en = LightevalTaskConfig(
    name="mmath500:en",
    prompt_function=_mmath500_prompt_fn("en"),
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
    ],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATES["en"]), generate(cache=True)],
    scorer=model_graded_fact(model=_get_scorer_model()),
)

mmath500_fi = LightevalTaskConfig(
    name="mmath500:fi",
    prompt_function=_mmath500_prompt_fn("fi"),
    hf_repo="LumiOpen/MATH-500_mt",
    hf_subset="default",
    hf_avail_splits=["fi"],
    evaluation_splits=["fi"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
    ],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATES["fi"]), generate(cache=True)],
    scorer=model_graded_fact(model=_get_scorer_model()),
)

TASKS_TABLE = [
    mmath500_en,
    mmath500_fi,
]
