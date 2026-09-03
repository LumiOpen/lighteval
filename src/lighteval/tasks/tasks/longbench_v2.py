"""
name:
LongBench v2

dataset:
THUDM/LongBench-v2

abstract:
Long-context multiple-choice benchmark covering single-document QA,
multi-document QA, long in-context learning, long dialogue understanding,
code repository understanding, and long structured-data understanding.

This implementation uses the official zero-shot prompt and strict answer
parser and issues one generation request per sample. Reasoning behavior is
controlled by the model and backend configuration, so the task supports both
reasoning and non-reasoning runs. It does not orchestrate the upstream
two-request CoT protocol.

languages:
english, chinese

tags:
long-context, reasoning, multiple-choice

paper:
https://arxiv.org/abs/2412.15204

source:
https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench-v2
"""

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


OFFICIAL_REPO_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
OFFICIAL_DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
OFFICIAL_DATASET_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
OFFICIAL_PROMPT_SHA256 = "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950"
EXPECTED_SAMPLES = 503
DEFAULT_MAX_MODEL_LEN = 65536
DEFAULT_MAX_TOKENS = 32768
DEFAULT_CHAT_TEMPLATE_MARGIN_TOKENS = 2048
PARSER_VERSION = "longbench-v2-official-strict-v1"
TRUNCATION_POLICY = "middle-token-truncation-v1"
PROMPT_VARIANT = "official-0shot"
REQUEST_MODE = "single-generation"

OFFICIAL_ZERO_SHOT_PROMPT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

DATASET_FIELDS = (
    "_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
    "context",
)


def extract_answer(response: str) -> str | None:
    """Apply the official LongBench v2 strict answer parser."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def render_prompt(record: dict[str, Any]) -> str:
    """Render the official zero-shot prompt for one dataset record."""
    missing = [field for field in DATASET_FIELDS if field not in record]
    if missing:
        raise ValueError(f"LongBench v2 record is missing fields: {missing}")
    if record["answer"] not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid LongBench v2 answer: {record['answer']!r}")

    replacements = {
        "$DOC$": str(record["context"]).strip(),
        "$Q$": str(record["question"]).strip(),
        "$C_A$": str(record["choice_A"]).strip(),
        "$C_B$": str(record["choice_B"]).strip(),
        "$C_C$": str(record["choice_C"]).strip(),
        "$C_D$": str(record["choice_D"]).strip(),
    }
    prompt = OFFICIAL_ZERO_SHOT_PROMPT
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt


def max_prompt_tokens(max_model_len: int, max_tokens: int) -> int:
    """Return the raw prompt budget after generation and chat-template reserves."""
    budget = max_model_len - max_tokens - DEFAULT_CHAT_TEMPLATE_MARGIN_TOKENS
    if budget <= 0:
        raise ValueError(
            f"MAX_MODEL_LEN ({max_model_len}) must be larger than MAX_TOKENS ({max_tokens}) plus the "
            f"LongBench v2 chat-template margin ({DEFAULT_CHAT_TEMPLATE_MARGIN_TOKENS})"
        )
    return budget


def _positive_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def _tokenizer():
    tokenizer_name = os.environ.get("LONG_BENCH_TOKENIZER") or os.environ.get("MODEL")
    if not tokenizer_name:
        raise ValueError("LongBench v2 requires MODEL or LONG_BENCH_TOKENIZER for prompt truncation")
    return _load_tokenizer(tokenizer_name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _verify_dataset_snapshot() -> None:
    from huggingface_hub import hf_hub_download

    dataset_path = Path(
        hf_hub_download(
            repo_id="THUDM/LongBench-v2",
            filename="data.json",
            repo_type="dataset",
            revision=OFFICIAL_DATASET_REVISION,
        )
    )
    actual_sha256 = _file_sha256(dataset_path)
    if actual_sha256 != OFFICIAL_DATASET_SHA256:
        raise ValueError(
            f"Pinned LongBench v2 dataset hash mismatch: expected {OFFICIAL_DATASET_SHA256}, got {actual_sha256}"
        )


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Unexpected batched chat-template token output")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
        raise ValueError("Chat template did not return token IDs")
    return value


def _middle_truncate(tokenizer: Any, input_ids: list[int], limit: int) -> str:
    left = limit // 2
    right = limit - left
    return tokenizer.decode(input_ids[:left] + input_ids[-right:], skip_special_tokens=True)


def truncate_prompt(
    tokenizer: Any,
    prompt: str,
    prompt_token_budget: int,
) -> tuple[str, dict[str, Any]]:
    """Middle-truncate the raw prompt, leaving a fixed margin for chat formatting."""
    if prompt_token_budget <= 0:
        raise ValueError("LongBench v2 prompt token budget must be positive")
    input_ids = _token_ids(tokenizer.encode(prompt))
    original_tokens = len(input_ids)
    was_truncated = original_tokens > prompt_token_budget
    truncated_tokens = original_tokens

    if was_truncated:
        raw_limit = min(original_tokens - 1, prompt_token_budget)
        while raw_limit > 0:
            prompt = _middle_truncate(tokenizer, input_ids, raw_limit)
            truncated_tokens = len(_token_ids(tokenizer.encode(prompt)))
            if truncated_tokens <= prompt_token_budget:
                break
            raw_limit -= max(truncated_tokens - prompt_token_budget, 1)
        else:
            raise ValueError("Could not fit the LongBench v2 raw prompt in the configured token budget")

    return prompt, {
        "input_tokens_before_truncation": original_tokens,
        "input_tokens_after_truncation": truncated_tokens,
        "max_prompt_tokens": prompt_token_budget,
        "chat_template_margin_tokens": DEFAULT_CHAT_TEMPLATE_MARGIN_TOKENS,
        "was_truncated": was_truncated,
        "truncation_policy": TRUNCATION_POLICY,
    }


def _prepare_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _verify_dataset_snapshot()
    max_model_len = _positive_env_int("MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN)
    max_tokens = _positive_env_int("MAX_TOKENS", DEFAULT_MAX_TOKENS)
    prompt, token_metadata = truncate_prompt(
        _tokenizer(),
        render_prompt(record),
        max_prompt_tokens(max_model_len, max_tokens),
    )
    metadata = {
        "_id": str(record["_id"]),
        "domain": str(record["domain"]),
        "sub_domain": str(record["sub_domain"]),
        "difficulty": str(record["difficulty"]),
        "length": str(record["length"]),
        "prompt_variant": PROMPT_VARIANT,
        "request_mode": REQUEST_MODE,
        "parser_version": PARSER_VERSION,
        "prompt_sha256": OFFICIAL_PROMPT_SHA256,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "dataset_sha256": OFFICIAL_DATASET_SHA256,
        "upstream_revision": OFFICIAL_REPO_REVISION,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        **token_metadata,
    }
    return prompt, metadata


def record_to_sample(record: dict[str, Any]) -> Sample:
    prompt, metadata = _prepare_record(record)
    return Sample(input=prompt, target=record["answer"], id=str(record["_id"]), metadata=metadata)


def longbench_v2_prompt(record: dict[str, Any], task_name: str = "") -> Doc:
    prompt, metadata = _prepare_record(record)
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[record["answer"]],
        gold_index=0,
        specific=metadata,
    )


def _score_is_correct(sample_score: SampleScore) -> bool:
    return bool(sample_score.score.value)


def _score_is_parsed(sample_score: SampleScore) -> bool:
    return bool((sample_score.score.metadata or {}).get("parsed"))


def _mean(values: list[float | int | bool]) -> float:
    return fmean(values) if values else 0.0


@metric
def strict_accuracy() -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        return _mean([_score_is_correct(score) for score in scores])

    return metric_fn


@metric
def invalid_output_rate() -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        return _mean([not _score_is_parsed(score) for score in scores])

    return metric_fn


@metric
def parsed_only_accuracy() -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        parsed_scores = [score for score in scores if _score_is_parsed(score)]
        return _mean([_score_is_correct(score) for score in parsed_scores])

    return metric_fn


@metric
def compensated_accuracy() -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        values = [
            1.0 if _score_is_correct(score) else 0.25 if not _score_is_parsed(score) else 0.0 for score in scores
        ]
        return _mean(values)

    return metric_fn


def _metadata_accuracy(scores: list[SampleScore], field: str, expected: str) -> float:
    selected = [score for score in scores if str((score.sample_metadata or {}).get(field)) == expected]
    return _mean([_score_is_correct(score) for score in selected])


def _metadata_accuracy_metric(field: str, expected: str) -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        return _metadata_accuracy(scores, field, expected)

    return metric_fn


@metric
def difficulty_easy_accuracy() -> Metric:
    return _metadata_accuracy_metric("difficulty", "easy")


@metric
def difficulty_hard_accuracy() -> Metric:
    return _metadata_accuracy_metric("difficulty", "hard")


@metric
def length_short_accuracy() -> Metric:
    return _metadata_accuracy_metric("length", "short")


@metric
def length_medium_accuracy() -> Metric:
    return _metadata_accuracy_metric("length", "medium")


@metric
def length_long_accuracy() -> Metric:
    return _metadata_accuracy_metric("length", "long")


LONG_BENCH_V2_INSPECT_METRICS = [
    strict_accuracy(),
    stderr(),
    invalid_output_rate(),
    parsed_only_accuracy(),
    compensated_accuracy(),
    difficulty_easy_accuracy(),
    difficulty_hard_accuracy(),
    length_short_accuracy(),
    length_medium_accuracy(),
    length_long_accuracy(),
]


@scorer(metrics=LONG_BENCH_V2_INSPECT_METRICS)
def longbench_v2_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        prediction = extract_answer(response)
        return Score(
            value=int(prediction == target.text),
            answer=prediction,
            explanation=f"Strict parser prediction: {prediction or 'invalid'}",
            metadata={"parsed": prediction is not None, "prediction": prediction},
        )

    return score


class LongBenchV2StrictAccuracy(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> int:
        prediction = extract_answer(model_response.final_text[0])
        return int(prediction == doc.choices[doc.gold_index])


longbench_v2_legacy_metric = SampleLevelMetric(
    metric_name="strict_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=LongBenchV2StrictAccuracy(),
    corpus_level_fn=_mean,
)


if hashlib.sha256(OFFICIAL_ZERO_SHOT_PROMPT.encode("utf-8")).hexdigest() != OFFICIAL_PROMPT_SHA256:
    raise RuntimeError("Embedded LongBench v2 official prompt hash mismatch")


longbench_v2 = LightevalTaskConfig(
    name="longbench2",
    prompt_function=longbench_v2_prompt,
    hf_repo="THUDM/LongBench-v2",
    hf_subset="default",
    hf_revision=OFFICIAL_DATASET_REVISION,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    generation_size=DEFAULT_MAX_TOKENS,
    metrics=[longbench_v2_legacy_metric],
    version=1,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=longbench_v2_scorer(),
)


TASKS_TABLE = [longbench_v2]
