import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import SampleScore, Score, Target

from lighteval.tasks.tasks import longbench_v2


class CharacterTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token) for token in token_ids)


def sample_record():
    return {
        "_id": "sample-id",
        "domain": "Long In-context Learning",
        "sub_domain": "New language translation",
        "difficulty": "hard",
        "length": "long",
        "question": "Which option is correct?",
        "choice_A": "alpha",
        "choice_B": "beta",
        "choice_C": "gamma",
        "choice_D": "delta",
        "answer": "C",
        "context": "0123456789",
    }


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The correct answer is (A)", "A"),
        ("The correct answer is B", "B"),
        ("**The correct answer is (C)**", "C"),
        ("The answer is (D)", None),
        ("the correct answer is (A)", None),
    ],
)
def test_extract_answer_uses_official_strict_parser(response, expected):
    assert longbench_v2.extract_answer(response) == expected


def test_official_prompt_hash_and_rendering():
    prompt = longbench_v2.render_prompt(sample_record())

    assert hashlib.sha256(longbench_v2.OFFICIAL_ZERO_SHOT_PROMPT.encode()).hexdigest() == (
        longbench_v2.OFFICIAL_PROMPT_SHA256
    )
    assert "$DOC$" not in prompt
    assert "0123456789" in prompt
    assert "Which option is correct?" in prompt
    assert "(C) gamma" in prompt


def test_dataset_snapshot_hash_mismatch_is_rejected(monkeypatch, tmp_path):
    dataset_path = tmp_path / "data.json"
    dataset_path.write_text("not the official dataset", encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(dataset_path))
    longbench_v2._verify_dataset_snapshot.cache_clear()

    with pytest.raises(ValueError, match="dataset hash mismatch"):
        longbench_v2._verify_dataset_snapshot()

    longbench_v2._verify_dataset_snapshot.cache_clear()


def test_middle_truncation_preserves_ends():
    tokenizer = CharacterTokenizer()
    prompt = "abcdefghijklmnopqrstuvwxyz"

    truncated, metadata = longbench_v2.truncate_prompt(tokenizer, prompt, 10)

    assert truncated == "abcde" + "vwxyz"
    assert metadata == {
        "input_tokens_before_truncation": 26,
        "input_tokens_after_truncation": 10,
        "max_prompt_tokens": 10,
        "chat_template_margin_tokens": 2048,
        "was_truncated": True,
        "truncation_policy": longbench_v2.TRUNCATION_POLICY,
    }


def test_record_to_sample_derives_prompt_budget_with_chat_template_margin(monkeypatch):
    monkeypatch.setenv("MODEL", "unused-in-test")
    monkeypatch.setenv("MAX_MODEL_LEN", "4096")
    monkeypatch.setenv("MAX_TOKENS", "1024")
    monkeypatch.setenv("VLLM_SERVER_ARGS", "malformed backend-specific settings are ignored by the task")
    monkeypatch.setattr(longbench_v2, "_tokenizer", lambda: CharacterTokenizer())
    monkeypatch.setattr(longbench_v2, "_verify_dataset_snapshot", lambda: None)

    sample = longbench_v2.record_to_sample(sample_record())

    assert sample.id == "sample-id"
    assert sample.target == "C"
    assert sample.metadata["max_prompt_tokens"] == 1024
    assert sample.metadata["max_model_len"] == 4096
    assert sample.metadata["max_tokens"] == 1024
    assert sample.metadata["chat_template_margin_tokens"] == 2048
    assert sample.metadata["input_tokens_after_truncation"] <= 1024
    assert sample.metadata["prompt_variant"] == "official-0shot"
    assert sample.metadata["request_mode"] == "single-generation"
    assert "reasoning_mode" not in sample.metadata


def test_default_context_budget():
    assert longbench_v2.DEFAULT_MAX_MODEL_LEN == 65536
    assert longbench_v2.DEFAULT_MAX_TOKENS == 32768
    assert longbench_v2.DEFAULT_CHAT_TEMPLATE_MARGIN_TOKENS == 2048
    assert longbench_v2.max_prompt_tokens(65536, 32768) == 30720


def test_invalid_context_budget_is_rejected(monkeypatch):
    monkeypatch.setenv("MAX_MODEL_LEN", "34816")
    monkeypatch.setenv("MAX_TOKENS", "32768")
    monkeypatch.setattr(longbench_v2, "_tokenizer", lambda: CharacterTokenizer())
    monkeypatch.setattr(longbench_v2, "_verify_dataset_snapshot", lambda: None)

    with pytest.raises(ValueError, match="must be larger"):
        longbench_v2.record_to_sample(sample_record())


def test_scorer_and_metadata_metrics():
    scorer = longbench_v2.longbench_v2_scorer()
    state = SimpleNamespace(output=SimpleNamespace(completion="The correct answer is (C)"))
    score = asyncio.run(scorer(state, Target("C")))

    assert score.value == 1
    assert score.answer == "C"
    assert score.metadata == {"parsed": True, "prediction": "C"}

    scores = [
        SampleScore(score=score, sample_metadata={"difficulty": "hard", "length": "long"}),
        SampleScore(
            score=Score(value=0, metadata={"parsed": True, "prediction": "B"}),
            sample_metadata={"difficulty": "easy", "length": "short"},
        ),
        SampleScore(
            score=Score(value=0, metadata={"parsed": False, "prediction": None}),
            sample_metadata={"difficulty": "hard", "length": "medium"},
        ),
    ]

    assert longbench_v2.strict_accuracy()(scores) == pytest.approx(1 / 3)
    assert longbench_v2.invalid_output_rate()(scores) == pytest.approx(1 / 3)
    assert longbench_v2.parsed_only_accuracy()(scores) == pytest.approx(1 / 2)
    assert longbench_v2.compensated_accuracy()(scores) == pytest.approx(1.25 / 3)
    assert longbench_v2.difficulty_hard_accuracy()(scores) == pytest.approx(1 / 2)
    assert longbench_v2.difficulty_easy_accuracy()(scores) == 0
    assert longbench_v2.length_long_accuracy()(scores) == 1
