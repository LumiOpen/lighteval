import sys
from types import SimpleNamespace

from lighteval.utils.language_detection import detect_expected_language


class FakeLangDetectException(Exception):
    pass


def install_fake_langdetect(monkeypatch, detect_langs):
    fake_langdetect = SimpleNamespace(
        DetectorFactory=SimpleNamespace(seed=None),
        LangDetectException=FakeLangDetectException,
        detect_langs=detect_langs,
    )
    monkeypatch.setitem(sys.modules, "langdetect", fake_langdetect)
    return fake_langdetect


def test_detect_expected_language_accepts_expected_language(monkeypatch):
    fake_langdetect = install_fake_langdetect(
        monkeypatch,
        lambda text: [SimpleNamespace(lang="fi", prob=0.99)],
    )

    result = detect_expected_language("Tämä on riittävän pitkä suomenkielinen vastaus.", "fi")

    assert result.is_expected
    assert result.detected_language == "fi"
    assert result.confidence == 0.99
    assert result.reason == "detected"
    assert fake_langdetect.DetectorFactory.seed == 0


def test_detect_expected_language_rejects_other_language(monkeypatch):
    install_fake_langdetect(
        monkeypatch,
        lambda text: [SimpleNamespace(lang="en", prob=0.99)],
    )

    result = detect_expected_language("This is a sufficiently long English response.", "fi")

    assert not result.is_expected
    assert result.detected_language == "en"
    assert result.confidence == 0.99
    assert result.reason == "detected"


def test_detect_expected_language_rejects_detection_errors(monkeypatch):
    def detect_langs(text):
        raise FakeLangDetectException("cannot detect")

    install_fake_langdetect(monkeypatch, detect_langs)

    result = detect_expected_language("Tämä on riittävän pitkä vastaus tunnistusta varten.", "fi")

    assert not result.is_expected
    assert result.detected_language is None
    assert result.confidence is None
    assert result.reason == "detection_error"


def test_detect_expected_language_rejects_empty_detection(monkeypatch):
    install_fake_langdetect(monkeypatch, lambda text: [])

    result = detect_expected_language("Tämä on riittävän pitkä vastaus tunnistusta varten.", "fi")

    assert not result.is_expected
    assert result.detected_language is None
    assert result.confidence is None
    assert result.reason == "no_detection"
