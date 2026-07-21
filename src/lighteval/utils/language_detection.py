from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageDetectionResult:
    expected_language: str
    detected_language: str | None
    confidence: float | None
    is_expected: bool
    reason: str


def detect_expected_language(
    text: str,
    expected_language: str,
) -> LanguageDetectionResult:
    stripped = text.strip()

    try:
        import langdetect
    except ImportError as e:
        raise ImportError(
            "Language-gated generation scoring requires `langdetect`. "
            "Install it with `pip install lighteval[extended_tasks]`."
        ) from e

    try:
        langdetect.DetectorFactory.seed = 0
        detected_languages = langdetect.detect_langs(stripped)
    except langdetect.LangDetectException:
        return LanguageDetectionResult(
            expected_language=expected_language,
            detected_language=None,
            confidence=None,
            is_expected=False,
            reason="detection_error",
        )

    if not detected_languages:
        return LanguageDetectionResult(
            expected_language=expected_language,
            detected_language=None,
            confidence=None,
            is_expected=False,
            reason="no_detection",
        )

    detected = detected_languages[0]
    detected_language = detected.lang
    confidence = detected.prob
    return LanguageDetectionResult(
        expected_language=expected_language,
        detected_language=detected_language,
        confidence=confidence,
        is_expected=detected_language == expected_language,
        reason="detected",
    )
