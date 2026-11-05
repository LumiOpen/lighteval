"""Finnish-adapted MultiIF verifier.

Rather than fork the 3400-line English verifier, this module subclasses the specific
checker classes that need Finnish behaviour and rebuilds INSTRUCTION_DICT. Everything
else (format/content/length-word/paragraph/postscript/etc.) is reused unchanged from
`lighteval.tasks.tasks.multiif.instructions`, since those checks are language-neutral.

Finnish adaptations:
  * change_case:english_capital / english_lowercase -> require langdetect == "fi"
    (the English versions hard-require "en" and would fail correct Finnish responses).
  * keyword:existence / frequency / forbidden_words -> simplemma-lemmatised matching
    (so inflected Finnish forms, e.g. "sotaa"/"sodan", match the keyword lemma "sota").
  * length_constraints:number_sentences -> Finnish sentence segmentation.
  * language:response_language works unchanged (langdetect; LANGUAGE_CODES already has fi).
"""

import langdetect

from lighteval.tasks.tasks.multiif import instructions as _en
from lighteval.tasks.tasks.multiif_fi import fi_morph


_REL = _en._COMPARISON_RELATION  # ("less than", "at least")


class KeywordChecker(_en.KeywordChecker):
    def check_following(self, value):
        return all(fi_morph.keyword_present(value, kw) for kw in self._keywords)


class KeywordFrequencyChecker(_en.KeywordFrequencyChecker):
    def check_following(self, value):
        actual = fi_morph.keyword_count(value, self._keyword)
        if self._comparison_relation == _REL[0]:
            return actual < self._frequency
        return actual >= self._frequency


class ForbiddenWords(_en.ForbiddenWords):
    def check_following(self, value):
        return not any(fi_morph.keyword_present(value, w) for w in self._forbidden_words)


class NumberOfSentences(_en.NumberOfSentences):
    def check_following(self, value):
        n = fi_morph.count_sentences_fi(value)
        if self._comparison_relation == _REL[0]:
            return n < self._num_sentences_threshold
        return n >= self._num_sentences_threshold


class CapitalLettersFinnishChecker(_en.CapitalLettersEnglishChecker):
    def check_following(self, value):
        assert isinstance(value, str)
        try:
            return value.isupper() and langdetect.detect(value) == "fi"
        except langdetect.LangDetectException:
            return True  # undetectable -> treat as followed (matches upstream behaviour)


class LowercaseLettersFinnishChecker(_en.LowercaseLettersEnglishChecker):
    def check_following(self, value):
        assert isinstance(value, str)
        try:
            return value.islower() and langdetect.detect(value) == "fi"
        except langdetect.LangDetectException:
            return True


# Rebuild the instruction map, swapping in the Finnish-adapted classes.
INSTRUCTION_DICT = dict(_en.INSTRUCTION_DICT)
INSTRUCTION_DICT[_en._KEYWORD + "existence"] = KeywordChecker
INSTRUCTION_DICT[_en._KEYWORD + "frequency"] = KeywordFrequencyChecker
INSTRUCTION_DICT[_en._KEYWORD + "forbidden_words"] = ForbiddenWords
INSTRUCTION_DICT[_en._LENGTH + "number_sentences"] = NumberOfSentences
INSTRUCTION_DICT[_en._CHANGE_CASES + "english_capital"] = CapitalLettersFinnishChecker
INSTRUCTION_DICT[_en._CHANGE_CASES + "english_lowercase"] = LowercaseLettersFinnishChecker
