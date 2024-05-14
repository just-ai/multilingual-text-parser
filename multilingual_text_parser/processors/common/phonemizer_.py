import phonemizer

from phonemizer.backend import BACKENDS
from phonemizer.phonemize import _phonemize

from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.lang_supported import espeak_available_languages

__all__ = ["Phonemizer"]


class Phonemizer(BaseSentenceProcessor):
    def __init__(self):
        self._espeak_phonemizer = None
        self._supported_languages = espeak_available_languages()
        self._punctuation_marks = phonemizer.punctuation.Punctuation.default_marks()
        self._separator = phonemizer.separator.Separator(phone="-", word=" ")

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        if kwargs.get("disable_phonemizer", False):
            return

        lang = kwargs["lang"].lower()
        if lang in self._supported_languages:
            if self._espeak_phonemizer is None:
                self._espeak_phonemizer = BACKENDS["espeak"](
                    lang,
                    punctuation_marks=self._punctuation_marks,
                    preserve_punctuation=True,
                    with_stress=True,
                    tie=False,
                    language_switch="keep-flags",
                    words_mismatch="ignore",
                    logger=None,
                )

            text = [token.text for token in sent.tokens]
            phonemes = _phonemize(
                self._espeak_phonemizer,
                text,
                separator=self._separator,
                strip=True,
                njobs=1,
                prepend_text=False,
                preserve_empty_lines=False,
            )

            for i in range(len(sent.tokens)):
                if (
                    sent.tokens[i].modifiers
                    and "phoneme" in sent.tokens[i].modifiers
                    and "ph" in sent.tokens[i].modifiers["phoneme"]
                ):
                    sent.tokens[i].phonemes = [
                        ph if len(ph) == 1 else tuple(p for p in ph)
                        for ph in sent.tokens[i].modifiers["phoneme"]["ph"].split("|")
                    ]
                elif sent.tokens[i].is_punctuation or "(en)" in phonemes[i]:
                    sent.tokens[i].phonemes = None
                else:
                    phonemes_cur = phonemes[i].split("-")
                    sent.tokens[i].phonemes = [
                        ph if len(ph) == 1 else tuple(p for p in ph)
                        for ph in phonemes_cur
                        if len(ph) > 0
                    ]

        else:
            for i in range(len(sent.tokens)):
                if sent.tokens[i].is_punctuation:
                    sent.tokens[i].phonemes = None
                else:
                    sent.tokens[i].phonemes = list(sent.tokens[i].text)
