import re
import sys
import decimal
import logging

from copy import deepcopy

import num2words

from multilingual_text_parser._constants import PUNCTUATION
from multilingual_text_parser.data_types import Sentence, Token
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["Normalizer"]

LOGGER = logging.getLogger("root")


class Normalizer(BaseSentenceProcessor):
    GPU_CAPABLE: bool = True
    MULTILANG: bool = True

    def __init__(self, lang: str = "ES", device: str = "cpu"):
        self.nums = re.compile(r"(\d+)?[.,]?\d+")
        self.langs = {
            "kk": "kz",
            "cs": "cz",
            "da": "dk",
            "fr-be": "fr_BE",
            "fr-ch": "fr_CH",
            "nb": "no",
            "pt-br": "pt_BR",
            "fr-fr": "fr",
        }
        self._lang = lang

        if sys.platform != "win32":
            if lang in ["EN", "DE", "ES", "RU"]:
                from nemo_text_processing.text_normalization.normalize import Normalizer

                self._normalizer = Normalizer(input_case="cased", lang=lang.lower())
                self._add_space = re.compile(f"([{PUNCTUATION}])")
            else:
                LOGGER.warning(f"NeMo NLP not support {lang} language!")
                self._normalizer = None  # type: ignore
        else:
            LOGGER.warning("NeMo NLP not support on Windows platform!")
            self._normalizer = None  # type: ignore

        self._is_support_lang = not (
            self._normalizer is None
            and lang.lower() not in self.langs
            and lang.lower() not in num2words.CONVERTER_CLASSES
        )

        if not self._is_support_lang:
            LOGGER.warning(f"Normalization for language {lang} is not support!")

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        if not self._is_support_lang:
            return

        lang = kwargs.get("lang", self._lang).lower()
        if lang in self.langs:
            lang = self.langs[lang]

        if self._normalizer is not None:
            old = sent.tokens
            text = " ".join([token.text for token in old])

            text = self._normalizer.normalize(text, verbose=False)
            text = self._add_space.sub(
                r" \1 ", text
            )  # добавление пробелов вокруг пунктуации

            tokens = text.split()

            new = []
            j = 0
            i = 0
            while i < len(old):
                if tokens[j] == old[i].text:
                    new.append(old[i])
                    j += 1
                    i += 1
                elif i == 0:
                    new.append(Token(tokens[j]))
                    j += 1
                    i += 1
                elif tokens[j - 1] == old[i - 1].text:
                    new.append(Token(tokens[j]))
                    j += 1
                    i += 1
                elif old[i].text not in tokens:
                    k = i + 1
                    while old[k].text not in tokens:
                        k += 1
                    while j < len(tokens) and old[k].text != tokens[j]:
                        new.append(Token(tokens[j]))
                        j += 1
                    i = k
                else:
                    while j < len(tokens) and tokens[j] != old[i].text:
                        new.append(Token(tokens[j]))
                        j += 1
                    if j < len(tokens):
                        if tokens[j] == old[i].text:
                            j += 1
                            new.append(old[i])
                    else:
                        new.append(old[i])
                        break
                    i += 1
            sent.tokens = new

        elif lang in num2words.CONVERTER_CLASSES:
            normalized = []
            for token in sent.tokens:
                if token.is_number:
                    text = self.nums.search(token.text)
                    if text:
                        num = text.group()
                        if lang == "it":
                            if num.isnumeric():
                                to_words = num2words.num2words(
                                    int(num), lang=lang
                                ).split()
                            else:
                                to_words = num2words.num2words(
                                    float(num), lang=lang
                                ).split()
                        else:
                            try:
                                to_words = num2words.num2words(num, lang=lang).split()
                            except decimal.InvalidOperation:
                                to_words = num2words.num2words(
                                    num.replace(",", "."), lang=lang
                                ).split()

                        normalized.append(to_words)
            new = []
            j = 0
            for i, token in enumerate(sent.tokens):
                if token.is_number:
                    new.extend(sent.tokens[j:i])
                    text_norm = normalized.pop(0)
                    for t in text_norm:
                        tok = deepcopy(token)
                        tok.text = t
                        new.append(tok)
                    j = i + 1
            new.extend(sent.tokens[j:])

            new_tokens = []
            for token in new:
                if not token.is_punctuation and "-" in token.text:
                    for word in token.text.split("-"):
                        token.text = word
                        new_tokens.append(deepcopy(token))
                else:
                    new_tokens.append(token)

            sent.tokens = new_tokens
