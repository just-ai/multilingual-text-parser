import logging

import regex as re

from multilingual_text_parser import Doc
from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.processors.common import Corrector

__all__ = ["CleanerRU"]

LOGGER = logging.getLogger("root")


class CleanerRU:
    punct = r""",.:;?!\-)"""

    def __init__(self, skip_invalid_sents: bool = False):
        self._skip_invalid_sents = skip_invalid_sents
        self._patterns = [
            (r"\S*@\S*", ""),
            (r"[[{<]", "("),
            (r"[\]}>]", ")"),
            (f"[^а-яёА-ЯЁ{self.punct}(\\s\t\n\r]", ""),
            (r"([\s\t\n\r])([ьъЬЪ]+)", " "),
            (f"(-)([{self.punct}]+)", r"\1"),
        ]

    def _clear(self, _str: str) -> str:
        _str = _str.replace("e", "е")

        for p, t in self._patterns:
            _str = re.sub(p, t, _str)

        _str = Corrector.trim_punctuation(_str)
        return _str

    def __call__(self, text: Doc, **kwargs) -> Doc:
        if text.sents is None:
            text.text = self._clear(text.text)
        else:
            clear_sents = []
            for sent in text.sents:
                processed_sents = self._process_sentence(sent)
                clear_sents.append(processed_sents)
            text.sents = clear_sents

        return text

    def _process_sentence(self, sent: Sentence, **kwargs):
        new_sents = []
        clear_text = self._clear(sent.text)
        if clear_text != sent.text:
            if not self._skip_invalid_sents:
                temp = Doc(clear_text, sentenize=True, tokenize=True).sents

                if len(temp) == 1:
                    # temp[0].orig = sent.orig
                    temp[0].ssml_insertions = sent.ssml_insertions
                    start_search_from = 0
                    for new_token in temp[0].tokens:
                        new_text = new_token.text
                        try:
                            for idx, old_token in enumerate(
                                sent.tokens[start_search_from:],
                                start_search_from,
                            ):
                                if new_text == old_token.text:
                                    start_search_from = idx + 1
                                    new_token.modifiers = old_token.modifiers
                                    break
                        except IndexError:
                            pass
                else:
                    LOGGER.warning(f"cleaning error for: {sent.text}")
                    temp = []
                new_sents += temp
            else:
                LOGGER.warning(f"invalid characters in: {sent.text}")
        else:
            new_sents.append(sent)
        return new_sents


if __name__ == "__main__":

    cleaner = CleanerRU()

    text = Doc("- - .12 стульев # just-ai@рф - ? - ")
    print(cleaner(text).text)
