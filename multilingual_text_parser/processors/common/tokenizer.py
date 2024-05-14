import re

from multilingual_text_parser._constants import PUNCTUATION
from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["Tokenizer"]


class Tokenizer(BaseSentenceProcessor):
    def __init__(self):
        self._add_space = re.compile(f"([{PUNCTUATION}])")

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        sent.text = self._add_space.sub(r" \1 ", sent.text)
        sent.tokenize()
