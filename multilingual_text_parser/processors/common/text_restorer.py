from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["OriginalTextRestorer"]


class OriginalTextRestorer(BaseSentenceProcessor):
    def __init__(self):
        pass

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        orig = sent.text_orig
        orig_lower = orig.lower()
        start = 0
        for token in sent.tokens:
            if token.normalized:
                token.text_orig = None
            else:
                pos = orig_lower[start:].find(token.text)
                if pos != -1:
                    token.text_orig = orig[start:][pos : pos + len(token.text)]
                    start += pos + len(token.text)
                else:
                    token.text_orig = None
