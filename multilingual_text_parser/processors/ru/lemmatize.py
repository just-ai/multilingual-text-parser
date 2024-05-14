import logging

from natasha import MorphVocab

from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.log_utils import trace

__all__ = ["LemmatizeRU"]

LOGGER = logging.getLogger("root")


class LemmatizeRU(BaseSentenceProcessor):
    def __init__(self):
        self._morph_vocab = MorphVocab()

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        for token in sent.tokens:
            try:
                token.lemmatize(self._morph_vocab)
            except Exception:
                LOGGER.error(trace(self, f"failed to lemmatize {token.text}", full=False))
                token.lemma = token.text
