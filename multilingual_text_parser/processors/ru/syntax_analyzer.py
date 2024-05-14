from natasha import Doc as NatashaDoc
from natasha import NewsEmbedding, NewsSyntaxParser, Segmenter

from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor

__all__ = ["SyntaxAnalyzerRU"]


class SyntaxAnalyzerRU(BaseSentenceProcessor):
    def __init__(self):
        self._emb = NewsEmbedding()
        self._syntax_parser = NewsSyntaxParser(self._emb)

    def _process_sentence(self, sent: Sentence, **kwargs):
        tmp = NatashaDoc(sent.text, sents=[sent])
        tmp.parse_syntax(self._syntax_parser)  # assign syntax to sent
        for token in sent.tokens:
            if token.is_punctuation:
                token.rel = None
                token.head_id = None
                token.id = None
