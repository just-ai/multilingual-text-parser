import re

from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.processors.base import BaseRawTextProcessor
from multilingual_text_parser.processors.common import Corrector
from multilingual_text_parser.processors.ru.normalizer_utils import Utils
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir

__all__ = ["SentenizerRU"]


class SentenizerRU(BaseRawTextProcessor):
    def __init__(self):
        vocabs_dir = get_root_dir() / "data/ru/vocabularies"
        self._modifiers_vocab = Utils.read_vocab(vocabs_dir / "num_modifiers.txt")

    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        doc.sentenize(tokenize=kwargs.get("tokenize", False))
        for sent in doc.sents:
            _text = Corrector.trim_punctuation(sent.text).strip()

            if not re.search("(дом|улица)", _text):
                for key in self._modifiers_vocab.keys():
                    _text = _text.replace(f" {key} . ", f" {key} ")

            _text = (
                _text.replace(" . ", " , ")
                .replace(" ! ", " , ")
                .replace(" ? ", " , ")
                .replace(" ; ", " , ")
            )
            sent.text = _text

        if not doc.sents:
            doc.sents = [""]
