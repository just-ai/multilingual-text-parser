from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.processors.base import BaseRawTextProcessor
from multilingual_text_parser.processors.common import Corrector
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["Sentenizer"]


class Sentenizer(BaseRawTextProcessor):
    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        doc.sentenize(tokenize=kwargs.get("tokenize", False))
        for sent in doc.sents:
            _text = Corrector.trim_punctuation(sent.text).strip()

            _text = (
                _text.replace(" . ", " ")
                .replace(" ! ", " , ")
                .replace(" ? ", " , ")
                .replace(" ; ", " , ")
            )

            sent.text = _text

        if not doc.sents:
            doc.sents = [""]
