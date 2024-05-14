import multilingual_text_parser

from multilingual_text_parser.data_types import Doc, Sentence

__all__ = ["BaseRawTextProcessor", "BaseTextProcessor", "BaseSentenceProcessor"]


class BaseRawTextProcessor:
    def __call__(self, doc: Doc, **kwargs) -> Doc:
        if doc.sents and not isinstance(
            self, multilingual_text_parser.processors.ru.homo_classifier.HomographerRU
        ):
            raise RuntimeError("This handler must be used before Spliter")
        self._process_text(doc, **kwargs)
        return doc

    def _process_text(self, doc: Doc, **kwargs):
        pass


class BaseTextProcessor:
    def __call__(self, doc: Doc, **kwargs) -> Doc:
        self._process_text(doc, **kwargs)
        return doc

    def _process_text(self, doc: Doc, **kwargs):
        pass


class BaseSentenceProcessor:
    def __call__(self, doc: Doc, **kwargs) -> Doc:
        # Lines below fail on text without words, e.g. text "\n". Consider changing to Warning.
        if not doc.sents:
            raise RuntimeError("This handler must be used after Sentenizer")
        for sent in doc.sents:
            self._process_sentence(sent, **kwargs)
        return doc

    def _process_sentence(self, sent: Sentence, **kwargs):
        pass
