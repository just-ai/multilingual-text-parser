from natasha import NewsEmbedding, NewsMorphTagger
from navec import Navec
from slovnet import Morph

from multilingual_text_parser.data_types import Doc, Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["PosTaggerRU"]


class PosTaggerRU(BaseSentenceProcessor):
    def __init__(self):
        self._emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._emb)

    def __call__(self, doc: Doc, **kwargs) -> Doc:
        doc.tag_morph(self._morph_tagger)
        return super().__call__(doc, **kwargs)

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        for token in sent.tokens:
            if token.pos == "PROPN":
                token.pos = "NOUN"
            if not token.is_punctuation and token.pos.startswith("PUNCT"):
                token.pos = "X"


if __name__ == "__main__":
    import json

    pos_tagger = PosTaggerRU()

    doc = Doc("отползавшим от ручья", sentenize=True, tokenize=True)
    doc = pos_tagger(doc)

    print(json.dumps(doc.to_dict(), ensure_ascii=False, indent=4))

    navec = Navec.load("P:\\TTS\\navec_news_v1_1B_250K_300d_100q.tar")
    morph = Morph.load("P:\\TTS\\slovnet_morph_news_v1.tar", batch_size=1)
    morph.navec(navec)

    markup = next(morph.map(["отползавшим мухам".split()]))
    for token in markup.tokens:
        print(f"{token.text:>20} {token.tag}")

    markup = next(morph.map(["отползавшим от ручья".split()]))
    for token in markup.tokens:
        print(f"{token.text:>20} {token.tag}")
