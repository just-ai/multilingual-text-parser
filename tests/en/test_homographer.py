import pytest

from multilingual_text_parser import (
    Corrector,
    HomographerEN,
    NormalizerEN,
    SentencesModifier,
    Sentenizer,
    SSMLApplier,
    SSMLCollector,
    SymbolsModifier,
    TextModifier,
    Tokenizer,
)
from multilingual_text_parser.data_types import Doc

symb_mode = SymbolsModifier()
text_mode = TextModifier()
corrector = Corrector()
sentenizer = Sentenizer()
sent_mode = SentencesModifier()
ssmlcol = SSMLCollector()
tokenizer = Tokenizer()
ssmlapp = SSMLApplier()
normalizer = NormalizerEN()
homographer = HomographerEN()

testdata = [
    (
        "It is possible now to read off the major and minor axes of this ellipse",
        ["AE1", "K", "S", "IY0", "Z"],
    ),
    (
        "The arms show two crossed silver axes on a blue background",
        ["AE1", "K", "S", "IH0", "Z"],
    ),
]


@pytest.mark.parametrize("text, expected", testdata)
def test_homographer(text, expected):
    doc = Doc(text)
    doc = text_mode(symb_mode(doc, **{"lang": "EN"}))
    doc = sentenizer(corrector(doc))
    doc = ssmlcol(sent_mode(doc))
    doc = ssmlapp(tokenizer(doc))
    doc = normalizer(doc)
    doc = homographer(doc)
    sent = doc.sents[0]
    assert expected in sent.get_attr("phonemes")
