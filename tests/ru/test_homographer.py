import pytest

from multilingual_text_parser import (
    Corrector,
    HomographerRU,
    RuleBasedNormalizerRU,
    SentencesModifierRU,
    SentenizerRU,
    SymbolsModifier,
    SyntaxAnalyzerRU,
    TextModifier,
    TextModifierRU,
    Tokenizer,
)
from multilingual_text_parser.data_types import Doc

symb_mode = SymbolsModifier()
text_mode = TextModifier()
text_mode_ru = TextModifierRU()
sent_mode = SentencesModifierRU()
sentenizer = SentenizerRU()
syntaxer = SyntaxAnalyzerRU()
tokenizer = Tokenizer()
corrector = Corrector()
normalizer = RuleBasedNormalizerRU()
homographer = HomographerRU()

testdata = [
    ("Она мне дорога", [None, None, "дорога+", None]),
    ("Какая длинная дорога", [None, None, "доро+га", None]),
    ("Он живет в замке", [None, None, None, "за+мке", None]),
    ("Я не закрываюсь на замок", [None, None, None, None, "замо+к", None]),
    (
        "он живет в замке и закрывается на замок.",
        [None, None, None, "за+мке", None, None, None, "замо+к", None],
    ),
    (
        "у него совсем нет волос, остался лишь один волос.",
        [None, None, None, None, "воло+с", None, None, None, None, "во+лос", None],
    ),
    (
        "все опустили го+ловы, потому что у него нет головы+.",
        [None, None, "го+ловы", None, None, None, None, None, None, "головы+", None],
    ),
]


@pytest.mark.parametrize("text, expected", testdata)
def test_homographer(text, expected):
    doc = Doc(text)
    doc = text_mode(symb_mode(doc, **{"lang": "RU"}))
    doc = text_mode_ru(symb_mode(doc))
    doc = sentenizer(corrector(doc))
    doc = tokenizer(sent_mode(doc))
    doc = syntaxer(doc)
    doc = text_mode.restore(doc)
    doc = normalizer(doc)
    doc = homographer(doc)
    sent = doc.sents[0]
    assert sent.get_attr("stress") == expected
