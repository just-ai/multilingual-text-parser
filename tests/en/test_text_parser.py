import pytest

from multilingual_text_parser.data_types import Doc, TokenUtils
from multilingual_text_parser.parser import TextParser

parser = TextParser(lang="EN")

testdata = [
    (  # type: ignore
        "How are you?",
        [
            ("HH", "AW1"),
            ("AA1", "R"),
            ("Y", "UW1"),
        ],
    ),
    (  # type: ignore
        "Help desk",
        [
            ("HH", "EH1", "L", "P"),
            ("D", "EH1", "S", "K"),
        ],
    ),
]


@pytest.mark.parametrize("utterance, expected", testdata)
def test_phonemizer(utterance, expected):
    doc = parser.process(Doc(utterance))
    attr = TokenUtils.get_attr(doc.tokens, ["phonemes"])
    phonemes = attr["phonemes"]
    assert phonemes == expected
