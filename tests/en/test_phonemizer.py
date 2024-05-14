import pytest

from multilingual_text_parser import Doc, TextParser, TokenUtils

parser = TextParser(lang="EN")

testdata = [
    (  # type: ignore
        "He+llo world",
        [
            ("HH", "AH1", "L", "OW0"),
            ("W", "ER1", "L", "D"),
        ],
    ),
    (  # type: ignore
        "I re+fuse to collect the refu+se around here",
        [
            ("AY1",),
            ("R", "IH1", "F", "Y", "UW0", "Z"),
            ("T", "UW1"),
            ("K", "AH0", "L", "EH1", "K", "T"),
            ("DH", "AH0"),
            ("R", "EH0", "F", "Y", "UW1", "Z"),
            ("ER0", "AW1", "N", "D"),
            ("HH", "IY1", "R"),
        ],
    ),
]


@pytest.mark.parametrize("utterance, expected", testdata)
def test_phonemizer(utterance, expected):
    doc = parser.process(Doc(utterance))
    attr = TokenUtils.get_attr(doc.tokens, ["phonemes"])
    phonemes = attr["phonemes"]
    assert phonemes == expected
