import pytest

from multilingual_text_parser import Doc, TextParser

PROCESSOR = TextParser(lang="EN")

BREAK_TEST_DATA = [
    (
        """
        <s> i </s> <s> have </s> <s> fun </s>
        """,
        [],
    ),
    (
        """
        <p> i </p> <p> have </p> <p> fun </p>
        """,
        [
            (-1, {"break": {"time": "0.3s"}}),
            (-1, {"break": {"time": "0.3s"}}),
            (-1, {"break": {"time": "0.3s"}}),
        ],
    ),
    (
        """
        <break time="0.3s"/> yes <audio> <break time="0.1s"/> right there,
        <break time="0.2s"/> you <break time="0.1s"/> know what <break time="0.28s"/> well <break time="200ms"/> what i mean  <break time="200ms"/></audio>.
        """,
        [
            (-1, {"break": {"time": "0.3s"}}),
            (0, {"audio": {}, "break": {"time": "0.1s"}}),
            (3, {"audio": {}, "break": {"time": "0.2s"}}),
            (4, {"audio": {}, "break": {"time": "0.1s"}}),
            (6, {"audio": {}, "break": {"time": "0.28s"}}),
            (7, {"audio": {}, "break": {"time": "200ms"}}),
            (10, {"audio": {}, "break": {"time": "200ms"}}),
        ],
    ),
]


def _get_pauses_sequences(tokens):
    sequences = []
    current_pauses = 0
    for i, token in enumerate(tokens):
        if token.text == "<SIL>":
            current_pauses += 1
        elif i > 0:
            if tokens[i - 1].text == "<SIL>":
                sequences.append(current_pauses)
            current_pauses = 0

    return sequences


@pytest.mark.parametrize("text, ssml_insertions", BREAK_TEST_DATA)
def test_break_self_closing_tag(text, ssml_insertions):
    doc = PROCESSOR.process(Doc(text))
    text_insertions = []
    for i, sent in enumerate(doc.sents):
        text_insertions.extend(sent.ssml_insertions)
    assert ssml_insertions == text_insertions
