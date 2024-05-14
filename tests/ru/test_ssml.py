import pytest

from multilingual_text_parser import Doc, TextParser

PROCESSOR = TextParser(lang="RU")

BREAK_TEST_DATA = [
    (
        """
        <s> улица </s> <s> фонарь </s> <s> аптека </s>
        """,
        [],
    ),
    (
        """
        <p> улица </p> <p> фонарь </p> <p> аптека </p>
        """,
        [
            (-1, {"break": {"time": "0.3s"}}),
            (-1, {"break": {"time": "0.3s"}}),
            (-1, {"break": {"time": "0.3s"}}),
        ],
    ),
    (
        """
        <break time="0.3s"/> в <audio> <break time="0.1s"/> том самом,
        <break time="0.2s"/> голодном <break time="0.1s"/> девяностом году <break time="0.28s"/> был <break time="200ms"/> открыт этот музей <break time="200ms"/></audio>.
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


LINGUISTIC_TEST_DATA = [
    (
        """
        На столе лежало <sub alias= "зелёное">красное</sub> яблоко.
        """,
        "на столе лежало зелёное яблоко.",
    ),
    (
        """
        <sub alias="ноль">один два</sub>
        """,
        "ноль ноль.",
    ),
    (
        """
        <say-as interpret-as="date" format=" ydm ">20212102</say-as>
        """,
        "двадцать первое февраля две тысячи двадцать первого года.",
    ),
    (
        """
        <say-as interpret-as="date" format="md">6.11</say-as>
        """,
        "одиннадцатое июня.",
    ),
    (
        """
        <say-as interpret-as="telephone">89542344213</say-as>
        """,
        "восемь -девятьсот пятьдесят четыре -двести тридцать четыре -сорок два -тринадцать.",
    ),
    (
        """
        <say-as interpret-as="telephone">23-43-11</say-as>
        """,
        "двадцать три -сорок три -одиннадцать.",
    ),
    (
        """
        <say-as interpret-as="characters">РЛС</say-as>
        """,
        "эрэлэс.",
    ),
    (
        """
        <say-as interpret-as="cardinal">42</say-as>
        """,
        "сорок два.",
    ),
    (
        """
        <say-as interpret-as="ordinal">42</say-as>
        """,
        "сорок второй.",
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


@pytest.mark.parametrize("text, result", LINGUISTIC_TEST_DATA)
def test_linguistic_tag(text, result):
    assert PROCESSOR.process(Doc(text)).text == result
