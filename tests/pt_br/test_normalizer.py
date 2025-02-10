import pytest

from multilingual_text_parser.parser import TextParser

from multilingual_text_parser.data_types import Doc

text_parser = TextParser(lang="PT-BR")


testdata = [
    # количественные числительные
    ("123", "cento e vinte e três"),
    ("2,5", "dois vírgula cinco"),
    ("2.5", "dois vírgula cinco"),
    (
        "100000000000000000000000000000",
        "um zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero zero",
    ),
    # порядковые числительные
    ("o 1º homem", "o primeiro homem"),
    ("a 2ª pessoa", "a segunda pessoa"),
    ("2-as pessoas", "segundas pessoas"),
    # даты и года
    (
        "Ela nasceu 2.08.1993",
        "ela nasceu no dia dois de agosto de mil, novecentos e noventa e três",
    ),
    (
        "Ela nasceu em agosto de 1992",
        "ela nasceu em agosto de mil, novecentos e noventa e dois",
    ),
    ("03.31.2001", "no dia trinta e um de março de dois mil e um"),
    ("2001/03/31", "no dia trinta e um de março de dois mil e um"),
    ("2001/31/03", "no dia trinta e um de março de dois mil e um"),
    ("1990-1995", "de mil, novecentos e noventa a mil, novecentos e noventa e cinco"),
    # символы
    ("№1", "número um"),
    ("23%", "vinte e três por cento"),
    ("-1", "menos um"),
    ("+1", "mais um"),
    # валюта
    ("$ 10", "dez dólares"),
    ("€ 10", "dez euros"),
    ("¥ 10", "dez yuan"),
    ("£ 10", "dez libras"),
    ("R$10", "dez reais"),
    ("$ 10.05", "dez dólares e cinco centavos"),
]


@pytest.mark.parametrize("text, expected", testdata)
def test_normalizer(text, expected):
    doc = text_parser.process(Doc(text))
    assert doc.text.rstrip(".") == expected
