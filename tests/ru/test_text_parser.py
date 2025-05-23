import pytest

from multilingual_text_parser.data_types import Doc, TokenUtils
from multilingual_text_parser.parser import TextParser
from multilingual_text_parser.processors import Corrector, TextModifier, TextModifierRU

text_modifier = TextModifier()
text_modifier_ru = TextModifierRU()

testdata = [
    ("1-34-23-42", " 1ЪХЪ34ЪХЪ23ЪХЪ42. "),
    ("кое-как они дошлии-", " коеЪХЪкак они дошлии-. "),
    ("со счетом 2:2;4:7", " со счётом 2ЪТЪ2;4ЪТЪ7. "),
    ("$680 #4", " ЪВДЪ680 #4. "),
    (
        "-3++4=7 +7965 (45,43 43.23) -как- №645 «Т-72»",
        " ЪХЪ3++4=7 ЪРЪ7965 (45ЪДСЪ43 43ЪДРЪ23) -как- ЪНЪ645 «ТЪХЪ72». ",
    ),
    ("34% 12@mail.ru 34#24", " 34ЪПЧЪ 12@mail.ru 34 решётка 24. "),
]


@pytest.mark.parametrize("text, expected", testdata)
def test_text_modifier(text, expected):
    doc = text_modifier_ru(text_modifier(Doc(text)))
    assert doc == expected


corrector = Corrector()

testdata = [
    ("ночь\n улица \nфонарь аптека", "ночь улица фонарь аптека ."),
    ("ночь.\n\nулица.\n\nфонарь.\nаптека", "ночь . улица . фонарь . аптека ."),
    ("ночь\nУлица \nФонарь,\nаптека", "ночь Улица . Фонарь , аптека ."),
    ("ночь\nУлица \nФонарь,\nаптека", "ночь Улица . Фонарь , аптека ."),
    ("М-м-м, вы уверены?", "М , вы уверены ?"),
    (
        "[   ночь ](1(улица){Фонарь}) аптека  .",
        "( ночь ) ( 1 ( улица ) ( Фонарь ) аптека .",
    ),
    (
        "[[[ ночь ]]] улица. . . Фонарь, ,, аптека ?? ! ! ",
        "( ночь ) улица . Фонарь , аптека ? !",
    ),
    (
        "- Я: /Хорошая 'сегодня' <погода> на улице/?!",
        "Я : Хорошая сегодня погода на улице ? !",
    ),
    (
        "Ребята, вы чего?&!! А как же «Терминатор», «Чужие», «Правдивая ложь»?",
        "Ребята , вы чего ? ! А как же Терминатор , Чужие , Правдивая ложь ?",
    ),
    (
        " Я зашел в зал, увидел группу старших товарищей, боксирующих на ринге (на настоящем ринге!), "
        "десяток боксерских груш, суровые мужские лица с плоскими носами и здоровенного тренера . "
        "Все было как в кино . «Эй, новичок! Давай в строй!» — крикнули мне .",
        "Я зашел в зал , увидел группу старших товарищей , боксирующих на ринге ( на настоящем ринге ! ) , "
        "десяток боксерских груш , суровые мужские лица с плоскими носами и здоровенного тренера . "
        "Все было как в кино . Эй , новичок ! Давай в строй ! — крикнули мне .",
    ),
    (
        "из-за угла показалась зелено-фиолетовая . Она петляла туда- сюда. И - отдельно -наблюдать. ",
        "из - за угла показалась зелено - фиолетовая . Она петляла туда - сюда . И - отдельно - наблюдать .",
    ),
]


@pytest.mark.parametrize("text, expected", testdata)
def test_corrector(text, expected):
    text = corrector(Doc(text), **{"lang": "RU"}).text
    assert text == expected


parser = TextParser(lang="RU")

testdata = [
    ("1. Первый ", "один, первый."),
    ("1) Первый ", "один) первый."),
    ("Купи 1.", "купи один."),
    ("но и болез -: ни", "но и болез: ни."),
    ("как-нибудь", "как нибудь."),
    ("65-я годовщина", "шестьдесят пятая годовщина."),
]


@pytest.mark.parametrize("text, expected", testdata)
def test_parser(text, expected):
    doc = parser.process(Doc(text))
    assert doc.sents[0].text == expected


testdata = [
    (  # type: ignore
        "рыбки и икринки",
        [
            ("R", "Y0", "P", "K0", "I"),
            ("I",),
            ("I", "K", "R0", "I0", "N", "K0", "I"),
        ],
    ),
    (  # type: ignore
        "большой прирост заболевших",
        [
            ("B", "A", "L0", "SH", "O0", "J0"),
            ("P", "R0", "I", "R", "O0", "Z", "D"),
            ("Z", "A", "B", "A", "L0", "E0", "F", "SH", "Y", "KH"),
        ],
    ),
    (  # type: ignore
        "ОТП",
        [
            ("O", "T", "Y", "P", "E0"),
        ],
    ),
]


@pytest.mark.parametrize("utterance, expected", testdata)
def test_phonemizer(utterance, expected):
    doc = parser.process(Doc(utterance))
    attr = TokenUtils.get_attr(doc.tokens, ["phonemes"])
    phonemes = attr["phonemes"]
    assert phonemes == expected
