import regex as re

from multilingual_text_parser._constants import (
    ALPHABET_DE,
    ALPHABET_ES,
    ALPHABET_KK,
    ALPHABET_UZ,
    PUNCTUATION,
)
from multilingual_text_parser.data_types import Doc, Sentence
from multilingual_text_parser.processors.base import (
    BaseRawTextProcessor,
    BaseSentenceProcessor,
)
from multilingual_text_parser.processors.common.ssml_processor import collect_ssml
from multilingual_text_parser.processors.ru.rulebased_normalizer import Utils
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir

__all__ = ["SymbolsModifier", "TextModifier", "SentencesModifier"]


class SentencesModifier(BaseSentenceProcessor):
    def __init__(self):
        self._clear = {
            "DE": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}{ALPHABET_DE} ]"),
            "ES": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}{ALPHABET_ES} ]"),
            "RU": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION} ]"),
            "EN": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION} ]"),
            "KK": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}{ALPHABET_KK} ]"),
            "UZ": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}{ALPHABET_UZ}' ]"),
        }
        self._rm_double_space = re.compile(r"\s+")

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        _str = sent.text

        if kwargs["lang"] in self._clear:
            _str = self._clear[kwargs["lang"]].sub("", _str).strip()

        _str = self._rm_double_space.sub(r" ", _str)
        _str = " " + _str.strip() + " "

        _str = _str.lower().replace(" ", "  ")

        sent.text = _str


class SymbolsModifier(BaseRawTextProcessor):
    def __init__(self):
        vocabs_dir = get_root_dir() / "data/common/vocabularies"
        self._punctuation_symbols_vocab = Utils.read_vocab(
            vocabs_dir / "punctuation_symbols.txt"
        )
        self._alphabet_symbols_vocab = Utils.read_vocab(
            vocabs_dir / "alphabet_symbols.txt"
        )

        self._sub_patterns = [
            (re.compile(r"([\d])(\s*)(\°F)([\s]*)"), r"\1\3\4"),
            (re.compile(r"([\d])(\s*)(\°[CС])([\s]*)"), r"\1°C\4"),
            (re.compile(r"(\w)—(\w)"), r"\1-\2"),
            (re.compile(r"(\s)--(\s)"), r"\1—\2"),
            (re.compile(r"(\s)---(\s)"), r"\1,-\2"),
            (re.compile(r"—(\d+)"), r"-\1"),
        ]
        self._sub_patterns_ru = [
            (
                re.compile(
                    r"([\d]+)(\s*| тыс | млн | млрд | трлн )([\€\$\£\€\¥\₽])([\s]*)"
                ),
                r"\3\1\2\4",
            ),
        ]
        self._sub_patterns_en = [
            (re.compile(r"&"), r" and "),
        ]
        self._find_multi_blank = re.compile(r"(\s)(-){4,}(\s)")

    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        _str = doc.text

        for k, v in self._punctuation_symbols_vocab.items():
            _str = _str.replace(k, v)

        if kwargs["lang"] in ["RU", "EN"]:
            for k, v in self._alphabet_symbols_vocab.items():
                _str = _str.replace(k, v)

        for pattern, replasment in self._sub_patterns:
            _str = pattern.sub(replasment, _str)

        math = re.search(self._find_multi_blank, _str)
        while math:
            a, b = math.regs[0]
            _str = _str[:a] + f" <break time='{(b - a - 2) * 100}ms'/> " + _str[b:]
            math = re.search(self._find_multi_blank, _str)

        if kwargs["lang"] == "RU":
            for pattern, replasment in self._sub_patterns_ru:
                _str = pattern.sub(replasment, _str)
        elif kwargs["lang"] == "EN":
            for pattern, replasment in self._sub_patterns_en:
                _str = pattern.sub(replasment, _str)

        doc.text = _str


class TextModifier(BaseRawTextProcessor):
    space_symbol = "ЪСЪ"  #
    hyphen_symbol = "ЪХЪ"  # -
    custom_stress = "ЪУЪ"  # +

    specific_symbols = {
        "-": (
            "Х",
            [
                r"(\d)[\-–−]([a-zA-Zа-яёА-ЯЁ])",
                r"(\s)[\-–−](\d)",
                r"(\d)[\-–−](\d)",
                r"(\w)[\-–−](\d)",
                r"(\d)[\-–−](\w)",
            ],
        ),
        "+": ("Р", [r"(\s)\+(\d)"]),
        ",": ("ДС", [r"(\d)[,](\d)"]),
        ".": ("ДР", [r"(\d)[.](\d)"]),
        "№": ("Н", [r"(\s)№(\d)"]),
        # "@": ("ПЧ", [r"([\w\d])[@]([\w\d])"]),
        "$": ("ВД", [r"([\d])\$([\s]*)", r"([\s])\$([\d])"]),
        "R$": ("ВБ", [r"([\d])R\$([\s]*)", r"([\s])R\$([\d])"]),
        "€": ("ВЕ", [r"([\d])\€([\s]*)", r"([\s])\€([\d])"]),
        "¥": ("ВЮ", [r"([\d])\¥([\s]*)", r"([\s])\¥([\d])"]),
        "₽": ("ВР", [r"([\d])\₽([\s]*)", r"([\s])\₽([\d])"]),
        "£": ("ВФ", [r"([\d])\£([\s]*)", r"([\s])\£([\d])"]),
        "с": ("ВС", [r"([\d])(с|С)([\s]*)", r"([\s])(с|С)([\d])"]),
        "%": ("ПЧ", [r"([\d])\%([\s]*)"]),
        "#": ("РШ", [r"([\s])\#([a-zA-Zа-яёА-ЯЁ])"]),
        # "#": ("РШ", [r"|(\d+)\#"]),
        "°": ("Г", [r"([\d])\°([\s]*)"]),
        ":": ("Т", [r"(\d)\:(\d)"]),
        "°F": ("ГФ", [r"([\d\s])\°F([\s]*)"]),
        "°C": ("ГЦ", [r"([\d\s])\°C([\s]*)"]),
        "/": ("СК", [r"(км|м|см|мм|\d+)/(ч|мин|с|\d+)"]),
        "'": ("АП", [r"([a-zA-Z])[\'’]([a-zA-Z])"]),
        "(": ("ОС", [r"(\d)\((\d)"]),
        ")": ("ЗС", [r"(\d)\)(\d)"]),
        "*": ("З", [r"([\s]*)\*([\d])", r"([\d])\*([\s]*)"]),
    }

    def __init__(self):
        self._preliminary_correction_1 = re.compile(r"Ъ[А-ЯЁ]+Ъ")
        self._preliminary_correction_2 = re.compile(r"([a-zA-Zа-яёА-ЯЁ])\+([\d])")
        self._preliminary_correction_3 = re.compile(r"([a-zA-Zа-яёА-ЯЁ])([\d])")
        self._preliminary_correction_4 = re.compile(r"([\d])([a-zA-Zа-яёА-ЯЁ])")
        self._preliminary_correction_5 = re.compile(r"(\s+)([№$€¥£₽])(\s+)([\d\.,]+)")
        self._preliminary_correction_6 = re.compile(r"([\d\.,]+)(\s+)([%°$€¥£₽])(\s+)")
        self._remove_url = re.compile(
            r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b", flags=re.MULTILINE
        )

        self._hyphen_word_correction = re.compile(
            r"([a-zA-Zа-яёА-ЯЁ])\-([a-zA-Zа-яёА-ЯЁ])"
        )
        self._stress_correction = re.compile(r"([a-zA-Zа-яёА-ЯЁ])\+([a-zA-Zа-яёА-ЯЁ]*)")

        self._specific_symbols_correction = {}
        for symb, val in self.specific_symbols.items():
            for i, reg in enumerate(val[1]):
                val[1][i] = re.compile(reg)
            self._specific_symbols_correction[symb] = (f"Ъ{val[0]}Ъ", val[1])

    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        collect_ssml(doc)

        _str = " " + doc.text.strip() + " "
        _str = _str.replace("USD", "$").replace("EUR", "€")

        _str = self._preliminary_correction_1.sub("", _str)
        _str = self._preliminary_correction_2.sub(r"\1 \2", _str)
        _str = self._preliminary_correction_3.sub(r"\1 \2", _str)
        _str = self._preliminary_correction_4.sub(r"\1 \2", _str)
        _str = self._preliminary_correction_5.sub(r"\1\2\4 ", _str)
        _str = self._preliminary_correction_6.sub(r"\1\3\4", _str)
        _str = self._remove_url.sub("", _str)

        _str = self._hyphen_word_correction.sub(r"\1" + self.hyphen_symbol + r"\2", _str)
        _str = self._stress_correction.sub(r"\1" + self.custom_stress + r"\2", _str)

        for symb, val in self._specific_symbols_correction.items():
            for reg in val[1]:
                for _ in range(2):
                    _str = reg.sub(r"\1" + val[0] + r"\2", _str)

        doc.text = _str

    @classmethod
    def _clean_text(cls, text: str) -> str:
        def replace(text: str, sub: str, rep: str, as_lower: bool):
            if as_lower:
                return text.replace(sub.lower(), rep.lower())
            else:
                return text.replace(sub, rep)

        text = replace(text, cls.space_symbol, " ", False)
        text = replace(text, cls.space_symbol, " ", True)

        text = replace(text, cls.hyphen_symbol, "-", False)
        text = replace(text, cls.hyphen_symbol, "-", True)

        text = replace(text, cls.custom_stress, "+", False)
        text = replace(text, cls.custom_stress, "+", True)

        for symb, val in cls.specific_symbols.items():
            symb = "" if val[0] == "РШ" else symb
            text = replace(text, f"Ъ{val[0]}Ъ", symb, False)
            text = replace(text, f"Ъ{val[0]}Ъ", symb, True)

        restor_text = re.compile(r"Ъ[a-zA-Zа-яёА-ЯЁ]+Ъ").sub("", text)
        return restor_text

    @classmethod
    def restore(cls, doc: Doc, **kwargs) -> Doc:
        if doc.sents is None:
            doc.text = cls._clean_text(doc.text)
        else:
            for sent in doc.sents:
                if sent.tokens is None:
                    sent.text = cls._clean_text(sent.text)
                else:
                    for token in sent.tokens:
                        if "ЪП" in token.text:
                            token.is_preposition = True
                            token.attr = {"case": token.text[2:4]}

                        elif "ършъ" in token.text:
                            token.emphasis = "accent"

                        token.text = cls._clean_text(token.text)

                    sent.text = sent.text

                sent.text_orig = cls._clean_text(sent.text_orig)

        return doc
