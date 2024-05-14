import re
import typing as tp
import logging

import num2words

from multilingual_text_parser.data_types import Sentence, Token
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["NormalizerPTBR"]

LOGGER = logging.getLogger("root")


class NormalizerPTBR(BaseSentenceProcessor):
    def __init__(self):
        self._lang = "pt_BR"
        self._currencies = {
            "$": ("dólar", "dólares"),
            "€": ("euro", "euros"),
            "¥": ("yuan", "yuan"),
            "£": ("libra", "libras"),
        }
        self._symbols = {
            "%": " por cento ",
            "№": " número ",
            "+": " mais ",
            "-": " menos ",
        }
        self._endings = {
            "m": "o",
            "f": "a",
            "s": "",
            "p": "s",
        }
        self._months = [
            "janeiro",
            "fevereiro",
            "março",
            "abril",
            "maio",
            "junho",
            "julho",
            "agosto",
            "setembro",
            "outubro",
            "novembro",
        ]
        self._nums = re.compile(r"(\d+)?[.,]?\d+")
        self._symbols_pattern = re.compile(r"^[%№+-]|[%№+-]$")
        self._ordinal = re.compile(r"\d+(º|-o|o|ª|-a|a|-os|os|-as|as)")
        self._currency = re.compile(r"[$€¥£]\d+|\d+([$€¥£]|r$)")
        self._years = re.compile(r"([0-2]?\d{3})\-([0-2]?\d{3})")

        # регулярки на разные форматы дат dmy/mdy/ymd...
        self._date1 = re.compile(
            r"^([0-2]?\d|3[01])[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d{3}|\d{2}))$"
        )
        self._date2 = re.compile(
            r"^(0?\d|1[0-2])[\./\-]([0-2]?\d|3[01])([\./\-](([0-2]?\d{3})|\d{2}))$"
        )
        self._date3 = re.compile(
            r"^([0-2]?\d{3}|\d{2})[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d|3[01]))$"
        )
        self._date4 = re.compile(
            r"^([0-2]?\d{3}|\d{2})[\./\-]([0-2]?\d|3[01])([\./\-](0?\d|1[0-2]))$"
        )

        # регулярки для порядковых числительных
        self._ordinal_mp = re.compile(r"\d+(-os|os)")
        self._ordinal_fp = re.compile(r"\d+(-as|as)")
        self._ordinal_ms = re.compile(r"\d+(º|-o|o)")
        self._ordinal_fs = re.compile(r"\d+(ª|-a|a)")

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        tokens = sent.tokens
        for token in tokens:
            token.text = self._symbols_pattern.sub(
                lambda x: self._symbols[x.group()], token.text
            )

            date = self.convert_date(token)
            if date:
                token.text = date
            elif self._currency.search(token.text):
                token.text = self.convert_currency(token.text)
            elif self._ordinal.search(token.text):
                token.text = self.convert_ordinal(token.text)
            elif self._years.search(token.text):
                token.text = self.convert_year_period(token.text)
            elif token.is_number:
                num = token.text.replace(",", ".")
                if token.interpret_as and token.interpret_as == "ordinal":
                    token.text = self._nums.sub(
                        lambda x: num2words.num2words(
                            x.group(), lang=self._lang, to="ordinal"
                        ),
                        num,
                    )
                if float(self._nums.search(num).group()) < 1000000000000000000:
                    token.text = self._nums.sub(
                        lambda x: num2words.num2words(x.group(), lang=self._lang), num
                    )
                else:
                    token.text = self._nums.sub(
                        lambda x: self.convert_digits(x.group()), num
                    )

        tokens = self._split_words(tokens, symb=" ")
        tokens = self._split_words(tokens, symb="-")
        sent.tokens = tokens

    def convert_currency(self, text):
        num = self._nums.search(text).group()
        cur = re.search("[$€¥£]", text).group()
        currency = num2words.num2words(num, lang=self._lang, to="currency")
        if "r$" not in text:
            currency = re.sub("real", self._currencies[cur][0], currency)
            return re.sub("reais", self._currencies[cur][1], currency)
        else:
            return currency

    def convert_ordinal(self, text):
        gender, number = None, None
        if self._ordinal_mp.search(text):
            gender, number = "m", "p"
        elif self._ordinal_fp.search(text):
            gender, number = "f", "p"
        elif self._ordinal_ms.search(text):
            gender, number = "m", "s"
        elif self._ordinal_fs.search(text):
            gender, number = "f", "s"
        num = self._nums.search(text)
        if num:
            text = " ".join(
                [
                    num[:-1] + self._endings[gender] + self._endings[number]
                    for num in num2words.num2words(
                        num.group(), lang=self._lang, to="ordinal"
                    ).split()
                ]
            )
        return text

    def convert_date(self, token):
        date2str = ""
        day, month, y = None, None, None
        format = token.format

        if self._date1.search(token.text) and (not format or format in ["dmy", "dm"]):
            date = self._date1.search(token.text)
            day, month, y = date[1], date[2], date[4]
        elif self._date2.search(token.text) and (not format or format in ["mdy", "md"]):
            date = self._date2.search(token.text)
            day, month, y = date[2], date[1], date[4]
        elif self._date3.search(token.text) and (not format or format in ["ymd", "ym"]):
            date = self._date3.search(token.text)
            day, month, y = date[4], date[2], date[1]
        elif self._date4.search(token.text) and (not format or format in ["ydm", "yd"]):
            date = self._date4.search(token.text)
            day, month, y = date[2], date[4], date[1]
        elif format:
            day, month, y = self.split_wo_sep(token.text, format)

        if day:
            if day == "1":
                date2str += "primeiro"
            else:
                date2str += "no dia " + num2words.num2words(day, lang=self._lang)
            if month:
                date2str += " de " + self._months[int(month) - 1]
        elif month:
            date2str += "em " + self._months[int(month) - 1]
        if y:
            date2str += " de " + num2words.num2words(y, lang=self._lang)
        return date2str

    def convert_year_period(self, text):
        date2str = ""
        y = self._years.search(text)
        date2str += (
            "de "
            + num2words.num2words(y[1], lang=self._lang)
            + " a "
            + num2words.num2words(y[2], lang=self._lang)
        )
        return date2str

    def convert_digits(self, text):
        result = []
        for char in text:
            result.append(num2words.num2words(char, lang=self._lang))
        return " ".join(result)

    @staticmethod
    def _split_words(tokens: tp.List[Token], symb: str) -> tp.List[Token]:
        new_tokens = []
        for token in tokens:
            if not token.is_punctuation and symb in token.text.strip(symb):
                words = token.text.split(symb)  # type: ignore
                words = [x for x in words if len(x) > 0]
                if token.stress:
                    words_stress = token.stress.split(symb)  # type: ignore
                    words = [x for x in words_stress if len(x) > 0]
                    assert len(words) == len(words_stress)
                else:
                    words_stress = [None] * len(words)

                for word, stress in zip(words, words_stress):
                    new_token = Token(word)
                    for attr in [
                        "pos",
                        "modifiers",
                        "normalized",
                        "emphasis",
                        "id",
                    ]:
                        value = getattr(token, attr)
                        setattr(new_token, attr, value)
                    if stress:
                        new_token.stress = stress
                    new_tokens.append(new_token)
            else:
                new_tokens.append(token)

        return new_tokens

    @staticmethod
    def split_wo_sep(text: str, format: str):
        d = m = y = "0"
        for f in format:
            if f == "d":
                d = text[:2]
                text = text[2:]
            if f == "m":
                m = text[:2]
                text = text[2:]
            if f == "y":
                y = text[:4]
                text = text[4:]
        return d, m, y
