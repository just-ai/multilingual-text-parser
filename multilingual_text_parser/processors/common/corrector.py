import regex as re

from multilingual_text_parser._constants import (
    ALPHABET_DE,
    ALPHABET_ES,
    ALPHABET_KK,
    ALPHABET_UZ,
    PUNCTUATION,
    PUNCTUATION_LEFT,
    PUNCTUATION_RIGHT,
)
from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.processors.base import BaseRawTextProcessor
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["Corrector"]


class Corrector(BaseRawTextProcessor):
    def __init__(self):
        self._patterns = [
            (r"\+", ""),
            (r" ", ""),
            (r'[«»\'”`"\\/|]+', " "),
            (r"[—­]+", "—"),
            (r"_+", " "),
            ("…", "."),
            (r"([\s\t\n\r])([ьъЬ]+)", " "),
            (r"(-\w){2,}", r""),
            (r"[\[{]", r"("),
            (r"[\]}]", r")"),
        ]
        self._clear = {
            "DE": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}\\s\t\n\r{ALPHABET_DE}]"),
            "ES": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}\\s\t\n\r{ALPHABET_ES}]"),
            "RU": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}\\s\t\n\r]"),
            "EN": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}\\s\t\n\r]"),
            "KK": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}\\s\t\n\r{ALPHABET_KK}]"),
            "UZ": re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION}\\s\t\n\r{ALPHABET_UZ}']"),
        }

        self._add_space_1 = re.compile(f"([{PUNCTUATION}])")
        self._add_space_2 = re.compile(r"(\d)([a-zA-Z])|([a-zA-Z])(\d)")
        self._duplicate_punct_1 = re.compile(f"([{PUNCTUATION}\\s\t\n\r])\\1+")
        self._duplicate_punct_2 = re.compile(f"( [{PUNCTUATION}])\\1+")
        self._rm_double_space = re.compile(r"\s+")

        self._punct_corr_0 = re.compile("(\\w+|[,:;])(\n|\r\n)(\\w+)")
        self._punct_corr_1 = re.compile("(\\w+)(\\s+\n+|\n+\\s+)([^A-ZА-Я0-9])")
        self._punct_corr_2 = re.compile("(\\w+)(\\s*\n+\\s*)([A-ZА-Я0-9])")
        self._punct_corr_3 = re.compile("(\\w+)(\\s*\r\n+\\s*)([A-ZА-Я0-9])")
        self._punct_corr_4 = re.compile(r"([a-zа-яё0-9])(\s+)([a-zа-яё0-9])")

        self._remove_serv_symbs = re.compile("([\\s\t\n\r ])+")

    @classmethod
    def trim_punctuation(cls, _str: str):
        _str_tmp1 = re.sub(r"^[\W]*", r"", _str)
        start_punct = _str[: -len(_str_tmp1)]
        start_punct = re.sub(f"[^{PUNCTUATION_LEFT} ]+", r"", start_punct)

        if _str and not _str[0].isnumeric():
            start_punct = "" if "(" not in start_punct else "("

        _str_tmp2 = re.sub(r"[\W]*$", r"", _str_tmp1)
        end_punct = _str_tmp1[len(_str_tmp2) :]
        end_punct = re.sub(f"[^{PUNCTUATION_RIGHT} ]+", r"", end_punct)
        end_punct = end_punct.replace(",", "")
        if end_punct == "":
            end_punct = "."

        return f"{start_punct} {_str_tmp2} {end_punct}"

    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        _str = doc.text
        for p, t in self._patterns:
            _str = re.sub(p, t, _str)

        # remove non-alphabetic characters
        if kwargs["lang"] in self._clear:
            _str = self._clear[kwargs["lang"]].sub("", _str).strip()
        if kwargs.get("add_trailing_punct_token", True):
            _str = self.trim_punctuation(_str)

        _str = self._punct_corr_0.sub(r"\1 \3", _str)

        _str = self._add_space_1.sub(r" \1 ", _str)
        _str = self._add_space_2.sub(r"\1 \2", _str)

        # remove duplicate punctuation
        _str = self._duplicate_punct_1.sub(r"\1", _str)
        _str = self._duplicate_punct_2.sub(r"\1", _str)

        # sentence splitting correction
        _str = self._punct_corr_1.sub(r"\1 \3", _str)
        _str = self._punct_corr_2.sub(r"\1. \3", _str)
        _str = self._punct_corr_3.sub(r"\1. \3", _str)
        _str = self._punct_corr_4.sub(r"\1 \3", _str)

        _str = self._remove_serv_symbs.sub(r" ", _str)

        _str = self._add_space_1.sub(r" \1 ", _str)
        _str = self._rm_double_space.sub(r" ", _str)

        _str = _str.strip()

        doc.text = _str


if __name__ == "__main__":
    corrector = Corrector()

    doc = Doc(
        """
        - -! (. . .. Так, в
        ближа#йшее время)     к плановому __ [приему] пациентов вернутся   городские
           больницы №№ 2 и 33,,,,,, клиническая больница святителя Луки    ,
        а также психиатрическая больница имени Кащенко.!!
        В «  Ленэкспо» закрыт госпиталь
        в павильоне № 7, пациенты остались только в павильоне № 5..... . .  . .

        Я чувствовал себя абсолютно бестолковым, пока инструктор не похвалил меня за какую - то мелочь ь - ! - -,?
        из-за угла показалась зелено-фиолетовая дорожка . Она петляла туда- сюда и приходилось - внимательно -наблюдать.
        из-за угла показалась зелено-фиолетовая . Она петляла туда- сюда. И - отдельно -наблюдать.
        65-я годовщина. 65-ясная годовщина. Ты-4434. 4-х.
        путь-дорога
        """
    )
    print(corrector(doc).text)
