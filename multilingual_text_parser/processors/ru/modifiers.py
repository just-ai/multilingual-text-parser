from pathlib import Path

import regex as re

from multilingual_text_parser._constants import PUNCTUATION
from multilingual_text_parser.data_types import Doc, Sentence
from multilingual_text_parser.processors.base import (
    BaseRawTextProcessor,
    BaseSentenceProcessor,
)
from multilingual_text_parser.processors.common.modifiers import TextModifier
from multilingual_text_parser.processors.ru.rulebased_normalizer import Utils
from multilingual_text_parser.thirdparty.ru.e2yo.e2yo.core import E2Yo
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir
from multilingual_text_parser.utils.profiler import Profiler

__all__ = ["TextModifierRU", "SentencesModifierRU"]


class TextModifierRU(BaseRawTextProcessor):
    def __init__(self):
        vocabs_dir = get_root_dir() / "data/ru/vocabularies"
        self._abbreviations_with_point_vocab = Utils.read_vocab(
            vocabs_dir / "abbreviations_with_point.txt"
        )

        for vocab in [
            self._abbreviations_with_point_vocab,
        ]:
            for key in list(vocab.keys()):
                value = vocab.pop(key)
                key = key.replace("-", TextModifier.hyphen_symbol)
                value = (
                    value.replace(" ", TextModifier.space_symbol)
                    .replace("-", TextModifier.hyphen_symbol)
                    .replace("+", TextModifier.custom_stress)
                )
                vocab[f" {key}. "] = f" {value}. "
                if key[-1] == ".":
                    vocab[f" {key}"] = f" {value} "
                    vocab[f" {key} "] = f" {value} "
                else:
                    vocab[f" {key} "] = f" {value} "

        self._e2yo = E2Yo()
        self._reshetka = re.compile(r"([\d])\#([\s]*)")

    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        if len(doc.text.strip().strip(".")) == 1:
            return

        _str = " " + doc.text.strip() + " "
        _str = self._reshetka.sub(r"\1" + " решётка " + r"\2", _str)

        for word, replacement_word in self._abbreviations_with_point_vocab.items():
            _str = _str.replace(word, replacement_word)

        _str = self._e2yo.replace(_str)

        doc.text = _str


class SentencesModifierRU(BaseSentenceProcessor):
    preposition_dicts = {
        "рд": "genitive.txt",
        "дт": "dative.txt",
        "вн": "accusative.txt",
        "тв": "instrumental.txt",
        "пр": "prepositional.txt",
    }

    def __init__(self):
        self._clear = re.compile(f"[^a-zA-Zа-яёА-ЯЁ0-9{PUNCTUATION} ]")
        self._rm_end_hyphen = re.compile(rf"(\D+)\s+[-]\s*([{PUNCTUATION}]+)")
        self._rm_double_space = re.compile(r"\s+")

        vocabs_dir = get_root_dir() / "data/ru/vocabularies"

        preposition_words = []
        for case, file in self.preposition_dicts.items():
            vocab_path = vocabs_dir / file
            words = Utils.read_vocab(vocab_path, as_list=True)
            words = [(x, case) for x in words if len(x) > 0]
            preposition_words += words

        preposition_words.sort(key=lambda x: len(x[0]), reverse=True)

        self._preposition_vocab = {}
        for word, case in preposition_words:
            self._preposition_vocab[word] = f"ЪП{case}Ъ{word}"

        self._abbreviation_with_hyphens_vocab = Utils.read_vocab(
            vocabs_dir / "abbreviation_with_hyphens.txt"
        )
        self._abbreviation_stress_vocab_lower = Utils.read_vocab(
            vocabs_dir / "abbreviation_stress_lower.txt"
        )
        self._abbreviation_stress_vocab_upper = Utils.read_vocab(
            vocabs_dir / "abbreviation_stress_upper.txt"
        )

        self._en_translit_vocab = Utils.read_vocab(vocabs_dir / "en_translit.txt")
        self._interpret_as_vocab = Utils.read_vocab(vocabs_dir / "interpret_as.txt")

        for vocab in [
            self._preposition_vocab,
            self._abbreviation_with_hyphens_vocab,
            self._abbreviation_stress_vocab_lower,
            self._abbreviation_stress_vocab_upper,
            self._en_translit_vocab,
            self._interpret_as_vocab,
        ]:
            for key in list(vocab.keys()):
                value = vocab.pop(key)
                key = key.replace("-", TextModifier.hyphen_symbol)
                value = (
                    value.replace(" ", TextModifier.space_symbol)
                    .replace("-", TextModifier.hyphen_symbol)
                    .replace("+", TextModifier.custom_stress)
                )
                vocab[f" {key.lower()} "] = f" {value} "
                vocab[f" {key.lower().replace(' ', '  ')} "] = f" {value} "

        self._names_vocab = Utils.read_vocab(vocabs_dir / "names.txt")

        for vocab in [self._names_vocab]:
            for key in list(vocab.keys()):
                value = vocab.pop(key)
                key = key.replace("-", TextModifier.hyphen_symbol)
                value = (
                    value.replace(" ", TextModifier.space_symbol)
                    .replace("-", TextModifier.hyphen_symbol)
                    .replace("+", TextModifier.custom_stress)
                )
                vocab[f" {key} "] = f" {value} "

        self._sub_patterns = [
            (
                re.compile(r"([\d*\s])([^\W0-9ЪIVXLCDMа-яa-z]{2,4})([\d*\s])"),
                r'\g<1> <say-as interpret-as="abbreviation"> \g<2> </say-as>  \g<3>',
            ),
            (
                re.compile(r"(\d)(\s*)([^\W0-9ЪIVXLCDMа-яa-z]{1})\s"),
                r'\g<1> <say-as interpret-as="abbreviation"> \g<3> </say-as> ',
            ),
            (
                re.compile(r"\s([^\W0-9ЪIVXLCDM]{1})(\d)"),
                r'<say-as interpret-as="abbreviation"> \g<1> </say-as> \g<2>',
            ),
            (
                re.compile(r"(\D)(\p{P}+)(\s+)([^\W0-9ЪIVXLCDMа-яa-z]{1})\s"),
                r'\g<1>\g<2> <say-as interpret-as="abbreviation"> \g<4> </say-as> ',
            ),
            (
                re.compile(r"([а-яa-z])(\s+)([^\W0-9ЪIVXLCDMа-яa-z]{1})\s"),
                r'\g<1> <say-as interpret-as="abbreviation"> \g<3> </say-as> ',
            ),
            (
                re.compile(r"(\d+)(\s+)(\d{3})"),
                r"\g<1>\g<3>",
            ),
        ]

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        _str = self._clear.sub("", sent.text).strip()
        _str = self._rm_double_space.sub(r" ", _str)
        _str = " " + _str.strip() + " "

        for word, replacement_word in self._abbreviation_stress_vocab_upper.items():
            _str = _str.replace(word.upper(), replacement_word)

        for word, replacement_word in self._names_vocab.items():
            _str = _str.replace(word, replacement_word)

        for pattern, replasment in self._sub_patterns:
            _str = pattern.sub(replasment, _str)

        _str = self._rm_end_hyphen.sub(r"\1 \2", _str)
        _str = _str.lower().replace(" ", "  ")

        for vocab in [
            self._interpret_as_vocab,
            self._preposition_vocab,
            self._abbreviation_with_hyphens_vocab,
            self._abbreviation_stress_vocab_lower,
            self._en_translit_vocab,
        ]:
            if kwargs.get("disable_translit", False):
                if vocab == self._en_translit_vocab:
                    continue

            for word, replacement_word in vocab.items():
                if "*" in word:
                    replacement_word = replacement_word.replace("*", "\\1")
                    _str = re.sub(word.replace("*", r"(\w*)"), replacement_word, _str)
                else:
                    _str = _str.replace(word, replacement_word)

        sent.text = _str


if __name__ == "__main__":
    from multilingual_text_parser.processors.common.corrector import Corrector
    from multilingual_text_parser.processors.common.modifiers import SymbolsModifier
    from multilingual_text_parser.processors.common.tokenizer import Tokenizer

    # from text_parser.text_processing.processors.ru.sentenizer import Sentenizer
    from multilingual_text_parser.processors.ru.sentenizer import SentenizerRU

    symb_mode = SymbolsModifier()
    text_mode = TextModifier()
    sent_mode = SentencesModifierRU()

    sentenizer = SentenizerRU()
    tokenizer = Tokenizer()
    corrector = Corrector()

    with Profiler(format=Profiler.Format.ms):
        text = Text(
            "-40 $% * Á А hello \r\r\n"
            "ЪХЪ ЪВПЫВЪ из—за 3 - 3-х во-вторых, "
            "Во-вторых тако+м-то 1240-ом 8-909-234-43-22 1990-2000-x гг "
            "10:34 43.3422 324,3532: "
            "лицом к лицу со страхом. "
            "Лицом к лицу со страхом; "
            "унес жизнь ф-том МВД СО COVID-19 pdf *.PDF"
            "232 ---**--- 232 -43 "
            "Еж еще был здесь ЦРУ"
        )

        text = text_mode(symb_mode(text))
        text = sentenizer(corrector(text))
        text = tokenizer(sent_mode(text))
        text = text_mode.restore(text)

    print(text.tokens)
