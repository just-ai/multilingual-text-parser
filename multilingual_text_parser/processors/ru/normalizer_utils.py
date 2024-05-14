import re
import base64
import typing as tp
import logging

from pathlib import Path

from multilingual_text_parser.data_types import Sentence, Token
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.processors.ru.morph_analyzer import MorphAnalyzerRU
from multilingual_text_parser.processors.ru.num_to_words import NumToWords
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.log_utils import trace

__all__ = ["BaseRule", "BaseNormalizer", "Utils"]

LOGGER = logging.getLogger("root")


class BaseRule:
    def __init__(self, morph: MorphAnalyzerRU, num2words: NumToWords):
        self._morph = morph
        self._num2words = num2words

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        pass

    def __call__(self, sent: Sentence, **kwargs):
        if kwargs["tagged"]:
            start = kwargs["token_idx_start"]
            end = kwargs["token_idx_end"]
            self.process(0, sent.tokens, start=start, end=end)
        else:
            idx = 0
            while idx < len(sent.tokens):
                self.process(idx, sent.tokens, start=0, end=0)
                idx += 1


class BaseNormalizer(BaseSentenceProcessor):
    def __init__(self):
        self._morph = MorphAnalyzerRU()
        self._num2words = NumToWords(self._morph)
        self._rules_ssml: tp.List[tp.Callable] = []
        self._rules_begin: tp.List[tp.Callable] = []
        self._rules: tp.List[tp.Callable] = []
        self._rules_tagged: tp.List[tp.Callable] = []

    @staticmethod
    def _split_words(
        tokens: tp.List[Token],
        symb: str,
        ssml_insertions_pos: tp.Optional[tp.List] = None,
    ) -> tp.List[Token]:
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
                        "head_id",
                        "rel",
                        "meta",
                    ]:
                        value = getattr(token, attr)
                        setattr(new_token, attr, value)
                    if stress:
                        new_token.stress = stress
                    new_tokens.append(new_token)

                if ssml_insertions_pos:
                    if token in ssml_insertions_pos:
                        idx = ssml_insertions_pos.index(token)
                        ssml_insertions_pos.remove(token)
                        ssml_insertions_pos.insert(idx, new_tokens[-1])
            else:
                new_tokens.append(token)

        return new_tokens

    @staticmethod
    def _clean(tokens: tp.List[Token]) -> tp.List[Token]:
        new_tokens = []
        find_digit = re.compile(r"[\d]")
        for token in tokens:
            if token.is_punctuation or not find_digit.findall(token.text):
                new_tokens.append(token)

        return new_tokens

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        # Сначала применяются правила, активируемые SSML-разметкой
        for rule in self._rules_ssml:
            try:
                rule(
                    sent,
                    **{
                        "tagged": False,
                    },
                )
            except Exception as e:
                sent.warning_messages.append(trace(self, e))

        if sent.tokens[0].tag:
            # Затем нормализуются токены по очереди по специфичным правилам (телефон и числительные с прилагательными)
            for rule in self._rules_begin:
                try:
                    rule(
                        sent,
                        **{
                            "tagged": False,
                        },
                    )
                except Exception as e:
                    sent.warning_messages.append(trace(self, e))

            idx = 0
            tags = ["date", "time", "roman", "ordinal", "digit", "fraction"]
            while idx < len(sent.tokens):
                start = idx
                if sent.tokens[idx].tag in tags:
                    tag = sent.tokens[idx].tag
                    while idx < len(sent.tokens) and sent.tokens[idx].tag == tag:
                        idx += 1
                    self._rules_tagged[tag](
                        sent,
                        **{
                            "tagged": True,
                            "token_idx_start": start,
                            "token_idx_end": idx,
                        },
                    )
                    if idx >= len(sent.tokens):
                        break
                    while sent.tokens[idx].tag == tag:
                        idx -= 1
                    while sent.tokens[idx].tag != tag:
                        idx -= 1
                idx += 1

        # Дальше все токены по очереди нормализуются другими правилами
        for rule in self._rules:
            try:
                if (
                    kwargs.get("disable_translit", False)
                    and rule.__class__.__name__ == "Translit"
                ):
                    continue

                rule(
                    sent,
                    **{
                        "tagged": False,
                    },
                )
            except Exception as e:
                sent.warning_messages.append(trace(self, e))

        if sent.ssml_insertions:
            ssml_insertions_pos = [
                sent.tokens[idx] if idx >= 0 else None for idx, _ in sent.ssml_insertions
            ]
        else:
            ssml_insertions_pos = []

        tokens = sent.tokens
        tokens = self._split_words(
            tokens, symb=" ", ssml_insertions_pos=ssml_insertions_pos
        )
        tokens = self._split_words(
            tokens, symb="-", ssml_insertions_pos=ssml_insertions_pos
        )
        tokens = self._clean(tokens)
        sent.tokens = tokens

        if sent.ssml_insertions:
            for idx, (item, token) in enumerate(
                zip(sent.ssml_insertions, ssml_insertions_pos)
            ):
                if item[0] >= 0:
                    new_pos = tokens.index(token)
                    sent.ssml_insertions[idx] = (new_pos, item[1])


class Utils:
    @staticmethod
    def read_vocab(
        vocab_path: Path,
        as_list: bool = False,
        add_capitalize: bool = False,
        add_title: bool = False,
        add_uppercase: bool = False,
    ) -> tp.Union[tp.List[str], tp.Dict[str, str]]:
        def _read(path: Path) -> str:
            if not path.exists():
                path = path.with_suffix(".bin")
            if not path.exists():
                raise FileNotFoundError(f"File {path.as_posix()} not found!")
            if path.suffix == ".bin":
                text = base64.b64decode(path.read_bytes()).decode()
            else:
                text = path.read_text(encoding="utf-8")
            return text

        if as_list:
            vocab = _read(vocab_path)
            if "," in vocab:
                vocab_as_list = vocab.split(",")
            elif ";" in vocab:
                vocab_as_list = vocab.split(";")
            else:
                vocab_as_list = vocab.split("\n")
            return vocab_as_list
        else:
            vocab = _read(vocab_path).split("\n")
            vocab_as_dict = {}
            for item in vocab:
                if item.startswith("#"):
                    continue
                elif "=" in item:
                    keys, value = item.split("=")
                    keys, value = keys.strip(), value.strip()
                    keys = keys.split(";")
                    for key in keys:
                        key = key.strip()
                        if len(key) == 0:
                            continue
                        vocab_as_dict[key] = value
                        if add_capitalize:
                            key = key[0].capitalize() + key[1:]
                            vocab_as_dict[key] = value
                        if add_title:
                            vocab_as_dict[key.title()] = value
                        if add_uppercase:
                            vocab_as_dict[key.upper()] = value
                elif "+" in item:
                    vocab_as_dict[item.replace("+", "")] = str(item.find("+"))

            return vocab_as_dict

    @staticmethod
    def find_prepositions(
        tokens: tp.List[Token],
        idx: int,
        left: int = 4,
        right: int = 0,
        all: bool = True,
    ) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
        preposition = {}
        a = max(0, idx - left)
        b = min(len(tokens), idx + right)
        for i in reversed(range(a, b)):
            if tokens[i].is_punctuation:
                break
            if tokens[i].is_preposition:
                # if tokens[i].norm in ["у", "о", "из"] and a != idx - 1:
                #     continue
                case = tokens[i].attr["case"]  # type: ignore
                if tokens[i].norm in ["в", "во", "за", "на"]:
                    case = "пр2"
                if tokens[i].norm in ["по"]:
                    case = "им"
                preposition[tokens[i].text] = {
                    "text": tokens[i].norm,
                    "case": case,
                    "pos": i,
                }
                if not all:
                    break
            elif tokens[i].pos in ["PRON", "NOUN"]:
                break

        return preposition
