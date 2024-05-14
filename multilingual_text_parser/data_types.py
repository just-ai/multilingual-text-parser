import re
import enum
import uuid
import typing as tp
import itertools

from copy import deepcopy
from string import punctuation

from natasha import Doc as NatashaDoc
from natasha import Segmenter
from natasha.doc import DocSent, DocToken

import multilingual_text_parser

from multilingual_text_parser._constants import (
    PUNCTUATION,
    PUNCTUATION_ALL,
    PUNCTUATION_DASH,
    PUNCTUATION_LEFT,
)

__all__ = ["Position", "Token", "Syntagma", "Sentence", "Doc", "TokenUtils"]


class Position(enum.Enum):
    first = 0
    internal = 1
    last = 2


class Token(DocToken):
    def __init__(self, token: tp.Union[str, DocToken]):
        self.__uuid = uuid.uuid4().hex
        self._text: str = ""
        self.text_orig: str = ""
        self.norm: str = ""
        self.emphasis: str = "no"
        self.stress: tp.Optional[tp.Union[str, tp.List[str]]] = None
        self.phonemes: tp.Optional[tp.Tuple[str, ...]] = None
        self.attr: tp.Optional[dict] = None
        self.is_preposition: bool = False
        self.is_sub: bool = False
        self.is_capitalize: bool = False
        self.is_name: bool = False
        self.normalized: bool = False
        self.modifiers = None
        self.interpret_as = None
        self.tag = None
        self.format = None
        self.id: str = None  # type: ignore
        self.token_id: str = None  # type: ignore
        self.rel: str = None  # type: ignore
        self.meta: tp.Dict[str, tp.Any] = {}
        self.from_ssml: bool = False
        self.prosody: tp.Optional[int] = None
        self.asr_pause: tp.Optional[str] = None

        if isinstance(token, str):
            super().__init__(None, None, token)
        else:
            super().__init__(*token.__dict__.values())

    def __len__(self) -> int:
        return len(self.text) if self.text else 0

    def __copy__(self):
        new_token = Token("")
        new_token.__dict__ = self.__dict__.copy()
        new_token.__uuid = uuid.uuid4().hex
        return new_token

    def __deepcopy__(self, memo):
        new_token = Token("")
        new_token.__dict__ = deepcopy(self.__dict__)
        new_token.__uuid = uuid.uuid4().hex
        return new_token

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.__uuid == other.__uuid
        else:
            return False

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str):
        if text is not None:
            if "+" in text and not text.startswith("+"):
                self.stress = text
                text = text.replace("+", "")
            else:
                self.stress = None
            self._text = text
            self.norm = self.remove_punctuation(text)

    @property
    def is_punctuation(self) -> bool:
        if self.norm is not None:
            return self.text != "" and len(self.norm) == 0
        else:
            return False

    @property
    def is_number(self) -> bool:
        if self.norm is not None:
            return re.search(r"\d", self.norm) is not None
        else:
            return False

    @property
    def is_abbreviation(self) -> bool:
        if self.text_orig is not None:
            return (
                not self.is_number
                and len(self.text_orig) > 1
                and self.text_orig == self.text_orig.upper()
            )
        else:
            return False

    @property
    def is_yo_ambiguous(self) -> bool:
        if self.norm is not None:
            return re.search("[eё]", self.norm) is not None
        else:
            return False

    @property
    def is_pause(self) -> bool:
        return "<SIL>" in self.text

    @property
    def is_word(self) -> bool:
        return (not (self.is_punctuation or self.is_pause)) and self.text != ""

    @property
    def num_phonemes(self) -> int:
        if self.phonemes is not None:
            return len(self.phonemes)
        else:
            return 0

    @staticmethod
    def remove_punctuation(text: str) -> str:
        text = text.lower().strip(" ")
        text = re.sub(f"([{PUNCTUATION_ALL}])", "", text)
        text = text.replace("ё", "е")
        return text

    def to_dict(self):
        return {
            "text": self.text,
            "norm": self.norm,
            "pos": self.pos,
            "accent": self.stress,
            "phonemes": self.phonemes,
        }


class TokenUtils:
    @staticmethod
    def group_tokens_by_word(
        tokens: tp.List[Token], sil_as_word: bool = True
    ) -> tp.List[tp.List[Token]]:
        group: tp.List[tp.List[Token]] = []
        punct_first = False
        for token in tokens:
            if (sil_as_word and not token.is_punctuation) or (
                not sil_as_word and not token.is_punctuation and not token.is_pause
            ):
                if (
                    group
                    and group[-1][-1].is_punctuation
                    and set(group[-1][-1].text) & set(PUNCTUATION_LEFT)
                    or punct_first
                ):
                    group[-1].append(token)
                else:
                    group.append([token])
                punct_first = False
            else:
                if set(token.text) & set(PUNCTUATION_LEFT):
                    group.append([token])
                else:
                    if group:
                        group[-1].append(token)
                    else:
                        group.append([token])
                        punct_first = True
        return group

    @staticmethod
    def get_text_from_tokens(
        tokens: tp.List[Token], as_norm: bool = False, with_capitalize: bool = False
    ) -> str:
        token_group = TokenUtils.group_tokens_by_word(tokens)

        words = []
        for tokens in token_group:
            if as_norm:
                word = [t.norm for t in tokens]
            else:
                word = []
                for t in tokens:
                    if not with_capitalize:
                        word.append(t.text)
                        continue

                    if tokens == token_group[0]:
                        if t == tokens[0] and t.is_word:
                            word.append(t.text.capitalize())
                            continue

                    if t.is_name or t.is_capitalize:
                        word.append(t.text.capitalize())
                        continue

                    word.append(t.text)

                    if t == tokens[0] and t.text in PUNCTUATION_DASH:
                        word.append(" ")

            words.append("".join(word))

        return " ".join(words)

    @staticmethod
    def get_stress_from_tokens(tokens: tp.List[Token]) -> str:
        token_group = TokenUtils.group_tokens_by_word(tokens)
        words = []
        for tokens in token_group:
            word = []
            for t in tokens:
                if t.stress:
                    t.stress = [t.stress] if not isinstance(t.stress, list) else t.stress
                    for idx, t_stress in enumerate(t.stress):
                        if idx == 0:
                            word.append(t_stress)
                        else:
                            word.append(" {" + t_stress + "} ")
                else:
                    word.append(t.text)

            joined = "".join(word).replace("}  {", ", ")
            words.append(joined)

        return " ".join(words)

    @staticmethod
    def _get_attr(
        token_group: tp.List[tp.List[Token]],
        attr_names: tp.List[str],
        with_punct: bool = False,
    ) -> tp.List[tp.Dict[str, tp.List]]:
        ret: tp.List[tp.Dict[str, tp.List]] = []
        for tokens in token_group:
            val: tp.Dict[str, tp.List] = {}
            for name in attr_names:
                val.setdefault(name, [])
                for token in tokens:
                    if with_punct or not token.is_punctuation:
                        val[name].append(getattr(token, name))
            ret.append(val)
        return ret

    @staticmethod
    def get_attr(
        tokens: tp.List[Token],
        attr_names: tp.List[str],
        with_punct: bool = False,
    ) -> tp.Dict[str, tp.List]:
        token_group = [tokens]
        return TokenUtils._get_attr(token_group, attr_names, with_punct)[0]

    @staticmethod
    def get_attr_by_words(
        tokens: tp.List[Token],
        attr_names: tp.List[str],
        with_punct: bool = False,
    ) -> tp.List[tp.Dict[str, tp.List]]:
        token_group = TokenUtils.group_tokens_by_word(tokens)
        return TokenUtils._get_attr(token_group, attr_names, with_punct)

    @staticmethod
    def get_word_tokens(tokens: tp.List[Token]) -> tp.List[Token]:
        return [t for t in tokens if not t.is_punctuation]


class Syntagma:
    def __init__(self, tokens: tp.List[Token]):
        self.tokens: tp.List[Token] = tokens
        self.text: str = TokenUtils.get_text_from_tokens(self.tokens)
        self.position: Position = Position.internal

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, item) -> Token:
        return self.tokens[item]

    @staticmethod
    def _find_word_tokens(tokens: tp.List[Token]) -> tp.Union[Token, None]:
        for token in tokens:
            if not token.is_punctuation:
                return token
        return None

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_words(self) -> int:
        return len(self.get_words())

    @property
    def num_phonemes(self) -> int:
        return sum([len(word) for word in self.get_phonemes()])

    def get_words(self) -> tp.List[Token]:
        group = TokenUtils.group_tokens_by_word(self.tokens)
        words: tp.List[Token] = []
        for tokens in group:
            word = self._find_word_tokens(tokens)
            if word:
                words.append(word)
        return words

    def get_phonemes(
        self, as_tuple: bool = False
    ) -> tp.Union[tp.List[tp.Tuple[str, ...]], tp.Tuple[str]]:
        words = self.get_words()
        if not as_tuple:
            return [word.phonemes for word in words if word.phonemes]
        else:
            return tuple(itertools.chain.from_iterable(self.get_phonemes()))

    def to_dict(self):
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "position": self.position.name,
        }


class Sentence(DocSent):
    def __init__(self, sent: DocSent = None, tokenize=True):
        self.text_orig: str = ""
        self.position: Position = Position.internal
        self.lang = None

        self._text: str = ""
        self._tokens: tp.List[Token] = []
        self._syntagmas: tp.Optional[tp.List[Syntagma]] = None

        self._add_space = re.compile(f"([{PUNCTUATION}])")
        self._remove_space = re.compile(r'\s([,.?!:;"](?:\s|$))')

        self.ssml_modidfiers = None
        self.ssml_insertions: tp.List = []

        self.warning_messages: tp.List[str] = []
        self.exception_messages: tp.List[str] = []

        self.meta: tp.Dict[str, tp.Any] = {}

        self.parser_version: str = multilingual_text_parser.__version__

        if sent is not None:
            self.text_orig = self._remove_space.sub(r"\1", sent.text)

            super().__init__(*sent.__dict__.values())
            if tokenize:
                self.tokenize()

    def __len__(self) -> int:
        if self.tokens:
            return len(self.tokens)
        else:
            return 0

    def __getitem__(self, item):
        if self.tokens:
            return self.tokens[item]

    @staticmethod
    def _find_word_tokens(
        tokens: tp.List[Token], sil_as_word: bool = True
    ) -> tp.Union[Token, None]:
        for token in tokens:
            if sil_as_word:
                if not token.is_punctuation:
                    return token
            else:
                if not token.is_punctuation and not token.is_pause:
                    return token
        return None

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, tokens: tp.List[Token]):
        if tokens and not isinstance(tokens[0], Token):
            self._tokens = [Token(token) for token in tokens]
        else:
            self._tokens = tokens

    @property
    def syntagmas(self):
        return self._syntagmas

    @syntagmas.setter
    def syntagmas(self, syntagmas: tp.List[Syntagma]):
        num_tokens = sum([len(s) for s in syntagmas])
        assert len(self) == num_tokens
        self._syntagmas = syntagmas

    @property
    def text(self):
        if self.tokens:
            return TokenUtils.get_text_from_tokens(self.tokens)
        else:
            return self._text

    @text.setter
    def text(self, text: str):
        self._text = text

    @property
    def capitalize(self):
        if self.tokens:
            return TokenUtils.get_text_from_tokens(self.tokens, with_capitalize=True)
        else:
            return self._text

    @property
    def stress(self):
        if self.tokens:
            return TokenUtils.get_stress_from_tokens(self.tokens)
        else:
            return self._text

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_words(self) -> int:
        return len(self.get_words())

    @property
    def num_phonemes(self) -> int:
        return sum([len(word) for word in self.get_phonemes()])

    def remove(self, token: Token):
        if self._tokens:
            self._tokens.remove(token)

    def tokenize(self):
        text = self._add_space.sub(r" \1 ", self.text)
        self.tokens = [token for token in text.split(" ") if len(token) > 0]

    def as_norm(self) -> str:
        return " ".join([token.norm for token in self.tokens if len(token.norm) > 0])

    def get_words(self, sil_as_word: bool = True) -> tp.List[Token]:
        group = TokenUtils.group_tokens_by_word(self.tokens, sil_as_word)
        words: tp.List[Token] = []
        for tokens in group:
            word = self._find_word_tokens(tokens, sil_as_word)
            if word:
                words.append(word)
        return words

    def get_words_with_punct(self, sil_as_word: bool = True) -> tp.List[tp.List[Token]]:
        return TokenUtils.group_tokens_by_word(self.tokens, sil_as_word)

    def get_attr(
        self, attr_name: str, with_punct: bool = True, group: bool = False
    ) -> tp.List[tp.Union[Token, tp.List[Token]]]:
        if group:
            attr_by_words = TokenUtils.get_attr_by_words(
                self.tokens, [attr_name], with_punct
            )
            return [item[attr_name] for item in attr_by_words]
        else:
            attr = TokenUtils.get_attr(self.tokens, [attr_name], with_punct)
            return attr[attr_name]  # type: ignore

    def get_phonemes(
        self, as_tuple: bool = False
    ) -> tp.Union[tp.List[tp.Tuple[str, ...]], tp.Tuple[str]]:
        words = self.get_words()
        if not as_tuple:
            return [word.phonemes for word in words if word.phonemes]
        else:
            return tuple(itertools.chain.from_iterable(self.get_phonemes()))

    def get_token_index(self, token: Token) -> int:
        return self.tokens.index(token)

    def get_word_index(self, word: Token) -> int:
        words = self.get_words_with_punct()
        for idx, tokens in enumerate(words):
            if word in tokens:
                return idx
        else:
            raise ValueError(f"word {word} not found!")

    def to_dict(self):
        ret = {"text": self.text, "position": self.position.name}
        if self.syntagmas:
            ret["syntagmas"] = [s.to_dict() for s in self.syntagmas]
        else:
            ret["tokens"] = [t.to_dict() for t in self.tokens]
        return ret


class Doc(NatashaDoc):
    def __init__(
        self,
        text: str,
        sentenize: bool = False,
        tokenize: bool = False,
        add_trailing_punct_token: bool = True,
    ):
        self.text_orig: str = text
        self.lang = None

        self._text: str = text
        self._sents: tp.Optional[tp.List[Sentence]] = None

        self.tags_replacement_map = None
        self.pauses_durations = None
        self.exception_messages: tp.List[str] = []

        self.meta: tp.Dict[str, tp.Any] = {}

        assert len(text) > 0, "text is empty!"

        text = text.strip()
        if text and text[-1] not in ".;!?" and add_trailing_punct_token:
            text += "."

        super().__init__(text)
        if sentenize:
            self.sentenize(tokenize)

    @staticmethod
    def text_from_sentence(sents: tp.List[Sentence]) -> "Text":
        _text = Text("", add_trailing_punct_token=False)
        _text._sents = sents
        _text._text = _text.text
        _text.text_orig = _text._text
        return _text

    def __len__(self) -> int:
        return len(self.tokens)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.text == other
        elif isinstance(other, Text):
            return self.text == other.text
        else:
            return ValueError

    @property
    def tokens(self):
        if self.sents:
            return list(
                itertools.chain.from_iterable([sent.tokens for sent in self.sents])
            )

    @tokens.setter
    def tokens(self, tokens: tp.List[Token]):
        pass

    @property
    def sents(self):
        return self._sents

    @sents.setter
    def sents(self, sents: tp.List[Sentence]):
        self._sents = sents
        if self.sents:
            self.tokens = list(
                itertools.chain.from_iterable(
                    [sent.tokens for sent in self.sents if sent.tokens]
                )
            )

    @property
    def text(self):
        if self.sents is not None:
            return " ".join([sent.text for sent in self.sents])
        else:
            return self._text

    @text.setter
    def text(self, text: str):
        self._text = text

    @property
    def capitalize(self):
        if self.sents is not None:
            return " ".join([sent.capitalize for sent in self.sents])
        else:
            return self._text

    @property
    def stress(self):
        if self.sents is not None:
            return " ".join([sent.stress for sent in self.sents])
        else:
            return self._text

    def sentenize(self, tokenize: bool = True):
        self.sents = [
            Sentence(sent, tokenize)
            for sent in Segmenter().sentenize(self.text)
            if sent.text
        ]

    def to_dict(self):
        return {"text": self.text, "sentence": [s.to_dict() for s in self.sents]}

    @property
    def exceptions(self) -> tp.List[str]:
        all_exceptions = self.exception_messages.copy()
        for sent in self.sents:
            all_exceptions += sent.exception_messages

        return all_exceptions

    def as_norm(self) -> str:
        if self.sents is not None:
            return " ".join(sent.as_norm() for sent in self.sents)
        else:
            return Token.remove_punctuation(self.text)


if __name__ == "__main__":

    utterance = """

    И вот появился на сцене Джеймс Кэмерон с,Титаником,- фильмом, который изменил всю киноиндустрию!
    """

    text = Text(utterance, sentenize=True, tokenize=True)
    print(TokenUtils.get_text_from_tokens(text.tokens))
