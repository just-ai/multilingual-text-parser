import typing as tp

import pymorphy2

from natasha.morph.vocab import OC_UD_POS
from pymorphy2.tagset import OpencorporaTag

__all__ = ["MorphAnalyzerRU"]


class MorphAnalyzerRU:
    def __init__(self, vocab_only: bool = False):
        self._vocab_only = vocab_only
        self._morph = pymorphy2.MorphAnalyzer(lang="ru")

        self.lat_pos_tags = list(OpencorporaTag.PARTS_OF_SPEECH)
        self.cyr_pos_tags = self._morph.lat2cyr(" ".join(self.lat_pos_tags))
        self.cyr_pos_tags = self.cyr_pos_tags.split(" ")
        self.ud_pos_tags = [self.oc_to_ud_pos(x) for x in self.lat_pos_tags]

        self.lat_case_tags = list(OpencorporaTag.CASES)
        self.cyr_case_tags = self._morph.lat2cyr(" ".join(self.lat_case_tags))
        self.cyr_case_tags = self.cyr_case_tags.split(" ")

    def oc_to_ud_pos(self, pos_tag: str):
        if pos_tag in OC_UD_POS.values():
            return pos_tag
        pos_tag = self._morph.cyr2lat(pos_tag)
        if pos_tag == "PRED":
            pos_tag = "ADVB"
        if pos_tag == "SCONJ":
            pos_tag = "CONJ"
        if pos_tag == "AUX":
            pos_tag = "VERB"
        if pos_tag == "X":
            pos_tag = "NOUN"
        if pos_tag == "SYM":
            pos_tag = "NOUN"
        return OC_UD_POS[pos_tag]

    def is_known(self, word: str):
        return self._morph.word_is_known(word)

    def parse(self, word: str):
        if self._vocab_only and not self._morph.word_is_known(word):
            raise ValueError(f"{word} not in dictionary!")
        return self._morph.parse(word)[0]

    def get_lexeme(
        self,
        word: tp.Union[str, pymorphy2.analyzer.Parse],
        gramm_filter: tp.Optional[list] = None,
        with_meta: bool = True,
    ):
        word = self.parse(word) if isinstance(word, str) else word
        lexeme = self._morph.get_lexeme(word)

        if gramm_filter:
            gramm_filter = [self._morph.cyr2lat(x) for x in gramm_filter if x]
            lexeme = [
                item for item in word.lexeme if all(x in item.tag for x in gramm_filter)
            ]

        if not with_meta:
            lexeme = [item.word for item in lexeme]

        return lexeme

    def get_tag(self, word: tp.Union[str, pymorphy2.analyzer.Parse]):
        word = self.parse(word) if isinstance(word, str) else word
        return word.tag

    def get_gramm(self, word: tp.Union[str, pymorphy2.analyzer.Parse]):
        word = self.parse(word) if isinstance(word, str) else word
        tag = self.get_tag(word)
        out = {}
        if tag.case:
            out["case"] = self._morph.lat2cyr(tag.case)
        if tag.number:
            out["number"] = self._morph.lat2cyr(tag.number)
        if tag.gender:
            out["gender"] = self._morph.lat2cyr(tag.gender)
        return out

    def get_pos(self, word: tp.Union[str, pymorphy2.analyzer.Parse]):
        tag = self.get_tag(word)
        if tag.POS:
            return self._morph.lat2cyr(tag.POS)

    def get_case(self, word: tp.Union[str, pymorphy2.analyzer.Parse]):
        tag = self.get_tag(word)
        if tag.case:
            return self._morph.lat2cyr(tag.case)

    def get_number(self, word: tp.Union[str, pymorphy2.analyzer.Parse]):
        tag = self.get_tag(word)
        if tag.number:
            return self._morph.lat2cyr(tag.number)

    def get_gender(self, word: tp.Union[str, pymorphy2.analyzer.Parse]):
        tag = self.get_tag(word)
        if tag.gender:
            return self._morph.lat2cyr(tag.gender)

    def inflect(
        self, word: tp.Union[str, pymorphy2.analyzer.Parse], required_grammemes: list
    ):
        word = self.parse(word) if isinstance(word, str) else word
        lexeme = self.find_lexeme(word.normalized, gramm_filter=required_grammemes)
        if lexeme:
            return lexeme.word

    def find_lexeme(
        self,
        word: tp.Union[str, pymorphy2.analyzer.Parse],
        ending: tp.Optional[str] = None,
        gramm_filter: tp.Optional[list] = None,
    ):
        word = self.parse(word) if isinstance(word, str) else word
        lexeme = word.lexeme

        if ending:
            lexeme = [x for x in lexeme if x.word.endswith(ending)]

        if gramm_filter:
            gramm_filter = [self._morph.cyr2lat(x) for x in gramm_filter if x]
            for gramm in gramm_filter:
                temp = [x for x in lexeme if gramm in x.tag]
                if temp:
                    lexeme = temp

        if lexeme:
            return lexeme[0]
