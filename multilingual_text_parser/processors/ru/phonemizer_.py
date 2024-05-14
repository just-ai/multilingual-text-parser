import re
import typing as tp
import itertools

from multilingual_text_parser._constants import PUNCTUATION_LEFT
from multilingual_text_parser.data_types import Doc, Position, Sentence, Syntagma
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.processors.ru.normalizer_utils import Utils
from multilingual_text_parser.processors.ru.transcriptor import TranscriptorRU
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir

__all__ = ["PhonemizerRU"]


class PhonemizerRU(BaseSentenceProcessor):
    def __init__(self):
        self._transcriptor = TranscriptorRU()

        vocab_root = get_root_dir() / "data/ru/vocabularies"
        self._phonetic_vocab = Utils.read_vocab(vocab_root / "phonetic.txt")
        self._phonetic_vocab = {
            k: tuple(v.split("|")) for k, v in self._phonetic_vocab.items()
        }

    def _find_next_word(
        self, num_phonemes, next_word, ph_by_phrase, is_trim: bool = False
    ) -> int:
        if len(next_word) == 1:
            if ph_by_phrase[num_phonemes - 1] == next_word[0]:
                return num_phonemes - 1
            else:
                return num_phonemes

        next_word = list(next_word)
        for shift in range(-2, 2):
            r = num_phonemes + shift
            a = max(0, r)
            b = min(r + len(next_word), len(ph_by_phrase))
            part_phrase = list(ph_by_phrase[a:b])
            if is_trim:
                if len(part_phrase[0]) == 1:
                    continue
                if next_word[-1][-1].isdigit():
                    next_word[-1] = next_word[-1][:-1]
                if part_phrase[-1][-1].isdigit():
                    part_phrase[-1] = part_phrase[-1][:-1]
                if part_phrase[0].endswith(next_word[0]) or next_word[0].endswith(
                    part_phrase[0]
                ):
                    part_phrase[0] = next_word[0]
                if part_phrase[-1].startswith(next_word[-1]) or next_word[-1].startswith(
                    part_phrase[-1]
                ):
                    part_phrase[-1] = next_word[-1]
            if part_phrase == next_word:
                num_phonemes += shift
                break
        else:
            if not is_trim:
                return self._find_next_word(num_phonemes, next_word, ph_by_phrase, True)
            else:
                return num_phonemes - 1
        return num_phonemes

    def _get_phonemes(
        self, phrase, num_phonemes, tokens, token_idx
    ) -> tp.Tuple[tp.List[str], tp.List[str]]:
        ph_word = phrase[:num_phonemes]
        ph_phrase = phrase[num_phonemes:]

        if num_phonemes == 0:
            idx = token_idx - 2
            while idx >= 0:
                prev_token = tokens[idx]
                if not prev_token.is_punctuation:
                    if len(prev_token.phonemes) > 1:
                        ph_word = tuple([prev_token.phonemes[-1]])
                        prev_token.phonemes = prev_token.phonemes[:-1]
                    else:
                        ph_word = self._transcriptor.transcribe_word(
                            tokens[token_idx - 1]
                        )
                    break
                idx -= 1
            else:
                ph_word = phrase[: num_phonemes + 1]
                ph_phrase = phrase[num_phonemes + 1 :]

        return ph_word, ph_phrase

    def __call__(self, doc: Doc, **kwargs) -> Doc:
        for sent_idx, sent in enumerate(doc.sents):
            if sent_idx == 0:
                sent.position = Position.first
            elif sent_idx + 1 == len(doc.sents):
                sent.position = Position.last

            self._process_sentence(sent, **kwargs)
        return doc

    @exception_handler
    def _process_sentence(self, sent: Sentence, is_debug: bool = False, **kwargs):
        if kwargs.get("disable_phonemizer", False):
            return

        ret = self._transcriptor.transcribe(sent)
        if not ret:
            sent.syntagmas = [Syntagma(sent.tokens)]
            return

        syntagmas = []
        token_idx = 0
        token_idx_last = 0
        for phrase_idx, (accent_words, ph_by_word, ph_by_phrase) in enumerate(ret):
            if len(accent_words) == 0:
                continue
            acc_word_idx = 0
            phoneme_idx = 0
            while token_idx < sent.num_tokens:
                token = sent.tokens[token_idx]
                token_idx += 1
                if token.is_punctuation:
                    continue
                acc_word = accent_words[acc_word_idx]
                acc_word = re.sub(r"^[ьъ]{1}", "", acc_word)
                acc_word = re.sub(r"^[ы]{1}", "и", acc_word)
                if token.text.startswith("что"):
                    acc_word = re.sub(r"^(што)", "что", acc_word)
                word_phonemes = ph_by_word[phoneme_idx]
                if not (
                    token.pos == "ADP"
                    and token.norm != acc_word.replace("+", "").replace("ё", "е")
                ):
                    num_phonemes = len(word_phonemes)
                    if (
                        num_phonemes > 1
                        and word_phonemes != ph_by_phrase[:num_phonemes]
                        and phoneme_idx + 1 < len(ph_by_word)
                    ):
                        next_word = ph_by_word[phoneme_idx + 1]
                        num_phonemes = self._find_next_word(
                            num_phonemes, next_word, ph_by_phrase
                        )
                    elif (
                        num_phonemes == 1 and word_phonemes != ph_by_phrase[:num_phonemes]
                    ):
                        num_phonemes = 0

                    token.phonemes, ph_by_phrase = self._get_phonemes(
                        ph_by_phrase, num_phonemes, sent.tokens, token_idx
                    )

                    token.stress = acc_word
                    acc_word_idx += 1
                    phoneme_idx += 1
                    if acc_word_idx == len(accent_words):
                        token.phonemes += ph_by_phrase
                        while (
                            token_idx < len(sent.tokens)
                            and sent.tokens[token_idx].is_punctuation
                        ):
                            if any(
                                s in sent.tokens[token_idx].text for s in PUNCTUATION_LEFT
                            ):
                                if token_idx + 1 == len(sent.tokens):
                                    token_idx += 1
                                break
                            else:
                                token_idx += 1
                        if is_debug:
                            assert len(ph_by_phrase) <= 1
                        break
                else:
                    acc_word = acc_word.replace("+", "")[: len(token.norm)]
                    if is_debug:
                        assert acc_word == token.norm
                    word_phonemes = ph_by_word[phoneme_idx][: len(acc_word)]
                    num_phonemes = len(word_phonemes)

                    if word_phonemes != ph_by_phrase[:num_phonemes]:
                        next_word = ph_by_word[phoneme_idx][len(acc_word) :]
                        num_phonemes = self._find_next_word(
                            num_phonemes, next_word, ph_by_phrase
                        )

                    token.phonemes, ph_by_phrase = self._get_phonemes(
                        ph_by_phrase, num_phonemes, sent.tokens, token_idx
                    )

                    token.stress = acc_word[: len(token.norm)]
                    accent_words[acc_word_idx] = accent_words[acc_word_idx][
                        len(token.norm) :
                    ]
                    ph_by_word[phoneme_idx] = ph_by_word[phoneme_idx][len(token.norm) :]

            syntagma = Syntagma(sent.tokens[token_idx_last:token_idx])
            for token in syntagma.tokens:
                if (
                    token.modifiers
                    and "phoneme" in token.modifiers
                    and "ph" in token.modifiers["phoneme"]
                ):
                    token.phonemes = tuple(token.modifiers["phoneme"]["ph"].split("|"))
                elif (
                    isinstance(token.stress, str) and token.stress in self._phonetic_vocab
                ):
                    token.phonemes = self._phonetic_vocab[token.stress]

            if phrase_idx == 0:
                syntagma.position = Position.first
            elif phrase_idx + 1 == len(ret):
                syntagma.position = Position.last
            if len(syntagma.tokens) > 0:
                syntagmas.append(syntagma)
            token_idx_last = token_idx

        sent.syntagmas = syntagmas

        words = sent.get_words()
        assert all([word.phonemes for word in words]), f"phoneme mistake: {sent.text}"

        if is_debug:
            ph_by_phrase = [item[2] for item in ret]
            ph_by_tokens = [item.phonemes for item in sent.tokens if item.phonemes]
            ph_by_phrase = list(itertools.chain.from_iterable(ph_by_phrase))
            ph_by_tokens = list(itertools.chain.from_iterable(ph_by_tokens))
            assert ph_by_tokens == ph_by_phrase


if __name__ == "__main__":
    import json

    from pos_tagger import PosTaggerRU

    pos_tagger = PosTaggerRU()
    phonemizer = PhonemizerRU()

    doc = Doc(
        "Кто - то пытается навязать тебе своё мнение и зомбировать мозг!",
        sentenize=True,
    )
    doc = pos_tagger(doc)
    doc = phonemizer(doc)

    print(json.dumps(doc.to_dict(), ensure_ascii=False, indent=4))
