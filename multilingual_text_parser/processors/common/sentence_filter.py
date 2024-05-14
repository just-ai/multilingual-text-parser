from multilingual_text_parser._constants import PUNCTUATION_ALL, PUNCTUATION_EXTEND
from multilingual_text_parser.data_types import Doc, Syntagma
from multilingual_text_parser.processors.base import BaseTextProcessor

__all__ = ["SentenceFilter"]


class SentenceFilter(BaseTextProcessor):
    def _process_text(self, doc: Doc, **kwargs):
        disable_phonemizer = kwargs.get("disable_phonemizer", False)

        def remove_no_valid_tokens(tokens):
            if disable_phonemizer:
                return

            for t in list(tokens):
                if t.num_phonemes:
                    t.phonemes = tuple(ph for ph in t.phonemes if ph not in ["", None])
                if not (t.is_punctuation or t.num_phonemes):
                    tokens.remove(t)

        def remove_first_no_word_tokens(tokens):
            if disable_phonemizer:
                return

            for t in list(tokens):
                if not t.is_word:
                    tokens.remove(t)
                else:
                    break

        invalid_symbols = set(PUNCTUATION_ALL) - set(PUNCTUATION_EXTEND)

        new_sents = []
        for sent in doc.sents:
            if sent.syntagmas is None:
                syntagmas = [[]]
                for token in sent.tokens:
                    syntagmas[-1].append(token)
                    if token.is_punctuation and token != sent.tokens[-1]:
                        syntagmas.append([])

                sent.syntagmas = [Syntagma(item) for item in syntagmas]

            for t in sent.tokens:
                if not t.is_word:
                    t.text = t.text.translate(
                        str.maketrans("", "", "".join(invalid_symbols))
                    )

            num_tokens = len(sent.tokens)
            remove_no_valid_tokens(sent.tokens)
            remove_first_no_word_tokens(sent.tokens)

            if num_tokens != len(sent.tokens):
                for synt in sent.syntagmas:
                    remove_no_valid_tokens(synt.tokens)
                    if synt == sent.syntagmas[0]:
                        remove_first_no_word_tokens(synt.tokens)

            new_syntagmas = []
            for synt in sent.syntagmas:
                if len(synt.tokens) > 0:
                    if len(synt.tokens) == 1 and synt.tokens[0].is_punctuation:
                        sent.tokens.remove(synt.tokens[0])
                        continue
                    new_syntagmas.append(synt)

            sent.syntagmas = new_syntagmas

            word_tokens = [t for t in sent.tokens if t.is_word]
            ph_by_word = sent.get_phonemes()

            if word_tokens and (ph_by_word or disable_phonemizer):
                new_sents.append(sent)
            else:
                doc.exception_messages += sent.exception_messages

        doc.sents = new_sents
