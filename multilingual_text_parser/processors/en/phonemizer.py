from typing import List

from multilingual_text_parser.data_types import Sentence, Token
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.thirdparty.en.g2p.g2p_en import G2p
from multilingual_text_parser.utils.decorators import exception_handler


class PhonemizerEN(BaseSentenceProcessor):
    def __init__(self):
        self.g2p = G2p()

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        if kwargs.get("disable_phonemizer", False):
            return

        phonemes, tokens = self.g2p(sent.stress)
        i = 0
        j = 0
        new_tokens: List[Token] = []
        while i < len(sent.tokens) and j < len(tokens):
            token = sent.tokens[i]
            if token.text.lower() == tokens[j][0]:
                token.pos = tokens[j][1]
                if not token.phonemes:
                    token.phonemes = tuple(phonemes[j])
                i += 1
                j += 1
            else:
                token.pos = "X"
                token.phonemes = ()
                i += 1

            if token.num_phonemes > 0:
                new_tokens.append(token)
            else:
                if len(new_tokens) == 0 and token.is_punctuation:
                    new_tokens.append(token)
                else:
                    if (
                        len(new_tokens) > 0
                        and new_tokens[-1].is_punctuation
                        and token.is_punctuation
                    ):
                        new_tokens[-1] = token

            if token.is_punctuation:
                token.pos = "PUNCT"
                token.phonemes = None

        sent.tokens = new_tokens


if __name__ == "__main__":
    from multilingual_text_parser.data_types import Text
    from multilingual_text_parser.parser import TextParser
    from multilingual_text_parser.utils.profiler import Profiler

    parser = TextParser(lang="EN", device="cpu", with_profiler=True)

    text = Text(
        """
    He+llo, world!
    """
    )

    with Profiler():
        text = parser.process(text)

    print(text.sents[0].get_attr("phonemes"))
