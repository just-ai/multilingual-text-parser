from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.processors.ru.morph_analyzer import MorphAnalyzerRU
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir


class NameFinder(BaseSentenceProcessor):
    def __init__(self):
        import dawg

        self._morph = MorphAnalyzerRU()

        # hagen_orf vocab containing no names
        _vocab_path = get_root_dir() / "data/ru/hagen_orf/stress_vocab_NOUN_1.dawg"
        self._vocab = {}
        _, _, _, max_form = _vocab_path.name.rstrip("*.dawg").split("_")
        format = ">" + "H" * int(max_form)
        dawg_record = dawg.RecordDAWG(format)
        try:
            self._vocab = dawg_record.load(_vocab_path.as_posix())
        except OSError:
            with _vocab_path.open("rb") as f:
                self._vocab = dawg_record.read(f)

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        for token in sent.tokens:
            if token.is_punctuation or not token.text_orig:
                token.is_name = False
            else:
                if token.pos is None:
                    pos_tag = self._morph.get_pos(token.text)
                else:
                    pos_tag = token.pos

                if pos_tag in self._morph.cyr_pos_tags:
                    pos_tag = self._morph.lat_pos_tags[
                        self._morph.cyr_pos_tags.index(pos_tag)
                    ]
                if pos_tag is not None:
                    pos_tag = self._morph.oc_to_ud_pos(str(pos_tag))

                if (
                    pos_tag == "NOUN"
                    and token.text_orig[0] == token.text_orig[0].upper()
                    and not token.text_orig[0].isdigit()
                ):
                    token.is_capitalize = True
                    if token.text not in self._vocab:
                        token.is_name = True
                else:
                    token.is_capitalize = False
                    token.is_name = False
