import json
import typing as tp

from pathlib import Path

from stressrnn import StressRNN

from multilingual_text_parser.data_types import Doc, Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.processors.ru.morph_analyzer import MorphAnalyzerRU
from multilingual_text_parser.processors.ru.pos_tagger import PosTaggerRU
from multilingual_text_parser.processors.ru.rulebased_normalizer import Utils
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir
from multilingual_text_parser.utils.profiler import Profiler

__all__ = ["AccentorRU"]


def _create_homographs_vocab(stress_vocab):
    merged_vocab = {}
    for pos_tag, pos_vocab in stress_vocab.items():
        for word, current_stress in pos_vocab.items():
            word_stress = merged_vocab.setdefault(word, set())
            word_stress.update(current_stress)
    return merged_vocab


class AccentorRU(BaseSentenceProcessor):
    def __init__(
        self,
        vocab_only: bool = False,
        vocab_path: tp.Optional[Path] = None,
        with_homographs: bool = False,
        skip_obvious: bool = False,
    ):
        import dawg

        self._morph = MorphAnalyzerRU()
        self._stress_rnn = StressRNN()
        self._vocab_only = vocab_only
        self._with_homographs = with_homographs
        self._skip_obvious = skip_obvious

        vocab_path = vocab_path or "data/ru/accentor"
        dict_root = get_root_dir() / vocab_path
        dict_path = list(dict_root.glob("*.dawg"))

        self._stress_vocab = {}
        for path in dict_path:
            _, _, pos_tag, max_form = path.name.rstrip("*.dawg").split("_")
            format = ">" + "H" * int(max_form)
            dawg_record = dawg.RecordDAWG(format)
            try:
                self._stress_vocab[pos_tag] = dawg_record.load(path.as_posix())
            except OSError:
                with path.open("rb") as f:
                    self._stress_vocab[pos_tag] = dawg_record.read(f)

        self._label_auto_stress = False
        if self._with_homographs:
            self._label_auto_stress = True
            self._stress_vocab = _create_homographs_vocab(self._stress_vocab)
            print(
                f"Homographs count {len({k: v for k, v in self._stress_vocab.items() if len(v) > 2})}"
            )

        vocabs_dir = get_root_dir() / "data/ru/vocabularies"
        vocab_path = vocabs_dir / "custom_stress.txt"
        vocab: tp.Dict[str, str] = Utils.read_vocab(vocab_path)  # type: ignore
        self._custom_stress_vocab: tp.Dict[str, tp.Tuple[int, ...]] = {}
        for word, word_stress in vocab.items():
            stress_pos = word_stress.find("+")
            if stress_pos >= 0:
                self._custom_stress_vocab[word] = (stress_pos,)

        with open(get_root_dir() / "data/ru/vocabularies/feats_dict.json") as json_file:
            self.feat_dict = json.load(json_file)

    def _stress_selection(
        self,
        word: str,
        stress_pos: tp.Tuple[int],
        pos_tag: str,
        grammemes: tp.Dict[str, str],
        vocab_pos_tag: str,
        question_mark: bool,
        all_pos: bool,
    ) -> tp.Tuple[int, ...]:
        stress_pos = tuple(x for x in stress_pos if x > 0)

        if grammemes is None:
            grammemes = {}

        if (
            not all_pos
            and vocab_pos_tag in self.feat_dict
            and word in self.feat_dict[vocab_pos_tag]
        ):
            f1 = {grammemes[g] for g in grammemes}
            m_inter = 0
            s = 0
            all_forms = self.feat_dict[vocab_pos_tag][word]
            for f in all_forms:
                f_set = set(f.split("|"))
                if f_set == f1:
                    s = all_forms[f]
                    return (s,)
                elif len(f_set.intersection(f1)) > m_inter:
                    m_inter = len(f_set.intersection(f1))
                    s = all_forms[f]
            return (s,)

        if all_pos or len(stress_pos) == 1:
            return stress_pos

        if vocab_pos_tag == "NOUN":
            if "-" in word and len(stress_pos) == 2:
                return stress_pos
            else:
                if grammemes.get("Number", "Sing") == "Sing":
                    return (min(stress_pos),)
                else:
                    return (max(stress_pos),)

        if vocab_pos_tag == "VERB" and pos_tag in ["NOUN", "VERB", "ADJ"]:
            if question_mark:
                return (min(stress_pos),)
            if pos_tag == "ADJ":
                return (min(stress_pos),)

        return (max(stress_pos),)

    def get_stress_from_vocab(
        self,
        word: str,
        pos_tag: str = None,  # type: ignore
        grammemes: tp.Dict[str, str] = None,  # type: ignore
        question_mark: bool = False,
        all_pos: bool = False,
    ) -> tp.Optional[tp.Tuple[int, ...]]:
        if len(word) == 1:
            return None

        word = word.lower()

        if word in self._custom_stress_vocab:
            return self._custom_stress_vocab[word]

        if pos_tag is None:
            pos_tag = self._morph.get_pos(word)

        if pos_tag in self._morph.cyr_pos_tags:
            pos_tag = self._morph.lat_pos_tags[self._morph.cyr_pos_tags.index(pos_tag)]

        pos_tag = self._morph.oc_to_ud_pos(str(pos_tag))

        if pos_tag == "DET":
            pos_tag = "PRON"

        if self._with_homographs:
            _vocab = self._stress_vocab
        elif pos_tag and pos_tag in self._stress_vocab:
            _vocab = self._stress_vocab[pos_tag]
        else:
            _vocab = None  # type: ignore

        if _vocab:
            if word in _vocab:
                stress_pos = _vocab[word]
                all_stress_pos = []
                for pos in stress_pos:
                    # filter padding artefacts from dawg saving
                    if pos == 0:
                        continue
                    pos = (pos,) if isinstance(pos, int) else pos
                    all_stress_pos.append(
                        self._stress_selection(
                            word,
                            pos,
                            pos_tag,
                            grammemes,
                            pos_tag,
                            question_mark,
                            all_pos,
                        )[0]
                    )
                return tuple(all_stress_pos)
            else:
                for vocab_pos, stress_vocab in self._stress_vocab.items():
                    if vocab_pos != pos_tag and word in stress_vocab:
                        stress_pos = stress_vocab[word][0]
                        return self._stress_selection(
                            word,
                            stress_pos,
                            pos_tag,
                            grammemes,
                            vocab_pos,
                            question_mark,
                            all_pos,
                        )

        if self._vocab_only:
            raise RuntimeError(f"word {word} not in dictionary!")

        return None

    def get_stress_from_rnn(
        self,
        word: str,
    ):
        if len(word) == 1:
            return

        for threshold in [0.75, 0.25]:
            try:
                stressed_text = self._stress_rnn.put_stress(
                    word,
                    stress_symbol="+",
                    accuracy_threshold=threshold,
                    replace_similar_symbols=True,
                )
                if "+" in stressed_text:
                    stress_poss: tp.List = []
                    for i, letter in enumerate(stressed_text):
                        if letter == "+":
                            stress_poss.append(i - len(stress_poss))
                    return tuple(stress_poss)
            except Exception:
                return []

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        question_mark = "?" in sent.text
        for token in sent.tokens:
            auto_stress = False
            if not token.is_punctuation and not token.is_number:
                if token.stress and "+" in token.stress:
                    continue

                stress_poss = self.get_stress_from_vocab(
                    word=token.text,
                    pos_tag=token.pos,
                    grammemes=token.feats,
                    question_mark=question_mark,
                )
                if stress_poss is None:
                    stress_poss = self.get_stress_from_rnn(
                        word=token.text,
                    )
                    if self._label_auto_stress and stress_poss:
                        auto_stress = True

                if stress_poss:
                    if not self._with_homographs:
                        token_stress = token.text
                        increment = 0
                        for pos in stress_poss:
                            pos += increment
                            token_stress = token_stress[:pos] + "+" + token_stress[pos:]
                            increment += 1
                        token.stress = token_stress
                    else:
                        all_token_stress = []

                        for pos in stress_poss:
                            token_text = (
                                token.text_orig if token.text_orig else token.text
                            )
                            if (
                                not self._skip_obvious
                                or token.is_name
                                or (
                                    self._skip_obvious
                                    and (len(stress_poss) > 1 or auto_stress)
                                )
                            ):
                                token_stress = token_text[:pos] + "+" + token_text[pos:]
                            else:
                                token_stress = token_text

                            all_token_stress.append(token_stress)
                            if auto_stress:
                                all_token_stress.append("{auto stress}")
                            token.stress = all_token_stress


if __name__ == "__main__":
    pos_tagger = PosTaggerRU()
    acc = AccentorRU()

    doc = Doc(
        """
        Плясали свои и приглашённые гости.
        """,
        sentenize=True,
        tokenize=True,
    )

    with Profiler(format=Profiler.Format.ms):
        doc = pos_tagger(doc)
        doc = acc(doc)

    for sent in doc.sents:
        print(sent.stress)
