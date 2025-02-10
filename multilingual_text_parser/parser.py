import os
import sys
import typing as tp
import logging
import itertools

from copy import deepcopy

from multilingual_text_parser import processors
from multilingual_text_parser._constants import (
    EN2IPA,
    INTONATION_TYPES,
    LOCALE_CODES_MAP,
    PHONEMES_ENGLISH,
    PHONEMES_IPA,
    PUNCTUATION,
    REL_NATASHA,
    REL_STANZA,
    RU2IPA,
    UNIVERSAL_POS,
)
from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.thirdparty.ru.russian_g2p.Grapheme2Phoneme import (
    Grapheme2Phoneme,
)
from multilingual_text_parser.utils import lang_supported as stanza_utils
from multilingual_text_parser.utils.init import init_class_from_config
from multilingual_text_parser.utils.lang_supported import espeak_available_languages
from multilingual_text_parser.utils.profiler import Profiler

__all__ = ["TextParser", "EmptyTextError"]

LOGGER = logging.getLogger("root")


if sys.platform == "win32":
    os.environ[
        "PHONEMIZER_ESPEAK_LIBRARY"
    ] = "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"


class EmptyTextError(Exception):
    pass


class TextParser:
    languages = "RU", "EN", "PT-BR", "KK", "KY", "MULTILANG"
    languages_espeak = espeak_available_languages()
    languages_stanza = stanza_utils.stanza_available_languages()

    pipe_multilingual = [
        "SymbolsModifier",
        "TextModifier",
        "Corrector",
        "Sentenizer",
        "SentencesModifier",
        "SSMLCollector",
        "Tokenizer",
        "SSMLApplier",
        "Normalizer",
        "Phonemizer",
        "BatchSyntaxAnalyzer",
        "SentenceFilter",
    ]

    pipe_ru = [
        "SymbolsModifier",
        "TextModifier",
        "TextModifierRU",
        "Corrector",
        "SentenizerRU",
        "SentencesModifierRU",
        "SSMLCollector",
        "Tokenizer",
        "SSMLApplier",
        "PosTaggerRU",
        "OriginalTextRestorer",
        "NameFinder",
        "SyntaxAnalyzerRU",
        "TaggerRU",
        "RuleBasedNormalizerRU",
        "HomographerRU",
        "AccentorRU",
        "PhonemizerRU",
        "SentenceFilter",
    ]

    pipe_en = [
        "SymbolsModifier",
        "TextModifier",
        "Corrector",
        "Sentenizer",
        "SentencesModifier",
        "SSMLCollector",
        "Tokenizer",
        "SSMLApplier",
        "NormalizerEN",
        "HomographerEN",
        "PhonemizerEN",
        "BatchSyntaxAnalyzer",
        "SentenceFilter",
    ]

    pipe_pt_br = [
        "SymbolsModifier",
        "TextModifier",
        "Corrector",
        "Sentenizer",
        "SentencesModifier",
        "SSMLCollector",
        "Tokenizer",
        "SSMLApplier",
        "NormalizerPTBR",
        "Phonemizer",
        "BatchSyntaxAnalyzer",
        "SentenceFilter",
    ]

    pipe_kk = [
        "SymbolsModifier",
        "TextModifier",
        "Corrector",
        "Sentenizer",
        "SentencesModifier",
        "SSMLCollector",
        "Tokenizer",
        "SSMLApplier",
        "NormalizerKK",
        "Phonemizer",
        "BatchSyntaxAnalyzer",
        "SentenceFilter",
    ]

    pipe_ky = [
        "SymbolsModifier",
        "TextModifier",
        "Corrector",
        "Sentenizer",
        "SentencesModifier",
        "SSMLCollector",
        "Tokenizer",
        "SSMLApplier",
        "NormalizerKY",
        "Phonemizer",
        "BatchSyntaxAnalyzer",
        "SentenceFilter",
    ]

    en2ipa = EN2IPA
    ru2ipa = RU2IPA

    def __init__(
        self,
        lang: str,
        device: str = "cpu",
        with_profiler: bool = False,
        cfg: tp.Optional[dict] = None,
    ):
        lang = TextParser.locale_to_language(lang)
        if not self.check_language_support(lang):
            raise ValueError(f"{lang} is not supported")

        self._cfg = cfg if cfg else {}
        self._lang = lang
        self._device = device
        self._with_profiler = with_profiler
        self._apply_text_restore = False

        if lang == "RU":
            self.pipe = self._cfg.get("pipe", self.pipe_ru)
        elif lang == "EN":
            self.pipe = self._cfg.get("pipe", self.pipe_en)
        elif lang == "PT-BR":
            self.pipe = self._cfg.get("pipe", self.pipe_pt_br)
        elif lang == "KK":
            self.pipe = self._cfg.get("pipe", self.pipe_kk)
        elif lang == "KY":
            self.pipe = self._cfg.get("pipe", self.pipe_ky)
        else:
            self.pipe = self._cfg.get("pipe", self.pipe_multilingual)

        self.components = {}
        for i, step_name in enumerate(self.pipe):
            handler_cls = getattr(processors, step_name)
            cfg = self._cfg.get(step_name, {})

            is_gpu_capable = getattr(handler_cls, "GPU_CAPABLE", False)
            if is_gpu_capable:
                cfg["device"] = device  # type: ignore

            is_multilang = getattr(handler_cls, "MULTILANG", False)
            if is_multilang:
                cfg["lang"] = lang  # type: ignore

            handler = init_class_from_config(handler_cls, cfg)()  # type: ignore
            self.components[f"{i}_{step_name}"] = handler

            if "TextModifier" in step_name:
                self._apply_text_restore = True

    def __call__(self, utterance: str) -> str:
        return self.process(
            Doc(utterance), disable_translit=True, disable_phonemizer=True
        ).text_with_capitalize

    def process(self, doc: Doc, **kwargs) -> Doc:
        doc = deepcopy(doc)
        kwargs["lang"] = self._lang

        for name, handler in self.components.items():
            with Profiler(
                name=name, format=Profiler.format.ms, enable=self._with_profiler  # type: ignore
            ):
                doc = handler(doc, **kwargs)
                if self._apply_text_restore and isinstance(handler, processors.Tokenizer):
                    doc = self.components["1_TextModifier"].restore(doc)
                if (
                    isinstance(handler, processors.SentenizerRU)
                    or isinstance(handler, processors.Sentenizer)
                    or isinstance(handler, processors.SSMLCollector)
                ):
                    if not doc.sents:
                        raise EmptyTextError

        if doc.sents:
            all_exceptions = list(
                itertools.chain(*[sent.exception_messages for sent in doc.sents])
            )
            last_processor_name = self.pipe[-1]
            for msg in all_exceptions:
                if last_processor_name in msg:
                    raise RuntimeError(f"Exception {msg} in the last processor.")
        else:
            raise EmptyTextError("|".join(doc.exceptions))

        for sent in doc.sents:
            sent.lang = self._lang

        doc.lang = self.lang
        return doc

    @property
    def device(self) -> str:
        return self._device

    @property
    def lang(self) -> str:
        return self._lang

    def _phonemes(self, version: tp.Optional[str] = None) -> tp.Tuple[str, ...]:
        if version is None:
            phonemes_russian = Grapheme2Phoneme().russian_phonemes
            phonemes_english = PHONEMES_ENGLISH
            phonemes_ipa = PHONEMES_IPA
        else:
            raise NotImplementedError

        if self._lang == "RU":
            return self._sort(phonemes_russian)
        elif self._lang == "EN":
            return self._sort(phonemes_english)
        elif self._lang == "MULTILANG":
            return (
                self._sort(phonemes_russian)
                + self._sort(phonemes_english)
                + self._sort(phonemes_ipa)
            )
        else:
            return self._sort(PHONEMES_IPA)

    @property
    def phonemes(self) -> tp.Tuple[str, ...]:
        return self._phonemes()

    @property
    def ipa_phonemes(self) -> tp.Tuple[str, ...]:
        return self._sort(PHONEMES_IPA)

    @property
    def num_symbols_per_phoneme(self) -> int:
        return 3 if self.is_ipa_phonemes else 1

    @property
    def is_ipa_phonemes(self) -> bool:
        if self._lang in ["RU", "EN"]:
            return False
        else:
            return True

    def _punctuation(self, version: tp.Optional[str] = None) -> tp.Tuple[str, ...]:
        if version is None:
            return self._sort(list(PUNCTUATION))
        else:
            raise NotImplementedError

    @property
    def punctuation(self):
        return self._punctuation()

    def _pos(self, version: tp.Optional[str] = None) -> tp.Tuple[str, ...]:
        if version is None:
            return self._sort(UNIVERSAL_POS)  # https://universaldependencies.org/u/pos
        else:
            raise NotImplementedError

    @property
    def pos(self):
        return self._pos()

    def _rel(self, version: tp.Optional[str] = None) -> tp.Tuple[str, ...]:
        if version is None:
            rel_natasha = REL_NATASHA
            rel_stanza = REL_STANZA
        else:
            raise NotImplementedError

        if self._lang == "RU":
            return self._sort(rel_natasha)
        elif self._lang == "MULTILANG":
            return self._sort(rel_natasha) + self._sort(rel_stanza)
        else:
            return self._sort(rel_stanza)

    @property
    def rel(self):
        return self._rel()

    def _intonation(self, version: tp.Optional[str] = None) -> tp.Tuple[str, ...]:
        if version is None:
            return self._sort(list(INTONATION_TYPES.values()))
        else:
            raise NotImplementedError

    @property
    def intonation(self):
        return self._intonation()

    @staticmethod
    def _sort(seq: tp.Union[tp.Tuple[str, ...], tp.List[str]]) -> tp.Tuple[str, ...]:
        return tuple(sorted(set(seq)))

    @staticmethod
    def locale_to_language(locale: str) -> str:
        lang = locale
        if locale.replace("-", "_") in LOCALE_CODES_MAP:
            lang = LOCALE_CODES_MAP[locale.replace("-", "_")]
        return lang

    @staticmethod
    def language_to_locale(lang: str) -> str:
        locale = lang
        languages_to_locate = {v: k for k, v in LOCALE_CODES_MAP.items()}
        if lang in languages_to_locate:
            locale = languages_to_locate[lang]
        return locale

    @staticmethod
    def check_language_support(lang: str) -> bool:
        lang = TextParser.locale_to_language(lang)
        return lang in TextParser.languages or lang.lower() in TextParser.languages_espeak


if __name__ == "__main__":
    _lang = "KK"
    _device = "cpu"

    print("Support languages:", TextParser.languages)
    print("eSpeak languages:", TextParser.languages_espeak)
    print("Stanza languages:", TextParser.languages_stanza)

    utterance_ru = """
        Фото на стр. 5 ярко иллюстрирует упомянутый выше феномен, который в 1889 году в ходе экспериментов наблюдал Эрнст Мах.
        На рис.3 изображена Валерия с тремя золотыми и одной серебряной медалью, которые она завоевала в Токио, прославив страну и родной регион.

        Объём продаж одноразовых масок в России снизился на 19% за 1,5 месяца.
        за 1,5 дня.
        за 1,5 века.

        PR-менеджер – это специалист по связям с общественностью, который отвечает за создание и поддержание благоприятного имиджа компании.
    """

    utterance_en = "!!!! Hello+. I have $5. <p> How are </p> you? I have $5.... .\n\n. http://www.ivona.com"

    utterance_es = "300: 29–30 así, la selección femenina podría promover la salud general de las poblaciones en esta especie."

    utterance_de = "Mein Name ist Anna. Ich komme aus Österreich und lebe seit drei Jahren in Deutschland. Ich bin 15 Jahre alt und habe zwei Geschwister: Meine Schwester heißt Klara und ist 13 Jahre alt, mein Bruder Michael ist 18 Jahre alt. Wir wohnen mit unseren Eltern in einem Haus in der Nähe von München. Meine Mutter ist Köchin, mein Vater arbeitet in einer Bank."

    utterance_kk = "155-ші мотоатқыш дивизиясы, қысқаша 155-ші мад — КСРО Қарулы Күштері Құрлық әскерлері мен Қазақстан Республикасы Қарулы Күштерінің Құрлық әскерлері құрамындағы құрама. 4-шi жеке механикаландырылған бригадасы — 4-ші жмехбр болып жаңадан жасақталды."

    utterance_uz = "101 reys — Yoshlik kinostudiyasi va Kinematografiya agentligi tomonidan tushirilgan oʻzbek filmi. Oʻzbekistonda film birinchi marotaba 1-iyun 2022-yilda koʻrsatilgan va filmning premyerasi Toshkentda „Kinematografiya Uyi“da boʻlib oʻtgan."

    utterance_bg = "Всеки човек има право на образование, hello."

    utterance_pt = "Era uma época de grandes transformações sociais - o começo dos anos setenta - e não havia ainda publicações sérias a respeito de Alquimia."

    utterance_ky = """
            — Дагы, куйчу! — Э, Кид, ашык болуп кетти го дейм? Виски менен спирт
        аралашканда оңой иш болбойт, анын үстүнө коньяк да, перцовка да, анан...
        — Куй дегенде, куя берсеңчи! Ичкилик камдап жаткан ким, менби же сенби?
        — Уюлгуган ала булоонун ара-сынан Мэйлмют Киддин мээримдүү күлүмсүрөгөнү
        көрүнүп турду .— Э-э, балам, ушул өлкөдө мен жашагандай жашап көрсөң,
        анын үстүнө кудайдын куттуу күнү кемиргениң сүрсүгөн балыктын эти болсо
        көрөр элем, рождество деген жылына бир эле жолу болорун дал ошондо гана
        түшүнөсүң. Ал эми ичкиликсиз рождество — кымындай алтыны жок кенге
        окшош! — Бул сөзүңдө түк калет жок! — деди рождествону майрамдоо үчүн
        өзүнүн Мэйзе-Мэйдеги участогунан келген Джим Белден. Чоң Джим акыркы эки
        айдан бери жалаң бугунун эти менен оокаттанып жатканын баары билише
        турган. — Баягыда биз бир жолу Танана уруусуна кандай ичимдик
        уюштурганыбыз эсиңде барбы? Унуткан эместирсиң? — Кантип унутайын!
    """

    utterances = {
        "RU": utterance_ru,
        "EN": utterance_en,
        "ES": utterance_es,
        "DE": utterance_de,
        "KK": utterance_kk,
        "UZ": utterance_uz,
        "BG": utterance_bg,
        "PT-BR": utterance_pt,
        "KY": utterance_ky,
    }

    _doc = utterances[_lang]

    text_processor = TextParser(lang=_lang, device=_device)
    text_processor.process(Doc(_doc))

    with Profiler(format=Profiler.Format.ms):
        _doc = text_processor.process(Doc(_doc))

    for idx, s in enumerate(_doc.sents):
        print(f"Sentence {idx+1}")
        print("text:", s.text)
        print("----")
        print("stress:", s.get_attr("stress"))
        print("----")
        print("pos:", s.get_attr("pos"))
        print("----")
        # print("feats:", s.get_attr("feats"))
        # print("----")
        print("rel:", s.get_attr("rel"))
        print("----")
        print("phonemes:", s.get_attr("phonemes"))
        print("----")
        print([i for i in s.get_attr("text")])
        print("----")
        if _lang == "RU":
            try:
                s.syntax.print()
            except:
                pass
    print("*" * 10)
    print("doc.exception_messages", _doc.exceptions)
    print("*" * 10)

    print("phonemes:", text_processor.phonemes)
    print("pos:", text_processor.pos)
    print("rel:", text_processor.rel)
