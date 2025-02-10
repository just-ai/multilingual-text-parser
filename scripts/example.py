import logging

from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.parser import TextParser
from multilingual_text_parser.utils.profiler import Profiler

LOGGER = logging.getLogger("root")


def parse(utterance: str, lang: str, device: str = "cpu", view: bool = False):
    text_parser = TextParser(lang=lang, device=device, with_profiler=False)
    text_parser.process(Doc(utterance))

    prof = Profiler(format=Profiler.Format.ms, auto_logging=False)

    for _ in range(10):
        doc = text_parser.process(Doc(utterance))

    print(f"{lang}: {prof.get_time() / 10} ms")
    print("---------")
    print(doc.text)
    print("---------")
    print(doc.capitalize)
    print("---------")
    print(doc.stress)
    print("---------")

    if 1:
        for sent in doc.sents:
            print(f"'{sent.capitalize}'")
            for tk in sent.tokens:
                print(
                    f"\t'{tk.text}' -> "
                    f"{tk.text_orig}|{tk.stress}|{tk.pos}|{tk.rel}|"
                    f"{tk.is_capitalize}|{tk.is_punctuation}|{tk.is_abbreviation}|"
                    f"{tk.phonemes}"
                )

    if view:
        for idx, sent in enumerate(doc.sents):
            print(sent.stress)
            LOGGER.warning(" ".join(sent.warning_messages))
            LOGGER.error(" ".join(sent.exception_messages))


if __name__ == "__main__":
    _device = "cpu"

    _utterance_ru = """
    Объем валовой добавленной стоимости в сельском хозяйстве, охоте и лесном хозяйстве России — 1,53 трлн руб.
    По данным Росстата, в 2007 г. общий валовой продукт сельского хозяйства России составил 2099,6 млрд руб.,
    из которых на растениеводство (земледелие) приходилось 1174,9 млрд руб. (55,96%), а на животноводство — 924,7 млрд руб.
    По категориям производителей больше всего продукции дали личные подсобные хозяйства (48,75% или на сумму 1023,6 млрд руб.);
    на втором месте — сельскохозяйственные организации, давшие 43,76% или 918,7 млрд руб.;
    меньше всего произвели фермерские хозяйства — 7,49% или на сумму 157,3 млрд руб.
    """

    parse(_utterance_ru, "RU", _device)

    _utterance_en = """
    Russia, or the Russian Federation, is a transcontinental country spanning Eastern Europe and Northern Asia.
    It is the largest country in the world by area, covering over 17,125,191 square kilometres (6,612,073 sq mi), and encompassing one-eighth of Earth's inhabitable landmass.
    Russia extends across eleven time zones and borders sixteen sovereign nations, the most of any country in the world. It is the ninth-most populous country and the most populous country
    in Europe, with a population of 145.5 million. Moscow, the capital, is the largest city entirely within Europe, while Saint Petersburg is the country's second-largest city and cultural centre.
    Other major urban areas include Novosibirsk, Yekaterinburg, Nizhny Novgorod and Kazan.
    """

    parse(_utterance_en, "EN", _device)
