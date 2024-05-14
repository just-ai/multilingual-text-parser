import re
import logging

from multilingual_text_parser.data_types import Token
from multilingual_text_parser.processors.ky.normalizer import NormalizerKY
from multilingual_text_parser.processors.ru.normalizer_rules.basic import PhoneRules
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["NormalizerKK"]

LOGGER = logging.getLogger("root")


class NormalizerKK(NormalizerKY):
    def __init__(self):
        super().__init__()
        self._lang = "kz"
        self.hundred = "жүз"
        self.zero = "нөл"
        self.point = "бүтін"

        self.ones = {
            1: "бір",
            2: "екі",
            3: "үш",
            4: "төрт",
            5: "бес",
            6: "алты",
            7: "жеті",
            8: "сегіз",
            9: "тоғыз",
        }

        self.tens = {
            1: "он",
            2: "жиырма",
            3: "отыз",
            4: "қырық",
            5: "елу",
            6: "алпыс",
            7: "жетпіс",
            8: "сексен",
            9: "тоқсан",
        }

        self.triplets = {
            1: "",
            2: "мың",
            3: "миллион",
            4: "миллиард",
            5: "триллион",
            6: "квадриллион",
            7: "секстиллион",
            8: "септиллион",
            9: "октиллион",
        }

        self.ones_ordinal = {
            0: "інші",
            1: "інші",
            2: "нші",
            3: "інші",
            4: "інші",
            5: "інші",
            6: "ншы",
            7: "нші",
            8: "інші",
            9: "інші",
        }

        self.tens_ordinal = {
            1: "ыншы",
            2: "сыншы",
            3: "ыншы",
            4: "ыншы",
            5: "інші",
            6: "ыншы",
            7: "інші",
            8: "інші",
            9: "ыншы",
        }

        self.triplets_ordinal = {
            0: "ыншы",
            1: "інші",
            2: "ыншы",
            3: "ыншы",
            4: "ыншы",
            5: "ыншы",
            6: "ыншы",
            7: "ыншы",
            8: "ыншы",
            9: "ыншы",
        }

        self.ones_ablative = {
            1: "ден",
            2: "ден",
            3: "төн",
            4: "төн",
            5: "тен",
            6: "дан",
            7: "ден",
            8: "ден",
            9: "дан",
        }

        self.curency = {
            "$": ("доллар", "цент"),
            "£": ("фунт стерлинг", "цент"),
            "€": ("еуро", "цент"),
            "¥": ("юань", "фынь"),
            "₽": ("рубль", "тиын"),
            "₸": ("теңге", "тиын"),
        }

        self.months_ky = {
            1: "қаңтар",
            2: "ақпан",
            3: "наурыз",
            4: "сәуір",
            5: "мамыр",
            6: "маусым",
            7: "шілде",
            8: "тамыз",
            9: "қыркүйек",
            10: "қазан",
            11: "қараша",
            12: "желтоқсан",
        }

        self._symbols = {
            "%": " пайыз ",
            "№": " саны ",
            "+": " плюс ",
            "-": " минус ",
        }

        self._phone = re.compile(
            r"^(\+7|8)[\(]?[0-9]{3}[\)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{2}[-\s\.]?[0-9]{2}$"
        )
        self.phone_rules = PhoneRules(None, None)

    def convert_phone(self, token: Token) -> str:
        phone = self.phone_rules.check_format(
            token.text.replace("(", "-").replace(")", "-"), "telephone"
        )
        if phone:
            phone2str = []
            for digit in phone:
                if digit.startswith("+"):
                    phone2str.append(self._symbols["+"])
                    digit = digit[1:]
                while digit.startswith("0"):
                    phone2str.append(self.zero)
                    digit = digit[1:]
                if digit:
                    phone2str.append(self.to_word(digit))
            return " ".join(phone2str)
        return token.text
