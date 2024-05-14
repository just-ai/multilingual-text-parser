import re
import typing as tp
import logging

from multilingual_text_parser.data_types import Sentence, Token
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler

__all__ = ["NormalizerKY"]

LOGGER = logging.getLogger("root")


class NormalizerKY(BaseSentenceProcessor):
    def __init__(self):
        self.hundred = "жүз"
        self.zero = "ноль"
        self.point = "бүтүн"

        self.ones = {
            1: "бир",
            2: "эки",
            3: "үч",
            4: "төрт",
            5: "беш",
            6: "алты",
            7: "жети",
            8: "сегиз",
            9: "тогуз",
        }

        self.tens = {
            1: "он",
            2: "жыйырма",
            3: "отуз",
            4: "кырк",
            5: "элүү",
            6: "алтымыш",
            7: "жетимиш",
            8: "сексен",
            9: "токсон",
        }

        self.triplets = {
            1: "",
            2: "миң",
            3: "миллион",
            4: "миллиард",
            5: "триллион",
            6: "квадриллион",
            7: "секстиллион",
            8: "септиллион",
            9: "октиллион",
        }

        self.ones_ordinal = {
            0: "унчу",
            1: "инчи",
            2: "нчи",
            3: "үнчү",
            4: "үнчү",
            5: "инчи",
            6: "нчы",
            7: "нчи",
            8: "инчи",
            9: "унчу",
        }

        self.tens_ordinal = {
            1: "унчу",
            2: "нчы",
            3: "унчу",
            4: "ынчы",
            5: "нчү",
            6: "ынчы",
            7: "инчи",
            8: "чи",
            9: "чу",
        }

        self.triplets_ordinal = {
            0: "унчу",
            1: "үнчү",
            2: "инчи",
            3: "унчу",
            4: "ынчы",
            5: "унчу",
            6: "унчу",
            7: "унчу",
            8: "унчу",
            9: "унчу",
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

        self.tens_ablative = {
            1: "дон",
            2: "дан",
            3: "дан",
            4: "тан",
            5: "дөн",
            6: "тан",
            7: "тен",
            8: "ден",
            9: "дон",
        }

        self.triplets_ablative = {
            1: "дөн",
            2: "ден",
            3: "дон",
            4: "дан",
            5: "дон",
            6: "дон",
            7: "дон",
            8: "дон",
            9: "дон",
        }

        self.months_kg = {
            1: "январь",
            2: "февраль",
            3: "март",
            4: "апрель",
            5: "май",
            6: "июнь",
            7: "июль",
            8: "август",
            9: "сентябрь",
            10: "октябрь",
            11: "ноябрь",
            12: "декабрь",
        }

        self.curency = {
            "$": ("доллар", "цент"),
            "£": ("фунт стерлинг", "цент"),
            "€": ("евро", "цент"),
            "¥": ("юань", "фынь"),
            "₽": ("рубль", "тыйын"),
            "с": ("сом", "тыйын"),
        }

        # регулярки на разные форматы дат dmy/mdy/ymd...
        self._date1 = re.compile(
            r"^([0-2]?\d|3[01])[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d{3}|\d{2}))$"
        )
        self._date2 = re.compile(
            r"^(0?\d|1[0-2])[\./\-]([0-2]?\d|3[01])([\./\-](([0-2]?\d{3})|\d{2}))$"
        )
        self._date3 = re.compile(
            r"^([0-2]?\d{3}|\d{2})[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d|3[01]))$"
        )
        self._date4 = re.compile(
            r"^([0-2]?\d{3}|\d{2})[\./\-]([0-2]?\d|3[01])([\./\-](0?\d|1[0-2]))$"
        )

        self._symbols = {
            "%": " процент",
            "№": " саны ",
            "+": " плюс ",
            "-": " минус ",
        }
        self._nums = re.compile(r"(\d+)?[.,]?\d+")
        self._phone = re.compile(r"\+996(\d{9}|\-\d{3}\-\d{3}\-\d{3})")
        self._symbols_pattern = re.compile(r"^[№+-]|%")
        self._currency = re.compile(r"[$€¥£₽с]\d+|\d+([$€¥£₽с])")
        self._ordinal = re.compile(r"^\d+\-[^\d\W]+$")

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        for token in sent.tokens:
            text = token.text
            text = self._symbols_pattern.sub(lambda x: self._symbols[x.group()], text)

            date = self.convert_date(token)
            if date:
                token.text = date
            elif self._currency.search(text):
                token.text = self.convert_currency(text)
            elif self._ordinal.search(text):
                parts = text.split("-")
                token.text = self.to_ordinal(parts[0]) + " " + parts[1]
            elif self._phone.search(token.text):
                token.text = self.convert_phone(token)
            elif token.is_number:
                num = text.replace(",", ".")
                if token.interpret_as and token.interpret_as == "ordinal":
                    token.text = self._nums.sub(
                        lambda x: self.to_ordinal(x.group()),
                        num,
                    )
                elif float(self._nums.search(num).group()) < 1000000000000000000:
                    token.text = self._nums.sub(
                        lambda x: f" {self.point} ".join(
                            [self.to_word(i) for i in x.group().split(".")]
                        ),
                        num,
                    )
                else:
                    token.text = self._nums.sub(
                        lambda x: " ".join([self.to_word(s) for s in x.group()]), num
                    )

        if sent.ssml_insertions:
            ssml_insertions_pos = [
                sent.tokens[idx] if idx >= 0 else None for idx, _ in sent.ssml_insertions
            ]
        else:
            ssml_insertions_pos = []

        tokens = sent.tokens
        tokens = self._split_words(
            tokens, symb=" ", ssml_insertions_pos=ssml_insertions_pos
        )
        tokens = self._split_words(
            tokens, symb="-", ssml_insertions_pos=ssml_insertions_pos
        )
        tokens = self._clean(tokens)
        sent.tokens = tokens

        if sent.ssml_insertions:
            for idx, (item, token) in enumerate(
                zip(sent.ssml_insertions, ssml_insertions_pos)
            ):
                if item[0] >= 0:
                    new_pos = tokens.index(token)
                    sent.ssml_insertions[idx] = (new_pos, item[1])

    def to_word(self, number: str) -> str:
        if abs(int(number)) == 0:
            return self.zero

        def convert(number: str):
            def parse_tens(number):
                tens = number // 10
                rem = self.ones[number % 10] if number % 10 > 0 else ""
                return f"{self.tens[tens]} {rem}" if tens else rem

            def parse_hundreds(number):
                hundreds = number // 100
                rem = parse_tens(number % 100) if number % 100 > 0 else ""
                if hundreds > 1:
                    return (
                        f"{self.ones[hundreds]} {self.hundred} {rem}"
                        if rem
                        else f"{self.ones[hundreds]} {self.hundred}"
                    )
                else:
                    return f"{self.hundred} {rem}"

            return parse_hundreds(number) if number >= 100 else parse_tens(number)

        def generate(number: str):
            triplets = self.to_triplets(str(number))
            words = []
            for index, word in enumerate(triplets):
                if int(word) != 0:
                    word = convert(int(word)).strip()
                    if word:
                        words.append(f"{word} {self.triplets[index + 1]}")
            return " ".join(words[::-1]).strip()

        return generate(number)

    def to_ordinal(self, number: str) -> str:
        def parseHundreds(number: str) -> str:
            suffix = (
                self.ones_ordinal[number % 10]
                if number % 10
                else self.tens_ordinal[(number % 100) // 10]
            )
            suffix = suffix or self.triplets_ordinal[1]
            return suffix

        def parseThousands(number: str) -> str:
            triplets = self.to_triplets(str(number))
            thousands = 0
            for triple in triplets:
                if int(triple) == 0:
                    thousands += 1
                else:
                    return (
                        self.triplets_ordinal[thousands + 1]
                        if thousands
                        else parseHundreds(int(triple))
                    )

        def generateSuffix(number: str) -> str:
            if abs(int(number)) == 0:
                return self.triplets_ordinal[0]
            return (
                parseThousands(int(number))
                if 1e3 < int(number)
                else parseHundreds(int(number))
            )

        return self.to_word(number) + generateSuffix(number)

    def to_triplets(self, number: str) -> str:
        triplets = []
        while number:
            triplets.append(number[-3:])
            number = number[: min(-3, len(number))]
        return triplets

    def convert_currency(self, text: str) -> str:
        num = self._nums.search(text).group()
        num = [self.to_word(n) for n in num.replace(",", ".").split(".")]
        cur = re.search("[$€¥£₽с]", text).group()
        currency = num[0] + " " + self.curency[cur][0]
        if len(num) == 2:
            currency = currency + " " + num[1] + " " + self.curency[cur][1]

        return currency

    def convert_date(self, token: Token) -> str:
        date2str = ""
        day, month, y = None, None, None
        format = token.format

        if self._date1.search(token.text) and (not format or format in ["dmy", "dm"]):
            date = self._date1.search(token.text)
            day, month, y = date[1], date[2], date[4]
        elif self._date2.search(token.text) and (not format or format in ["mdy", "md"]):
            date = self._date2.search(token.text)
            day, month, y = date[2], date[1], date[4]
        elif self._date3.search(token.text) and (not format or format in ["ymd", "ym"]):
            date = self._date3.search(token.text)
            day, month, y = date[4], date[2], date[1]
        elif self._date4.search(token.text) and (not format or format in ["ydm", "yd"]):
            date = self._date4.search(token.text)
            day, month, y = date[2], date[4], date[1]
        elif format:
            day, month, y = self.split_wo_sep(token.text, format)

        if day:
            date2str += self.to_ordinal(day)
            date2str += " "
        if month:
            date2str += self.months_kg[int(month)]
            date2str += " "
        if y:
            date2str += self.to_word(y)

        return date2str

    @staticmethod
    def split_wo_sep(text: str, format: str) -> tp.List[str]:
        d = m = y = "0"
        for f in format:
            if f == "d":
                d = text[:2]
                text = text[2:]
            if f == "m":
                m = text[:2]
                text = text[2:]
            if f == "y":
                y = text[:4]
                text = text[4:]
        return d, m, y

    def convert_phone(self, token: Token) -> str:
        phone = ["плюс"]
        text = token.text.replace("+", "")

        if "-" in text:
            parts = text.split("-")
            phone.extend([self.to_word(part) for part in parts])
        else:
            while text:
                phone.append(text[:3])
                text = text[3:]
        phone = [self.to_word(p) if p.isdigit() else p for p in phone]

        return " ".join(phone)

    @staticmethod
    def _split_words(
        tokens: tp.List[Token],
        symb: str,
        ssml_insertions_pos: tp.Optional[tp.List] = None,
    ) -> tp.List[Token]:
        new_tokens = []
        for token in tokens:
            if not token.is_punctuation and symb in token.text.strip(symb):
                words = token.text.split(symb)  # type: ignore
                words = [x for x in words if len(x) > 0]
                if token.stress:
                    words_stress = token.stress.split(symb)  # type: ignore
                    words = [x for x in words_stress if len(x) > 0]
                    assert len(words) == len(words_stress)
                else:
                    words_stress = [None] * len(words)

                for word, stress in zip(words, words_stress):
                    new_token = Token(word)
                    for attr in [
                        "pos",
                        "modifiers",
                        "normalized",
                        "emphasis",
                        "id",
                        "head_id",
                        "rel",
                        "meta",
                    ]:
                        value = getattr(token, attr)
                        setattr(new_token, attr, value)
                    if stress:
                        new_token.stress = stress
                    new_tokens.append(new_token)

                if ssml_insertions_pos:
                    if token in ssml_insertions_pos:
                        idx = ssml_insertions_pos.index(token)
                        ssml_insertions_pos.remove(token)
                        ssml_insertions_pos.insert(idx, new_tokens[-1])
            else:
                new_tokens.append(token)

        return new_tokens

    @staticmethod
    def _clean(tokens: tp.List[Token]) -> tp.List[Token]:
        new_tokens = []
        find_digit = re.compile(r"[\d]")
        for token in tokens:
            if token.is_punctuation or not find_digit.findall(token.text):
                new_tokens.append(token)

        return new_tokens
