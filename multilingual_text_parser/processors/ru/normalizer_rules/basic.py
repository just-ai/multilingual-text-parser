import re
import typing as tp
import logging

from transliterate import translit

from multilingual_text_parser.data_types import Token
from multilingual_text_parser.processors.ru.normalizer_utils import BaseRule, Utils
from multilingual_text_parser.utils.fs import get_root_dir

LOGGER = logging.getLogger("root")


class AddressRules(BaseRule):
    """
    Класс для обработки адресов
    Сейчас
    1. "г" заменяет на город (не склоняет)
    2. число с дробью приводит к виду "_число_ дробь _число_"
    """

    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._pattern = re.compile(r"дом|здание")

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]):
        if text == "г":
            if idx > 0:
                if not re.match(r"\d+", tokens[idx - 1].text):
                    text = "город"
                    return text
            else:
                text = "город"
                return text
        elif (
            (text in ["с", "стр"])
            and self._pattern.search(" ".join([token.text for token in tokens]))
            and idx + 1 < len(tokens)
            and tokens[idx + 1].text.isdigit()
        ):
            if text in ["с"] and tokens[idx + 1].text != ".":
                return

            return "строение"
        elif re.match(r"\d+/\d+", text) and idx > 0:
            if tokens[idx - 1].text in ["дом", "корпус", "строение", "здание"]:
                num1, sign, num2 = re.sub(r"(\d+)/(\d+)", r"\1 дробь \2", text).split()
                num1 = self._num2words.transform(num1)
                num2 = self._num2words.transform(num2)
                return " ".join([num1, sign, num2])
        return

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        _text = self.check_format(idx, _text, tokens)
        if _text:
            tokens[idx].text = _text


class NumAdjRules(BaseRule):
    # 20-летний/кратный/разовый и т.д.
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]):
        words = text.split("-")
        if text.count("-") == 1:
            if words[0].isdigit() and re.search(
                re.compile(r"[а-яА-ЯйЙёЁ]{4,}"), words[1]
            ):
                return words

        elif tokens[idx].text.isdigit() and tokens[idx + 1].text == "-":
            if re.search(re.compile(r"(и|или|,)"), tokens[idx + 2].text) and (
                re.search(re.compile(r"\d+\-[а-яА-ЯйЙёЁ]{4,}"), tokens[idx + 3].text)
                or re.search(re.compile(r"\d+"), tokens[idx + 3].text)
                and tokens[idx + 4].text == "-"
            ):
                return words
            elif (
                re.search(re.compile(r"\d+"), tokens[idx + 2].text)
                and tokens[idx + 3].text == "-"
                and re.search(re.compile(r"(и|или|,)"), tokens[idx + 4].text)
                or re.search(re.compile(r"\d+\-[а-яА-ЯйЙёЁ]{4,}"), tokens[idx + 2].text)
            ):
                return words

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        words = self.check_format(idx, _text, tokens)

        if words:
            if len(words) >= 1:
                if words[0].endswith("1") and not words[0].endswith("11"):
                    _text = (self._num2words.transform(words[0], case="рд", number="ед"))[
                        :-2
                    ]
                else:
                    _text = self._num2words.transform(words[0], case="рд", number="ед")

            if len(words) == 1:
                tokens[idx].text = _text
            else:
                tokens[idx].text = f"{_text} {words[1]}".replace(" ", "")


class DataPeriodRules(BaseRule):
    # с "гггг" (по/до) "гггг"/с "гггг"/по "гггг"/с мая 2020/ с 12 мая 2013 и т.д.
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        vocab_root = get_root_dir() / "data/ru/vocabularies"
        self._months = Utils.read_vocab(vocab_root / "months.txt", as_list=True)
        self._months = dict(zip(range(1, len(self._months) + 1), self._months))

        self._months_all_form = []
        for month in self._months.values():
            self._months_all_form += self._morph.get_lexeme(month, with_meta=False)
        self._months_all_form = set(self._months_all_form)

        self._triger_words_reduction = {"г": "год", "гг": "год"}

    @staticmethod
    def check_preposition(num_id: int, tokens: tp.List[Token]):
        if num_id >= 0:
            if (
                tokens[num_id].text == "с"
                or tokens[num_id].text == "по"
                or tokens[num_id].text == "до"
            ):
                return tokens[num_id].text

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]) -> str:
        # ф-ция возвращает id_числит-ого, обозн-щий год, и его падеж
        if text.isdigit() and 900 < int(text) < 2500:
            if self.check_preposition(idx - 1, tokens):
                # для фраз с/по 2020 или с 2019 по 2030
                preposition = self.check_preposition(idx - 1, tokens)
                if preposition != "до":
                    return preposition
                else:
                    context = tokens[: idx - 1]
                    context.reverse()
                    counter = 0
                    for token in context:
                        counter += 1
                        if counter <= 6 and token.text == "с":
                            return preposition
                        elif counter > 6:
                            break

            elif tokens[idx - 1].text in self._months_all_form:
                if self.check_preposition(idx - 2, tokens):
                    # для фраз с/по мая 2020 или с мая 2019 по август 2030 и другие вариации
                    preposition = self.check_preposition(idx - 2, tokens)
                    return preposition

                elif (
                    self.check_preposition(idx - 3, tokens)
                    and tokens[idx - 2].text.isdigit()
                ):
                    # для фраз с/ 17 мая 2020 или с 17 мая 2019 по август 2030 и другие вариации
                    preposition = self.check_preposition(idx - 3, tokens)
                    return preposition

        return ""

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        preposition = self.check_format(idx, _text, tokens)
        num_case = ""
        if preposition == "с" or preposition == "до":
            num_case = "рд"
        elif preposition == "по":
            num_case = "им"
        if num_case != "":
            tokens[idx].text = self._num2words.transform(
                _text, case=num_case, number="ед", ordinal=True
            )
            if (
                idx + 1 < len(tokens)
                and tokens[idx + 1].text in self._triger_words_reduction
            ):
                year = self._triger_words_reduction[tokens[idx + 1].text]
                tokens[idx + 1].text = self._morph.inflect(year, [num_case, "ед"])


class ReductionInDocsRules(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._triger_words_reduction = {
            "абз": "абзац",
            "граф": "график",
            "диагр": "диаграмма",
            "р": "рисунок",
            "рис": "рисунок",
            "с": "страница",
            "стр": "страница",
            "табл": "таблица",
        }

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]) -> bool:
        if text in self._triger_words_reduction:
            return True

        return False

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        if self.check_format(idx, _text, tokens):
            preposition = Utils.find_prepositions(tokens, idx, left=2)
            if preposition:
                preposition = preposition.keys()  # type: ignore
                if "на" in preposition or "в" in preposition:
                    case = "пр"
                    if idx > 0 and tokens[idx - 1] and tokens[idx - 1].text.isdigit():
                        if _text != "с" or _text != "р":
                            _text = self._triger_words_reduction[_text]
                            gender = self._morph.get_gender(_text)
                            tokens[idx].text = self._morph.inflect(_text, [case])
                            tokens[idx - 1].text = self._num2words.transform(
                                tokens[idx - 1].text,
                                case=case,
                                gender=gender,
                                number="ед",
                                ordinal=True,
                            )

                    elif (
                        (idx < len(tokens) - 1)
                        and tokens[idx + 1]
                        and tokens[idx + 1].text.isdigit()
                    ) or (
                        (idx < len(tokens) - 2)
                        and tokens[idx + 2]
                        and tokens[idx + 2].text.isdigit()
                    ):
                        if _text in ["с", "р"] and tokens[idx + 1].text != ".":
                            return

                        _text = self._triger_words_reduction[_text]
                        tokens[idx].text = self._morph.inflect(_text, [case])
                        tokens[idx].is_preposition = False


class PhoneRules(BaseRule):
    def check_format(self, text: str, interpret_as: tp.Optional[str]) -> tp.List[str]:
        if text.count("-") > 1 and not text.startswith("-"):
            phone = text.split("-")
            if all(digit.replace("+", "").isdigit() for digit in phone):
                return phone

        phone = []
        if text.startswith("+"):
            phone.append(text[:1])
            text = text[1:]

        if interpret_as == "telephone" and text.isdigit():
            if phone:
                phone[0] += text[:1]
                text = text[1:]
            if text.startswith("7") or text.startswith("8"):
                phone.append(text[:1])
                text = text[1:]
            if len(text) == 10:
                for i in [3, 3, 2, 2]:
                    phone.append(text[:i])
                    text = text[i:]
            elif len(text) == 6:
                for i in [3, 3]:
                    phone.append(text[:i])
                    text = text[i:]
            elif len(text) == 4:
                for i in [2, 2]:
                    phone.append(text[:i])
                    text = text[i:]
            else:
                k = 3
                while text:
                    phone.append(text[:k])
                    text = text[k:]
                    if len(text) % k != 0:
                        k -= 1

            return phone

        return []

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[idx]
        phone = self.check_format(current_token.text, current_token.interpret_as)
        if phone:
            phone2str = []
            for digit in phone:
                num_start_zero = 0
                while digit.startswith("0"):
                    num_start_zero += 1
                    digit = digit[1:]
                if num_start_zero == 1:
                    phone2str.append("ноль")
                elif num_start_zero > 1:
                    phone2str.append(
                        self._num2words.transform(num_start_zero, noun="ноль")
                    )
                if digit:
                    phone2str.append(self._num2words.transform(digit))
            current_token.text = " - ".join(phone2str)


class TimeRules(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._time_word = ["час", "минута", "секунда"]

    def check_format(self, text: str):
        if text.count(":") == 2:
            h, m, s = text.split(":")
            if h.isdigit() and m.isdigit() and s.isdigit():
                hi, mi, si = int(h), int(m), int(s)
                if 0 <= hi <= 23 and 0 <= mi <= 59 and 0 <= si <= 59:
                    return hi, mi, si

        if text.count(":") == 1:
            h, m = text.split(":")
            if h.isdigit() and m.isdigit():
                hi, mi = int(h), int(m)
                if 0 <= hi <= 23 and 0 <= mi <= 59:
                    return hi, mi, None

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[idx]
        if (
            current_token.text == "часа"
            and idx > 0
            and tokens[idx - 1].text in ["два", "три", "четыре"]
        ):
            current_token.stress = "часа+"

        if idx > 0 and not tokens[idx - 1].is_preposition:
            return

        time = self.check_format(current_token.text)
        if time:
            h, m, s = time
            case = "им"

            prepositions = Utils.find_prepositions(tokens, idx, right=1)
            if prepositions:
                last_preposition = list(prepositions.items())[0]
                if last_preposition[1]["text"] not in ["за", "на"]:
                    case = last_preposition[1]["case"]
                if last_preposition[1]["text"] in ["в", "во"]:
                    case = "им"

            time = []
            time += [self._num2words.transform(h, case, noun="час")]
            time += [self._num2words.transform(m, case, noun="минута")]
            if s:
                time += [self._num2words.transform(s, case, noun="секунда")]

            time2str = " ".join(time)
            if time2str.endswith("часе"):
                time2str = time2str.replace("часе", "часу")

            time2str = time2str.replace("ноль минут", "ро+вно")
            time2str = time2str.replace("мину", "мину+")
            time2str = time2str.replace("часа", "часа+")
            current_token.text = time2str


class DigitRules(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._digit_search = re.compile(r"[0-9]")
        self._pattern = re.compile(r"([0-9]+)")

    def check_format(self, text: str) -> bool:
        if self._digit_search.search(text):
            return True
        else:
            return False

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[idx]
        if self.check_format(current_token.text):
            digit = self._pattern.search(current_token.text)
            while digit:
                a, b = digit.span()
                digit_to_str = self._num2words.transform(digit.group(1))
                current_token.text = (
                    f"{current_token.text[:a]} {digit_to_str} {current_token.text[b:]}"
                )
                digit = self._pattern.search(current_token.text)

            current_token.text = re.sub(r"[^\w\s]", "", current_token.text).strip()


class Translit(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._latin_search = re.compile(r"[a-zA-Z]")
        self._cyr_search = re.compile(r"[а-яА-Я]")
        lat = "aceopxqmwykiebdrtvlnuhfQACEHKMOPTXIWYKBDRTVLHUF"
        cyr = "асеорхкмвукиевдртвлнунфКАСЕНКМОРТХИВУКВДРТВЛНУФ"
        self._trans = str.maketrans(lat, cyr)

    def check_format(self, text: str) -> bool:
        if self._latin_search.search(text):
            return True
        else:
            return False

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        if self.check_format(_text):
            if not self._cyr_search.search(_text) and len(_text) > 1:
                _text = translit(_text.lower(), "ru")
            tokens[idx].text = _text.translate(self._trans)


class RomanRules(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._rom_val = {
            "I".lower(): 1,
            "V".lower(): 5,
            "X".lower(): 10,
            "L".lower(): 50,
            "C".lower(): 100,
            "D".lower(): 500,
            "M".lower(): 1000,
        }

    def check_format(self, text: str) -> bool:
        return set(text) <= set(self._rom_val.keys())

    def roman_to_int(self, s):
        int_val = 0
        for i in range(len(s)):
            if i > 0 and self._rom_val[s[i]] > self._rom_val[s[i - 1]]:
                int_val += self._rom_val[s[i]] - 2 * self._rom_val[s[i - 1]]
            else:
                int_val += self._rom_val[s[i]]
        return str(int_val)

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[idx]
        if self.check_format(current_token.text):
            case = None
            gender = "мр"
            number = "ед"
            ordinal = False
            if idx + 1 < len(tokens) and tokens[idx + 1].text.startswith("век"):
                case = self._morph.get_case(tokens[idx + 1].text)
                number = self._morph.get_number(tokens[idx + 1].text)
                ordinal = True
                gender = "ср"
            elif idx > 0:
                case = self._morph.get_case(tokens[idx - 1].text)
                gender = self._morph.get_gender(tokens[idx - 1].text)
                ordinal = True

            current_token.text = self._num2words.transform(
                self.roman_to_int(current_token.text),
                case=case,
                gender=gender,
                number=number,
                ordinal=ordinal,
            )
