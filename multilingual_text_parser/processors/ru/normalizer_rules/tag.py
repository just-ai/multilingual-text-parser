import re
import typing as tp
import logging

from multilingual_text_parser.data_types import Token
from multilingual_text_parser.processors.ru.normalizer_utils import BaseRule, Utils
from multilingual_text_parser.utils.fs import get_root_dir

LOGGER = logging.getLogger("root")


class FractionRulesTag(BaseRule):
    """Класс для обработки дробных чисел, которые написаны через "/" Не склоняет по
    падежам."""

    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]):
        pass

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        idx = start
        _text = tokens[idx].text
        res = _text.split("/")
        if res[0] == "1":
            num1 = self._num2words.transform(res[0], gender="жр")
            num2 = self._num2words.transform(
                res[1], case="им", number="ед", gender="жр", ordinal=True
            )
            tokens[idx].text = " ".join([num1, num2])
        else:
            num1 = self._num2words.transform(res[0], gender="жр")
            num2 = self._num2words.transform(res[1], case="рд", number="мн", ordinal=True)
            tokens[idx].text = " ".join([num1, num2])


class DateRulesTag(BaseRule):
    # с "гггг" (по/до) "гггг"/с "гггг"/по "гггг"/с мая 2020/ с 12 мая 2013 и т.д.
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        # Словарь {число - месяц}
        vocab_root = get_root_dir() / "data/ru/vocabularies"
        self._int_to_months = Utils.read_vocab(vocab_root / "months.txt", as_list=True)
        self._int_to_months = dict(
            zip(range(1, len(self._int_to_months) + 1), self._int_to_months)
        )

        # словарь для развертывания аббривиатур г/гг
        self._triger_words_reduction = {
            "г": "ед",
            "гг": "мн",
        }
        # словарь для определния падежа по предлогу
        self._prep_to_case = {
            "с": "рд",
            "до": "рд",
            "от": "рд",
            "около": "рд",
            "по": "им",
            "на": "им",
            "в": "пр",
            "к": "дт",
            "датирован": "тв",
        }
        # словарь {месяц - число}
        self._months_to_int = {
            "янв": 1,
            "фев": 2,
            "мар": 3,
            "апр": 4,
            "мая": 5,
            "июн": 6,
            "июл": 7,
            "авг": 8,
            "сен": 9,
            "окт": 10,
            "ноя": 11,
            "дек": 12,
        }

        # триггерные регулярки
        self.word_ingen = re.compile(
            r"(середин[еы]|начал[еа]|конц[еа]|половин[еы]|трети|уровн[юя])"
        )
        self.word_innom = re.compile(r"(середину|начало|конец|половину|треть)")
        self._kvartal = re.compile(r"квартал")

        # регулярки на паттерн годов и числа даты
        self._year = re.compile(r"^[0-2]?\d{3}(\-([0-2]?\d{3}|\w{1,3}))?$")
        self._year_short = re.compile(r"^\d{1}[0](\-(\d{1}[0]|\w{1,3}))?$")
        self._year_pl = re.compile(r"\-(е|х|ых)")
        self._years = re.compile(r"([0-2]?\d{3})\-([0-2]?\d{3})")
        self._day = re.compile(r"^[0-3]?\d(-?[а-я]{1,3})?$")

        # регулярки на разные форматы дат dmy/mdy/ymd...
        self._date1 = re.compile(
            r"^([0-2]?\d|3[01])[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d{3}|\d{2}))?$"
        )
        self._date2 = re.compile(
            r"^(0?\d|1[0-2])[\./\-]([0-2]?\d|3[01])([\./\-]([0-2]?\d{3})|\d{2})?$"
        )
        self._date3 = re.compile(
            r"^([0-2]?\d{3}|\d{2})[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d|3[01]))?$"
        )
        self._date4 = re.compile(
            r"^([0-2]?\d{3}|\d{2})[\./\-]([0-2]?\d|3[01])([\./\-](0?\d|1[0-2]))?$"
        )
        self._date5 = re.compile(r"^(0?\d|1[0-2])[\./\-]([0-2]?\d{3}|\d{2})$")

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]) -> str:
        pass

    def split_wo_sep(self, text: str, format: str):
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

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        toks = [
            token.text for token in tokens[start:end]
        ]  # собираем тегированный чанк токенов
        for i, token in enumerate(toks):
            idx = start + i

            # находим по регуляркам нужные части (год/месяц/день)
            year = self._year.search(token)
            day = self._day.search(token)
            date2str = []
            month, y = None, None
            format = tokens[idx].format

            if self._date1.search(token) and (not format or format in ["dmy", "dm"]):
                date = self._date1.search(token)
                day, month, y = date[1], date[2], date[4]
            elif self._date2.search(token) and (not format or format in ["mdy", "md"]):
                date = self._date2.search(token)
                day, month, y = date[2], date[1], date[4]
            elif self._date3.search(token) and (not format or format in ["ymd", "ym"]):
                date = self._date3.search(token)
                day, month, y = date[4], date[2], date[1]
            elif self._date4.search(token) and (not format or format in ["ydm", "yd"]):
                date = self._date4.search(token)
                day, month, y = date[2], date[4], date[1]
            elif self._date5.search(token) and (not format or format == "my"):
                date = self._date5.search(token)
                month, y = date[1], date[2]
            elif format:
                day, month, y = self.split_wo_sep(token, format)

            if year is None and day is None and month is None and y is None:
                year = self._year_short.search(token)

            if day:  # обрабатываем день
                if type(day) != str:
                    day = day[0]
                num_case = "рд"
                num_gender = "мр"
                if (idx - 1 >= 0 and tokens[idx - 1].text in ["по", "на"]) or len(
                    tokens
                ) == 2:
                    num_case = "им"
                    num_gender = "ср"
                elif idx + 1 <= len(tokens) and self._kvartal.search(
                    tokens[idx + 1].text
                ):
                    num_case = self._morph.get_case(tokens[idx + 1].text)
                    num_gender = "мр"
                date2str.append(
                    self._num2words.transform(
                        day, case=num_case, gender=num_gender, ordinal=True
                    )
                )
            if month:  # обрабатываем месяц
                date2str.append(
                    self._morph.inflect(self._int_to_months[int(month)], ["рд"])
                )
            if y:  # обрабатываем год (если в составе даты через точку или слеш)
                y = self._num2words.transform(
                    y, case="рд", gender=num_gender, ordinal=True
                )
                if y.startswith("одн"):
                    y = " ".join(y.split()[1:])
                date2str.append(y)
                noun = self._morph.inflect("год", ["рд"])
                if not (toks[-1].startswith("год") and len(toks[-1]) < 6):
                    date2str.append(noun.replace("го", "го+"))
            if date2str:  # собираем дату
                tokens[idx].text = " ".join(date2str)
            if year:  # обрабатываем год (если он сам по себе)
                num_case = "им"
                num_num = "ед"
                if self._year_pl.search(token):
                    num_num = "мн"
                if self._years.search(token):
                    num_num = "мн"
                    years = self._years.search(token)
                    years = [years[1], years[2]]
                else:
                    years = [year[0]]
                if idx - 1 >= 0:
                    if tokens[idx - 1].text in self._prep_to_case:
                        num_case = self._prep_to_case[tokens[idx - 1].text]
                    elif (
                        len(tokens[idx - 1].text) >= 3
                        and tokens[idx - 1].text[:3] in self._months_to_int
                    ):
                        num_case = "рд"
                    else:
                        if re.search(self.word_ingen, tokens[idx - 1].text):
                            num_case = "рд"
                        elif re.search(self.word_innom, tokens[idx - 1].text):
                            num_case = "им"
                        elif idx - 2 >= 0 and tokens[idx - 2].text in self._prep_to_case:
                            if tokens[idx - 1].pos == "ADJ":
                                num_case = self._prep_to_case[tokens[idx - 2].text]
                            else:
                                num_case = "рд"
                if num_case == "пр":
                    years = [
                        year + ("-ом" if not year.endswith("3") else "-ем")
                        if "-" not in year
                        else year
                        for year in years
                    ]
                    if (
                        idx + 1 - start < len(toks)
                        and tokens[idx + 1].text in self._triger_words_reduction
                    ):
                        tokens[idx + 1].text = (
                            "году"
                            if self._triger_words_reduction[tokens[idx + 1].text] == "ед"
                            else "годах"
                        )
                res = []
                for year in years:
                    r = self._num2words.transform(
                        year, case=num_case, number=num_num, ordinal=True
                    )
                    if r.startswith("одн"):
                        r = " ".join(r.split()[1:])
                    res.append(r)
                tokens[idx].text = " ".join(res)
                if (
                    idx + 1 < len(tokens)
                    and tokens[idx + 1].text in self._triger_words_reduction
                ):
                    tokens[idx + 1].text = self._morph.inflect("год", [num_case, num_num])
                    if num_num == "мн" and tokens[idx + 1].text == "года":
                        tokens[idx + 1].text = "годы"


class TimeRulesTag(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        # регулярка, которая ищет паттерн времени "чч:мм:сс"
        self._pattern = re.compile(r"([0-1][0-9]|2[0-3])\:([0-5][0-9])(\:([0-5][0-9]))?")

    def check_format(self, text: str):
        pass

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[start:end][0]  # токен времени всегда один
        time = self._pattern.search(current_token.text.strip())
        if time and time.span() == (0, len(current_token.text.strip())):
            h, m, s = time.group(1), time.group(2), time.group(4)
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


class OrdinalRulesTag(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        vocab_root = get_root_dir() / "data/ru/vocabularies"
        self._suffix = Utils.read_vocab(vocab_root / "num_modifiers.txt")

    def check_format(self, text: str):
        pass

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[start]
        if re.search(r"\d+", current_token.text):
            if start + 1 != len(tokens) and tokens[start + 1].text in self._suffix:
                noun = self._suffix[tokens[start + 1].text]
                tokens.remove(tokens[start + 1])
            else:
                noun = None

            num = self._num2words.transform(current_token.text, ordinal=True, noun=noun)
            current_token.text = num


class DigitRulesTag(BaseRule):
    # TODO: issue with exaples like: "теряет около 1 <prosody>млрд долларов[47]</prosody>."
    # After normalization млрд -> миллиардов w/o modifiers

    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        vocab_root = get_root_dir() / "data/ru/vocabularies"
        self._prefix = Utils.read_vocab(vocab_root / "num_prefix.txt")
        self._suffix = Utils.read_vocab(vocab_root / "num_suffix.txt")
        self._modifiers = Utils.read_vocab(vocab_root / "num_modifiers.txt")
        self._as_ordinal = Utils.read_vocab(
            vocab_root / "num_as_ordinal.txt", as_list=True
        )
        self._big_suffix = {"°f": "градус_по_фаренгейту", "°c": "градус"}
        self._big_modifiers = {"ч": "в час", "мин": "в минуту", "с": "в секунду"}
        self.one_and_half = re.compile(r"1[,.]5")
        self._short_suffix = re.compile(r"\-(й|я|е|х|м|ых|ом|ым)")

    def check_format(self, text: str):
        pass

    def _remove_point(self, idx, tokens):
        if idx + 1 < len(tokens) - 1:
            if tokens[idx + 1].text == ".":
                tokens.remove(tokens[idx + 1])
            elif tokens[idx + 1].text == "!":
                tokens[idx + 1].text = "."

    def _normalize(self, prefix, digit, suffix, is_float, tokens, idx):
        if re.match("^[а-яА-Яa-zA-Z- ]*$", digit):
            return digit

        case = "им"
        gender = "мр"
        ending = None
        ordinal = False

        if suffix in ["градус_по_фаренгейту", "градус"]:
            parts = suffix.split("_")
            ending = " ".join(parts[1:]) if len(parts) > 1 else None
            suffix = "градус"

        if prefix is None or prefix in self._prefix.values():
            prepositions = Utils.find_prepositions(tokens, idx, left=2)
            if prepositions:
                last_preposition = list(prepositions.items())[0]
                if last_preposition[1]["text"] not in ["за", "на"]:
                    case = last_preposition[1]["case"]
                if idx + 1 < len(tokens) and last_preposition[1]["text"] in [
                    "в",
                    "во",
                ]:
                    case = "им"
                    gender = self._morph.get_gender(tokens[idx + 1].norm)
                    if not gender:
                        gender = "жр"
                if is_float and "от" in prepositions:
                    ordinal = True

            if idx + 2 < len(tokens) and suffix is None:
                if re.search("(дом|улица)", " ".join([t.text for t in tokens])):
                    word_gender = self._morph.get_gender(tokens[idx + 1].text)
                    if word_gender:
                        gender = word_gender
                elif tokens[idx + 2].text in self._modifiers or re.match(
                    r"(км|м|см|мм)/(ч|мин|с)", tokens[idx + 2].text
                ):
                    parts = tokens[idx + 2].text.split("/")
                    suffix = self._modifiers[parts[0]]
                    ending = self._big_modifiers[parts[1]] if len(parts) > 1 else None
                    tokens.remove(tokens[idx + 2])
                    self._remove_point(idx + 1, tokens)
                elif tokens[idx + 1].text in self._modifiers or re.match(
                    r"(км|м|см|мм)/(ч|мин|с)", tokens[idx + 1].text
                ):
                    parts = tokens[idx + 1].text.split("/")
                    suffix = self._modifiers[parts[0]]
                    ending = self._big_modifiers[parts[1]] if len(parts) > 1 else None
                    tokens.remove(tokens[idx + 1])
                    self._remove_point(idx, tokens)
                elif tokens[idx + 1].text in self._as_ordinal:
                    suffix = tokens[idx + 1].text
                    gender = self._morph.get_gender(suffix)
                    ordinal = True
                    tokens.remove(tokens[idx + 1])
                else:
                    word_gender = self._morph.get_gender(tokens[idx + 1].text)
                    if word_gender:
                        gender = word_gender

            if self._short_suffix.search(digit):
                if digit.endswith("-х") and not digit.endswith("0-х"):
                    ordinal = False
                else:
                    ordinal = True

        if is_float:
            safe_suffix = None
            if suffix in ["рубль", "доллар", "евро", "юань"]:
                if idx + 1 < len(tokens) and tokens[idx + 1].text in [
                    "тыс",
                    "млн",
                    "млрд",
                    "трлн",
                ]:
                    safe_suffix = suffix
                    suffix = None

            if re.search(self.one_and_half, digit) and len(digit) == 3:
                text = "полтора"
                d2str = self._morph.inflect(text, [case])
                if suffix:
                    if case == "им":
                        d2str += f" {self._morph.inflect(suffix, [case, 'ед'])}"
                    else:
                        d2str += f" {self._morph.inflect(suffix, [case, 'мн'])}"
            else:
                gender = "жр"
                d2str = self._num2words.transform(
                    digit, case=case, gender=gender, noun=suffix
                )
                d2str = d2str.replace("целые", "целых").replace("целыми", "целых")

            if safe_suffix:
                d2str += f" {self._morph.inflect(safe_suffix, ['рд', 'мн'])}"
                suffix = safe_suffix
        else:
            if tokens[idx].interpret_as == "ordinal":
                ordinal = True
            if tokens[idx].interpret_as == "cardinal":
                ordinal = False
            s = ""
            while digit.startswith("0") and len(digit) > 1:
                s += "ноль "
                digit = digit[1:]
            d2str = self._num2words.transform(
                digit, case=case, gender=gender, ordinal=ordinal, noun=suffix
            )
            if s:
                d2str = s + d2str

        if suffix and idx + 1 < len(tokens) and tokens[idx + 1].text in self._modifiers:
            d2str = d2str.split()
            word = tokens[idx + 1].text
            if word in ["тыс", "млн", "млрд", "трлн"]:
                word = self._modifiers[word]
                if is_float:
                    word = self._morph.inflect(word, ["рд", "ед"])
                else:
                    word = self._morph.inflect(
                        word,
                        [
                            self._morph.get_case(d2str[-1]),
                            self._morph.get_number(d2str[-1]),
                        ],
                    )
                d2str.insert(-1, f" {word} ")
                d2str = " ".join(d2str)
                tokens.remove(tokens[idx + 1])
                self._remove_point(idx, tokens)
            else:
                word = self._modifiers[word]
                word = self._morph.inflect(
                    word,
                    [self._morph.get_case(d2str[-1]), "мн"],
                )
                d2str = " ".join(d2str + [word])
                tokens.remove(tokens[idx + 1])
                self._remove_point(idx, tokens)

        if suffix:
            self._remove_point(idx, tokens)

        if prefix:
            d2str = f"{prefix} {d2str}"

        if d2str.endswith("часе"):
            d2str = d2str.replace("часе", "часу")
        if d2str.endswith("штуки") and d2str.count(" ") > 1:
            d2str = d2str.replace("штуки", "штук")
        if ending:
            d2str += " " + ending

        return d2str

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        for i, t in enumerate(tokens[start:end]):
            if re.search(r"\d+", t.text):
                current_token = t
                text = current_token.text
                is_float = False
                prefix, suffix = None, None
                composition = None
                middle = None
                result = []

                if text.startswith("№"):
                    result.append("номер")
                    text = text[1:]

                if text.count(".") > 1 or text.count(",") > 1:
                    text = text.replace(".", "").replace(",", "")

                if text and text[0] in self._prefix:
                    if text[0] in ["$", "£", "€", "¥", "₽"]:
                        suffix = self._prefix[text[0]]
                    else:
                        prefix = self._prefix[text[0]]
                    text = text[1:]

                if text and text[-1] in self._suffix:
                    suffix = self._suffix[text[-1]]
                    text = text[:-1]
                elif text and text[-2:] in self._big_suffix:
                    suffix = self._big_suffix[text[-2:]]
                    text = text[:-2]

                if "." in text or "," in text:
                    is_float = True

                if "/" in text:
                    composition = text.split("/")
                    middle = " дробь "
                elif re.search("[:()-]", text) and not re.search("[а-я]+", text):
                    composition = re.split("[:()-]", text)
                elif "-" in text and not re.search(r"^\d+-[а-я]+$", text):
                    composition = text.split("-")

                if composition:
                    for i, digit in enumerate(composition):
                        if digit.isdigit():
                            result.append(self._num2words.transform(digit))
                        else:
                            result.append(digit)
                        if middle and i < len(composition) - 1:
                            result.append(middle)
                else:
                    result.append(
                        self._normalize(prefix, text, suffix, is_float, tokens, start + i)
                    )

                current_token.text = " ".join(result)


class RomanRulesTag(BaseRule):
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
        self._expr = re.compile(r"^[ixvlcdm]+$")

    def check_format(self, text: str) -> bool:
        pass

    def roman_to_int(self, s):
        int_val = 0
        for i in range(len(s)):
            if i > 0 and self._rom_val[s[i]] > self._rom_val[s[i - 1]]:
                int_val += self._rom_val[s[i]] - 2 * self._rom_val[s[i - 1]]
            else:
                int_val += self._rom_val[s[i]]
        return str(int_val)

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        idx = start
        current_token = tokens[idx]
        num = self._expr.search(current_token.text)
        if num:
            case = None
            gender = "мр"
            number = "ед"
            ordinal = False
            if idx + 1 < len(tokens) and tokens[idx + 1].text.startswith("век"):
                case = self._morph.get_case(tokens[idx + 1].text)
                number = self._morph.get_number(tokens[idx + 1].text)
                ordinal = True
                gender = "ср"
            if (
                idx + 2 < len(tokens)
                and tokens[idx + 1].text.startswith("в")
                and tokens[idx + 2].text.startswith(".")
            ):
                if idx != 0 and tokens[idx - 1].text.startswith("в"):
                    tokens[idx + 1].text = "ве+ке"
                elif idx != 0 and tokens[idx - 1].text.startswith("из"):
                    tokens[idx + 1].text = "ве+ка"
                else:
                    tokens[idx + 1].text = "век"
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
