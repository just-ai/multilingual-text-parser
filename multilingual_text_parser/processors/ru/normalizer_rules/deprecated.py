import re
import typing as tp
import logging

from multilingual_text_parser.data_types import Token
from multilingual_text_parser.processors.ru.normalizer_utils import BaseRule, Utils
from multilingual_text_parser.utils.fs import get_root_dir

LOGGER = logging.getLogger("root")


class FractionRules(BaseRule):
    """Класс для обработки дробных чисел, которые написаны через "/" Не склоняет по
    падежам."""

    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]):
        if re.match(r"\d+/\d+", text):
            return text.split("/")
        elif re.match(r"№\d+/\d+", text):
            result = ["номер"]
            result.extend(re.findall(r"№(\d+)/(\d+)", text)[0])
            return result

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        _text = self.check_format(idx, _text, tokens)
        if _text:
            if _text[0] == "номер":
                num1 = self._num2words.transform(_text[1])
                num2 = self._num2words.transform(_text[2])
                tokens[idx].text = " ".join(["номер", num1, num2])
            elif _text[0] == "1":
                num1 = self._num2words.transform(_text[0], gender="жр")
                num2 = self._num2words.transform(
                    _text[1], case="им", number="ед", gender="жр", ordinal=True
                )
                tokens[idx].text = " ".join([num1, num2])
            else:
                num1 = self._num2words.transform(_text[0])
                num2 = self._num2words.transform(
                    _text[1], case="рд", number="мн", ordinal=True
                )
                tokens[idx].text = " ".join([num1, num2])


class YearPeriodRules(BaseRule):
    # в середине первого дня/месяца ит.д/ в начале 2010
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._triger_words_reduction = {"г": "год", "гг": "год"}

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]) -> str:
        if text.isdigit() and 900 < int(text) < 2500 and idx - 1 > 0:
            word_ingen = re.compile(
                r"(середин[еы]|начал[еа]|конц[еа]|половин[еы]|трети|уровн[юя])"
            )
            word_innom = re.compile(r"(середину|начало|конец|половину|треть)")
            if re.search(word_ingen, tokens[idx - 1].text):
                return "рд"
            elif re.search(word_innom, tokens[idx - 1].text):
                return "им"

        return ""

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        num_case = self.check_format(idx, _text, tokens)
        if num_case:
            tokens[idx].text = self._num2words.transform(
                _text, case=num_case, number="ед", ordinal=True
            )
            if (
                idx + 1 < len(tokens)
                and tokens[idx + 1].text in self._triger_words_reduction
            ):
                year = self._triger_words_reduction[tokens[idx + 1].text]
                tokens[idx + 1].text = self._morph.inflect(year, [num_case, "ед"])


class OneAndHalfRules(BaseRule):
    # для обработки фраз с дробью 1.5
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._triger_words = ["день", "неделя", "месяц", "год", "десятилетие", "век"]

    def check_format(self, idx: int, text: str, tokens: tp.List[Token]) -> bool:
        one_and_half = re.compile(r"1[,.]5")
        if (
            re.search(one_and_half, text)
            and idx + 1 < len(tokens)
            and tokens[idx + 1].lemma in self._triger_words
        ):
            return True

        return False

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        _text = tokens[idx].text
        if self.check_format(idx, _text, tokens):
            tokens[idx].text = "полтора"


class YearRules(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self._triger_words = {"год", "век"}
        self._triger_words_reduction = {
            "г": "год",
            "гг": "год",
            "в": "век",
            "вв": "век",
        }

        self._months = {
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

        self._triger_words_all = []
        for triger in self._triger_words:
            for word in self._morph.get_lexeme(triger, with_meta=False):
                if word in self._triger_words_reduction:
                    continue
                self._triger_words_all.append(word)

        self._triger_words_all += list(self._triger_words_reduction.keys())
        self._triger_words_all = set(self._triger_words_all)

    def check_format(self, text: str) -> bool:
        if text.isdigit() and 100 < int(text) < 3000:
            return True
        elif text.count("-") == 1:
            a, b = text.split("-", 1)
            return a.isdigit() and len(a) <= 4 and not b.isdigit()
        else:
            return False

    def _remove_point(self, idx, tokens):
        if idx + 1 < len(tokens):
            if tokens[idx + 1].text == ".":
                tokens.remove(tokens[idx + 1])
            elif tokens[idx + 1].text == "!":
                tokens[idx + 1].text = "."

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[idx]

        if idx > 0 and current_token.norm[:3] in self._months:
            if tokens[idx - 1].text.isdigit():
                date = f"{tokens[idx-1].text}.{self._months[current_token.text[:3]]}"
                current_token.text = date
                tokens.remove(tokens[idx - 1])
                return

        if idx > 0 and current_token.text in self._triger_words_all:
            if not self.check_format(tokens[idx - 1].text):
                return

            if current_token.text == "в" and not tokens[idx + 1].is_punctuation:
                return

            if current_token.text in self._triger_words_reduction:
                case = "им"
                number = "ед" if len(current_token.text) == 1 else "мн"
                triger_word = self._triger_words_reduction[current_token.text]
            else:
                if current_token.text == "лет":  # ?
                    case = "им"
                    number = "ед"
                else:
                    case = self._morph.get_case(current_token.text)
                    number = self._morph.get_number(current_token.text)
                triger_word = current_token.text

            prepositions = Utils.find_prepositions(tokens, idx)

            years: tp.List[tp.Dict[str, tp.Any]] = []
            for i in reversed(range(max(idx - 4, 0), idx)):
                word = tokens[i].text
                if tokens[i].is_punctuation and word != ",":
                    break

                if self.check_format(word):
                    years.append(
                        {"token": tokens[i], "case": case, "number": number, "pos": i}
                    )
                    if "-" in tokens[i].text or case == "рд":
                        years[-1]["ordinal"] = True

                if word in prepositions:
                    for year in reversed(years):
                        if "preposition" not in year and (
                            (year["pos"] - prepositions[word]["pos"] <= 2)
                            or (idx > 2 and tokens[idx - 2].norm == "и")
                        ):
                            year["preposition"] = prepositions[word]
                            if (
                                year["preposition"]["text"] == "к"
                                and triger_word == "года"
                            ):
                                if "-" not in year["token"].text:
                                    year["token"].text += "-ого"

                            if (
                                year["preposition"]["text"] == "в"
                                and triger_word == "году"
                            ):
                                if "-" not in year["token"].text:
                                    year["token"].text += (
                                        "-ом"
                                        if not year["token"].text.endswith("3")
                                        else "-ем"
                                    )

                            if (
                                year["preposition"]["text"] == "в"
                                and triger_word == "года"
                            ):
                                if "-" not in year["token"].text:
                                    year["token"].text += "-го"

                            year["case"] = prepositions[word]["case"]
                            if "ordinal" not in year and year["preposition"]["text"] in [
                                "от",
                                "около",
                            ]:
                                year["ordinal"] = False
                            else:
                                year["ordinal"] = True
                        else:
                            break

            for year in years:
                if "-е" in year["token"].text:
                    year["case"] = "им"

                if "-" not in year["token"].text and "preposition" in year:
                    if year["preposition"]["text"] in ["на"]:
                        if year["preposition"]["pos"] + 1 == year["pos"]:
                            year["case"] = "им"
                        else:
                            year["token"].text += "-ого" if number == "ед" else "-х"
                            year["case"] = "рд"
                    if year["preposition"]["text"] in ["с", "со"]:
                        year["token"].text += "-ого" if number == "ед" else "-х"
                    if year["preposition"]["text"] in ["в", "во"]:
                        year["token"].text += (
                            ("-ом" if not year["token"].text.endswith("3") else "-ем")
                            if number == "ед"
                            else "-х"
                        )
                if current_token.text == "лет":
                    year["ordinal"] = False

            if "-" not in years[-1]["token"].text:
                if "датирован" in tokens[idx - 2].text:
                    years[-1]["token"].text += "-м"
                    years[-1]["ordinal"] = True

                if tokens[idx - 2].norm[:3] in self._months:
                    years[-1]["token"].text += "-ого"

            for year in reversed(years):
                if (triger_word == "век" or year["number"] == "мн") and year[
                    "case"
                ] == "пр2":
                    year["case"] = "пр"

                year["token"].text = self._num2words.transform(
                    digit=year["token"].text,
                    case=year["case"],
                    number=year["number"],
                    ordinal=year.get("ordinal", False),
                    noun=triger_word
                    if triger_word != current_token.text and year == years[0]
                    else None,
                )

                if year["token"].text.startswith("одн"):
                    if year["token"].stress is not None:
                        year["token"].text = " ".join(year["token"].stress.split()[1:])
                    else:
                        year["token"].text = " ".join(year["token"].text.split()[1:])

                if year["number"] == "мн" and year["token"].text.endswith("года"):
                    if year["token"].stress is not None:
                        year["token"].text = year["token"].stress.replace("года", "годы")
                    else:
                        year["token"].text = year["token"].text.replace("года", "годы")

            if triger_word != current_token.text:
                self._remove_point(idx, tokens)
                tokens.remove(current_token)


class DateRules(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        vocab_root = get_root_dir() / "data/ru/vocabularies"
        self._months = Utils.read_vocab(vocab_root / "months.txt", as_list=True)
        self._months = dict(zip(range(1, len(self._months) + 1), self._months))

        self._modifiers = Utils.read_vocab(vocab_root / "num_modifiers.txt")

        self._triger_words = self._morph.get_lexeme("год", with_meta=False) + [
            "г",
            "гг",
        ]
        self._months_all_form = []
        for month in self._months.values():
            self._months_all_form += self._morph.get_lexeme(month, with_meta=False)
        self._months_all_form = set(self._months_all_form)

    def check_format(self, text: str, format: tp.Optional[str]):
        def split_wo_sep(text: str, format: str):
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

        d = m = y = "0"
        if text.count(".") == 2:
            if format == "mdy":
                m, d, y = text.split(".")
            elif format == "ymd":
                y, m, d = text.split(".")
            elif format == "ydm":
                y, d, m = text.split(".")
            else:
                d, m, y = text.split(".")

        if text.count(".") == 1:
            if format == "md":
                m, d = text.split(".")
            elif format == "ym":
                y, m = text.split(".")
            elif format == "my":
                m, y = text.split(".")
            else:
                d, m = text.split(".")

        if text.count(".") == 0:
            if format == "d":
                d = text
            elif format == "m":
                m = text
            elif format == "y":
                y = text
            elif format and len(format) == 3 and len(text) == 8:
                d, m, y = split_wo_sep(text, format)
            elif format and len(format) == 2 and len(text) == 6:
                d, m, y = split_wo_sep(text, format)
            elif format and len(format) == 2 and len(text) == 4:
                d, m, y = split_wo_sep(text, format)
            else:
                return

        if d.isdigit() and m.isdigit() and y.isdigit():
            di, mi, yi = int(d), int(m), int(y)
            if format:
                if 0 <= di <= 31 and 0 <= mi <= 12 and 0 <= yi <= 9999:
                    return di, mi, yi
            else:
                if 0 <= yi <= 9999:
                    if 1 <= di <= 31 and 1 <= mi <= 12:
                        return di, mi, yi
                    elif 1 <= mi <= 31 and 1 <= di <= 12:
                        return mi, di, yi

    def check_format2(self, text: str) -> bool:
        if text.isdigit() and len(text) <= 2:
            return True
        elif text.count("-") == 1:
            a, b = text.split("-", 1)
            return a.isdigit() and len(a) <= 2 and not b.isdigit()
        else:
            return False

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        if idx + 1 < len(tokens) and tokens[idx + 1].text in self._modifiers:
            return

        current_token = tokens[idx]
        date = self.check_format(current_token.text, current_token.format)

        case = "рд"
        gender = "мр"
        if len(tokens) == 2:
            case = "им"
            gender = "ср"

        prepositions = Utils.find_prepositions(tokens, idx, all=False)
        if prepositions:
            last_preposition = list(prepositions.items())[0]
            if last_preposition[1]["text"] in ["по", "на"]:
                case = "им"
                gender = "ср"

        if date:
            d, m, y = date

            date2str = []
            if d:
                date2str.append(
                    self._num2words.transform(d, case, gender=gender, ordinal=True)
                )
            if m:
                date2str.append(self._morph.inflect(self._months[m], ["рд"]))
            if y:
                date2str.append(
                    self._num2words.transform(y, "рд", gender=gender, ordinal=True)
                )
                date2str.append(self._morph.inflect("год", ["рд"]))

            current_token.text = " ".join(date2str)

            if idx + 1 < len(tokens) - 1:
                if tokens[idx + 1].text in self._triger_words:
                    tokens.remove(tokens[idx + 1])

        if (
            idx > 0
            and current_token.text in self._months_all_form
            and self.check_format2(tokens[idx - 1].text)
        ):
            case = self._morph.get_case(current_token.text)
            tokens[idx - 1].text = self._num2words.transform(
                tokens[idx - 1].text, case=case, gender=gender, ordinal=True
            )


class DigitRules(BaseRule):
    # TODO: issue with exaples like: "теряет около 1 <prosody>млрд долларов[47]</prosody>."
    # After normalization млрд -> миллиардов w/o modifiers
    _as_ordinal = {"место"}

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

    def check_format(self, text: str):
        prefix, suffix = None, None

        if text.count(".") > 1 or text.count(",") > 1:
            text = text.replace(".", "").replace(",", "")
            return prefix, text, suffix, False

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

        if not text:
            return False

        if "." in text:
            a, b = text.split(".", 1)
            if a.isdigit() and b.isdigit():
                return prefix, text, suffix, True

        if "," in text:
            a, b = text.split(",", 1)
            if a.isdigit() and b.isdigit():
                return prefix, text, suffix, True

        if text.isdigit():
            return prefix, text, suffix, False

        if text.count("-") == 1:
            a, b = text.split("-")
            if a.isdigit() and not b.isdigit():
                return prefix, text, suffix, False
            if not a.isdigit() and b.isdigit():
                if prefix is None:
                    return a, b, suffix, False

    def _remove_point(self, idx, tokens):
        if idx + 1 < len(tokens):
            if tokens[idx + 1].text == ".":
                tokens.remove(tokens[idx + 1])
            elif tokens[idx + 1].text == "!":
                tokens[idx + 1].text = "."

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        current_token = tokens[idx]
        ret = self.check_format(current_token.text)
        if ret:
            prefix, digit, suffix, is_float = ret
            case = "им"
            gender = "мр"
            ordinal = False
            ending = None

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
                    elif tokens[idx + 1].interpret_as not in [
                        "characters",
                        "abbreviation",
                    ]:
                        word_gender = self._morph.get_gender(tokens[idx + 1].text)
                        if word_gender:
                            gender = word_gender

                if "-" in digit.lstrip("-") and not tokens[idx + 1].text.startswith(
                    "лет"
                ):
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

                gender = "жр"
                d2str = self._num2words.transform(
                    digit, case=case, gender=gender, noun=suffix
                )
                d2str = d2str.replace("целые", "целых").replace("целыми", "целых")

                if safe_suffix:
                    d2str += f" {self._morph.inflect(safe_suffix, ['рд', 'мн'])}"
                    suffix = safe_suffix
            else:
                if current_token.interpret_as == "ordinal":
                    ordinal = True
                if current_token.interpret_as == "cardinal":
                    ordinal = False
                s = ""
                while digit.startswith("0") and len(digit) > 1:
                    s += "ноль "
                    digit = digit[1:]
                if len(digit) > 33:
                    d2str = " ".join([self._num2words.transform(d) for d in digit])
                else:
                    d2str = self._num2words.transform(
                        digit, case=case, gender=gender, ordinal=ordinal, noun=suffix
                    )
                if s:
                    d2str = s + d2str

            if (
                suffix
                and idx + 1 < len(tokens)
                and tokens[idx + 1].text in self._modifiers
            ):
                d2str = d2str.split()
                word = tokens[idx + 1].text
                if word in ["тыс", "млн", "млрд", "трлн"]:
                    word = self._modifiers[word]
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

            current_token.text = d2str
