import re
import sys
import base64
import logging

from pathlib import Path

import torch

from num2words import num2words
from transformers import AutoTokenizer

from multilingual_text_parser._constants import PUNCTUATION
from multilingual_text_parser.data_types import Sentence, Token
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.processors.common import Corrector
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir

__all__ = ["NormalizerEN"]

LOGGER = logging.getLogger("root")


class NormalizerEN(BaseSentenceProcessor):
    GPU_CAPABLE: bool = True

    def __init__(self, device: str = "cpu"):
        if sys.platform == "win32":
            LOGGER.warning("NeMo NLP not support on Windows platform!")

        self._device = device
        self._classes = ["cardinal", "digit", "rcardinal", "rordinal", "plain", "same"]
        self._invalid_symbols = re.compile(f"[^a-zA-Z{PUNCTUATION} ]")

        vocab_root = get_root_dir() / "data/en/vocabularies"
        self.dict_orig = self.read_vocabs(vocab_root / "abbreviations_orig.txt")
        self.dict_norm = self.read_vocabs(vocab_root / "abbreviations_norm.txt")
        self.dict_point = self.read_vocabs(vocab_root / "abbreviations_point.txt")

        self._tokenizer_model_path = get_root_dir() / "data/en/tokenizer/albert-base-v2"
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_model_path)

        self._tagger_model_path = get_root_dir() / "data/en/tagger/tagger_full.bin"
        self._tagger = torch.load(self._tagger_model_path, map_location="cpu")
        self._tagger.to(self._device).eval()

        self._normalizer = None

        self._sub_patterns = [
            (re.compile(r"(([\d,]*\d+) (st|rd|th|nd)(?![a-z]))"), self.parse_ordinal),
            (re.compile(r"((\$|€|¥|£)\s?([\d\.,]+)(m|bn|trn)?)"), self.parse_currency),
            (re.compile(r"(\+|\-)?([\d\.,]+)(°(f|c)?)"), self.parse_degrees),
            (re.compile(r"[\+%№#]"), self.parse_symbols),
            (re.compile(r"\s\/\s"), " per "),
        ]
        self._add_space = re.compile(f"([{PUNCTUATION}])")

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        if self._normalizer is None and sys.platform != "win32":
            from nemo_text_processing.text_normalization.normalize import Normalizer

            self._normalizer = Normalizer(input_case="cased", lang="en")

        old = sent.tokens
        tokens_orig = [token.text for token in old]
        text = " ".join(tokens_orig)

        text = self.preprocessing(text, sent.text_orig)
        if not re.match("^[a-zA-Z -,.!?()]+$", text) or re.search(
            r"\b[IVXLCDM]+\b", sent.text_orig
        ):
            text = self.tagging(text)

        if self._normalizer is not None:
            text = self._normalizer.normalize(text, verbose=False).lower()

        text = re.sub(
            f"(one) ({'|'.join(list(self.dict_norm.values()))})", self.one_measure, text
        )
        text = text.replace(" - ", "ЪХЪ")
        text = re.sub("[-/]", " ", Corrector.trim_punctuation(text))
        text = re.sub(r"(\w+)( \' )(\w+)", r"\1'\3", text)
        text = text.replace("ЪХЪ", " - ")

        text_orig = " " + self._add_space.sub(r" \1 ", sent.text_orig) + " "
        text = " " + self._add_space.sub(r" \1 ", text) + " "

        for token in tokens_orig:
            if (
                re.match("[a-z]+", token)
                and len(token) <= 3
                and token not in ["the"]
                and f" {token.upper()} " in text_orig
            ):
                text = text.replace(f" {token} ", f" {' '.join(token)} ")

        tokens = text.strip().replace("  ", " ").split()

        new = []
        j = 0
        i = 0
        while j < len(tokens) and i < len(old):
            if tokens[j] == old[i].text:
                new.append(old[i])
                j += 1
                i += 1
            elif i == 0:
                new.append(Token(tokens[j]))
                j += 1
                i += 1
            elif tokens[j - 1] == old[i - 1].text:
                new.append(Token(tokens[j]))
                j += 1
                i += 1
            elif old[i].text not in tokens:
                k = i + 1
                while old[k].text not in tokens:
                    k += 1
                while j < len(tokens) and old[k].text != tokens[j]:
                    new.append(Token(tokens[j]))
                    j += 1
                i = k
            else:
                while j < len(tokens) and tokens[j] != old[i].text:
                    new.append(Token(tokens[j]))
                    j += 1
                if j < len(tokens):
                    if tokens[j] == old[i].text:
                        j += 1
                        new.append(old[i])
                else:
                    new.append(old[i])
                    break
                i += 1

        sent.tokens = new

    @staticmethod
    def one_measure(tok):
        return f"{tok[1]} {tok[2][:-1]}"

    @staticmethod
    def read_vocabs(path: Path):
        if not path.exists():
            path = path.with_suffix(".bin")
        if not path.exists():
            raise FileNotFoundError(f"File {path.as_posix()} not found!")
        if path.suffix == ".bin":
            file = base64.b64decode(path.read_bytes()).decode()
        else:
            file = path.read_text(encoding="utf-8")

        d = {}
        for line in file.split("\n"):
            if not line:
                continue
            before, after = line.split(" = ")
            d[before] = after

        return d

    def preprocessing(self, text, text_orig):
        for pattern, replasment in self._sub_patterns:
            text = pattern.sub(replasment, text)
        for abb in self.dict_norm:
            text = re.sub(rf"\b{abb}\b", self.dict_norm[abb], text)
        for abb in self.dict_orig:
            if re.search(rf"{abb}", text_orig):
                text = re.sub(rf"\b{abb.lower()}\b", self.dict_orig[abb], text)
        for abb in self.dict_point:
            if re.search(rf"{abb}\.", text_orig):
                text = re.sub(rf"\b{abb.lower()}\b", self.dict_point[abb], text)
        return text

    def tagging(self, text, max_length: int = 128):
        tokens = []
        tokens_before = text.split()
        for i, t in enumerate(tokens_before):
            if re.match(r"^\d+[\-]\d+$", t):
                b, m, e = re.findall(r"^(\d+)([\-])(\d+)$", t)[0]
                tokens.extend([b, m, e])
            elif t == "." and i < len(tokens_before) - 1:
                continue
            else:
                tokens.append(t)

        tokenized_inputs = self._tokenizer(
            tokens,
            return_tensors="pt",
            is_split_into_words=True,
        ).to(self._device)
        res = self._tagger(**tokenized_inputs).logits.argmax(-1)

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        labels = []
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx != previous_word_idx:
                    labels.append(self._classes[res[0][i]])
                previous_word_idx = word_idx

        tokens_processed = []
        for i, tok in enumerate(tokens):
            try:
                cl = labels[i]
            except Exception as e:
                if max_length < 256:
                    return self.tagging(text, int(max_length * 1.25))

            if cl == "cardinal":
                tok = self.parse_cardinal(tok)
            elif cl == "digit":
                tok = self.parse_digit(tok)
            elif cl == "rordinal":
                tok = self.parse_rordinal(tok)
            elif cl == "rcardinal":
                tok = self.parse_rcardinal(tok)
            elif cl == "plain" and tok in [":", "-"]:
                tok = "to"
            tokens_processed.append(tok)
        return " ".join(tokens_processed)

    @staticmethod
    def parse_currency(tok):
        di = {
            "$": ["dollar", "dollars", "cent", "cents"],
            "¥": ["yen", "yen", "sen", "sen"],
            "£": ["pound", "pounds", "penny", "pence"],
            "₽": ["ruble", "rubles", "kopeck", "kopecks"],
            "m": 1000000,
            "bn": 1000000000,
            "trn": 1000000000000,
        }
        cur, num = tok[2], tok[3].replace(",", "")
        if tok[4]:
            num = str(di[tok[4]] * float(num))
        num = num2words(num, to="currency").replace("-", " ").replace(",", "")
        num = re.sub(" zero cents", "", num)
        num = re.sub("zero euro ", "", num)
        if not re.search("one euro", num):
            num = re.sub("euro", "euros", num)
        if cur == "€":
            return num
        else:
            num = re.sub("euros", di[cur][1], num)
            num = re.sub("euro", di[cur][0], num)
            num = re.sub("cents", di[cur][3], num)
            num = re.sub("cent", di[cur][2], num)
        return num

    @staticmethod
    def parse_ordinal(tok):
        num = tok[2]
        num = num.replace(",", "")
        num = num2words(num, to="ordinal").replace("-", " ").replace(",", "")
        return num

    @staticmethod
    def parse_degrees(tok):
        di = {
            "°": ["degree", "degrees"],
            "°f": ["degree fahrenheit", "degrees fahrenheit"],
            "°c": ["degree celsius", "degrees celsius"],
            "+": "plus",
            "-": "minus",
        }
        sign = None
        if tok[1]:
            sign = di[tok[1]]
        num = num2words(tok[2].replace(",", "")).replace("-", " ").replace(",", "")
        if tok[1] == "1":
            word = re.sub(tok[3], di[tok[3]][0], tok[3])
        else:
            word = re.sub(tok[3], di[tok[3]][1], tok[3])
        if sign:
            return f"{sign} {num} {word}"
        else:
            return f"{num} {word}"

    @staticmethod
    def parse_symbols(tok):
        di = {r"\+": " plus ", "%": " percent ", "№": " number ", "#": " number "}
        tok = tok[0]
        if tok == "+":
            return re.sub(f"\\{tok}", di[f"\\{tok}"], tok)
        else:
            return re.sub(tok, di[tok], tok)

    @staticmethod
    def parse_cardinal(tok):
        num = tok.replace(",", "")
        num = re.sub("^[−-]", "minus ", num)
        ans = re.findall(r"([^\d]*)(\d*\.?\d+)(.*)", num)
        if not ans:
            return num
        ans = (
            ans[0][0]
            + num2words(ans[0][1]).replace("-", " ").replace(",", "")
            + ans[0][2]
        )
        return ans

    @staticmethod
    def parse_digit(num):
        arr = [
            "o",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        begining = ""
        while num.startswith("0"):
            begining += f"{arr[0]} "
            num = num[1:]
        if not num:
            return begining[:-1]
        try:
            n = int("".join(re.findall(r"\d+", num)))
            ans = ""
            while n:
                ans = arr[n % 10] + " " + ans
                n = int(n / 10)
            if begining:
                ans = begining + ans
            return ans[:-1]
        except Exception:
            return num

    @staticmethod
    def roman_to_int(s):
        roman = {
            "i": 1,
            "v": 5,
            "x": 10,
            "l": 50,
            "c": 100,
            "d": 500,
            "m": 1000,
            "iv": 4,
            "ix": 9,
            "xl": 40,
            "xc": 90,
            "cd": 400,
            "cm": 900,
        }
        i = 0
        num = 0
        while i < len(s):
            if i + 1 < len(s) and s[i : i + 2] in roman:
                num += roman[s[i : i + 2]]
                i += 2
            else:
                num += roman[s[i]]
                i += 1
        return num

    def parse_rordinal(self, tok):
        if re.match("^[ivxlcdm]+$", tok):
            num = self.roman_to_int(tok)
            return num2words(num, to="ordinal").replace("-", " ").replace(",", "")
        else:
            return tok

    def parse_rcardinal(self, tok):
        if re.match("^[ivxlcdm]+$", tok):
            num = self.roman_to_int(tok)
            return num2words(num).replace("-", " ").replace(",", "")
        else:
            return tok


if __name__ == "__main__":
    from multilingual_text_parser.data_types import Text
    from multilingual_text_parser.parser import TextParser
    from multilingual_text_parser.utils.profiler import Profiler

    parser = TextParser(lang="EN", device="cuda", with_profiler=True)

    text = Text(
        """
    I am Elizabeth II. Only 3-5 years old children are allowed. E.g. i have $100000!!
    You’ve got only 10% of it. I am the 1st, you are the 2nd.
    159 W. Popplar Ave., Ste. 5, St. George, CA 12345 Only 10,000 people live in this city.
    It's −3°C today. It's +30°C outside. Call me on 8-900-555-55-55.
    He drives up to 40 km/h. He works in bros Inc. The World War II took place in 1939-1945.
    """
    )

    norm = parser.process(text)
    print(norm.text)

    with Profiler():
        parser.process(text)
