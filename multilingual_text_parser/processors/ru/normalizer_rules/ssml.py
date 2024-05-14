import typing as tp
import logging

from multilingual_text_parser.data_types import Token
from multilingual_text_parser.processors.ru.normalizer_utils import BaseRule

LOGGER = logging.getLogger("root")


class CharactersSSML(BaseRule):
    def __init__(self, morph, num2words):
        super().__init__(morph, num2words)
        self.vocab = {
            "(": "открытая круглая скобка",
            ")": "закрытая правая скобка",
            "@": "собака",
            "#": "решетка",
            "&": "амперсанд",
            "a": "э+й",
            "b": "би+",
            "c": "си+",
            "d": "ди+",
            "e": "и",
            "f": "э+ф",
            "g": "джи+",
            "h": "э+йч",
            "i": "а+й",
            "j": "дже+й",
            "k": "ке+й",
            "l": "э+ль",
            "m": "э+м",
            "n": "э+н",
            "o": "о+у",
            "p": "пи+",
            "q": "кю+",
            "r": "а+р",
            "s": "э+с",
            "t": "ти+",
            "u": "ю+",
            "v": "ви+",
            "w": "да+бл ю",
            "x": "э+кс",
            "y": "уа+й",
            "z": "зэ+т",
        }

    def process(self, idx: int, tokens: tp.List[Token], start: int = 0, end: int = 0):
        token = tokens[idx]
        if token.interpret_as in ["characters", "abbreviation"]:
            new_text = []
            if token.text.isdigit():
                tokens[idx].text = " ".join(
                    [self._num2words.transform(d) for d in token.text]
                )
            elif token.text.isalpha() and not token.stress:
                # если аббревиатуру можно прочитать как слово: ОПЕК, НАТО, АТО
                if token.interpret_as == "abbreviation" and len(token.text) >= 3:
                    is_vowel = token.text[0] in "ауоиэыяюеё"
                    for char in token.text[1:]:
                        if is_vowel and char in "ауоиэыяюеё":
                            break
                        elif not is_vowel and char not in "ауоиэыяюеё":
                            break
                        is_vowel = char in "ауоиэыяюеё"
                    else:
                        return

                for char in token.text:
                    if char in "бвгджзптшц":
                        new_text.append(f"{char}э")
                    elif char in "лмнрсф":
                        new_text.append(f"э{char}")
                    elif char in "кхщ":
                        new_text.append(f"{char}а")
                    elif char in "ч":
                        new_text.append(f"{char}е")
                    elif char in self.vocab:
                        new_text.append(self.vocab[char])
                    else:
                        new_text.append(f"{char}")
                if new_text[-1][-1] in "ауоиэыяюеё":
                    new_text.append("+")
                else:
                    if len(new_text[-1]) == 2:
                        new_text[-1] = f"{new_text[-1][0]}+{new_text[-1][1]}"
                    else:
                        new_text.insert(len(new_text) - 1, "+")
                tokens[idx].text = "".join(new_text)
            elif token.text in self.vocab:
                tokens[idx].text = self.vocab[token.text]
