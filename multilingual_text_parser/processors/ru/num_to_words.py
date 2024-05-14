import os
import sys
import typing as tp
import logging
import subprocess

from os import environ as env
from subprocess import Popen

from num2words import num2words

from multilingual_text_parser.thirdparty.ru.e2yo.e2yo.core import E2Yo
from multilingual_text_parser.utils.fs import get_root_dir
from multilingual_text_parser.utils.log_utils import trace
from multilingual_text_parser.utils.zmq_patterns import ZMQPatterns, find_free_port

__all__ = ["NumToWords"]

LOGGER = logging.getLogger("root")


class NumToWords:
    def __init__(self, morph):
        self._morph = morph
        self._e2yo = E2Yo()

        try:
            free_port = int(env.setdefault("CyrillerPort", str(find_free_port())))
            cyriller_path = get_root_dir() / "data/ru/cyriller/pyCyriller"
            if sys.platform == "win32":
                cyriller_path = cyriller_path.with_suffix(".exe")
            else:
                os.chmod(cyriller_path, 0o777)
                pass

            self._cyriller_proc = Popen([cyriller_path.as_posix(), str(free_port)])
            self._cyriller_client = ZMQPatterns.client(f"127.0.0.1:{free_port}")

        except Exception as e:
            LOGGER.error(trace(self, e))
            self._cyriller_client = None

    def __del__(self):
        if self._cyriller_client:
            self._cyriller_client.send_string("exit")
            try:
                try:
                    self._cyriller_proc.wait(timeout=3)
                finally:
                    env.pop("CyrillerPort", None)
                    self._cyriller_proc.kill()
            except Exception as e:
                pass

    @classmethod
    def _prepare(cls, digit: tp.Union[str, int, float]):
        prefix = None
        ending = None
        if isinstance(digit, str):
            if digit.startswith("+"):
                prefix = "плюс"
            if digit.startswith("-"):
                prefix = "минус"
            digit = digit.lstrip("-").lstrip("+")
            if "-" in digit:
                digit, ending = digit.split("-", 1)
        else:
            digit = str(digit)

        return prefix, digit, ending

    def transform(
        self,
        digit: tp.Union[str, int, float],
        case: tp.Optional[str] = None,
        gender: tp.Optional[str] = None,
        number: tp.Optional[str] = None,
        ordinal: bool = False,
        noun: tp.Optional[str] = None,
    ):
        prefix, digit, suffix = self._prepare(digit)
        digit_th = None

        if noun:
            noun_parse = self._morph.parse(noun)
        else:
            noun_parse = None

        if case is None:
            if noun:
                case = self._morph.get_case(noun_parse)
                noun = noun_parse.normal_form
            else:
                case = "им"
        else:
            if noun:
                if ordinal or suffix:
                    noun = self._morph.inflect(noun_parse, [case, number])
                else:
                    noun = noun_parse.normal_form

        if gender is None:
            if noun:
                gender = self._morph.get_gender(noun_parse)
            else:
                gender = "мр"

        if ordinal and digit.isdigit() and len(digit) >= 4 and digit.endswith("00"):
            digit_th = int(digit[:-3] + "000")
            digit = int(digit[-3:])

        if suffix:
            digit = f"{digit}:{'all'}:{gender}"
        else:
            digit = f"{digit}:{case}:{gender}"

        if number:
            digit = f"{digit}:{number}"

        if ordinal:
            digit = f"{digit}:ordinal"

        if noun:
            request = f"{digit}|{noun}"
        else:
            request = f"{digit}"

        if sys.platform == "win32":
            request = request.replace(".", ",")
        else:
            request = request.replace(",", ".")

        response = None

        if self._cyriller_client:
            response = self._cyriller_client.request_as_string(request)

        if response is None or response == "error":
            try:
                digit = digit.split(":", 1)[0]
                digit = digit.replace(",", "").replace(".", "")
                if "/" in digit:
                    a, b = digit.split("/")
                    a = num2words(a, ordinal, "ru")
                    b = num2words(b, ordinal, "ru") + (f" {noun}" if noun else "")
                    response = f"{a} {b}"
                else:
                    response = num2words(digit, ordinal, "ru") + (
                        f" {noun}" if noun else ""
                    )
            except Exception as e:
                LOGGER.error(e)
                response = ""
            suffix = None

        if suffix:
            resp_all = response.split(":")
            var_number = -1
            for idx, digit in enumerate(resp_all):
                number_split = digit.split(" ")
                last_number = number_split[-(2 if noun else 1)]
                if last_number.endswith(suffix):
                    var_number = idx
                    break

            if var_number < 0:
                idx = 0
                number_split = resp_all[idx].split(" ")
                last_number = number_split[-(2 if noun else 1)]
                lexeme = self._morph.parse(last_number).lexeme
                for item in lexeme:
                    if item.word.endswith(suffix):
                        var_number = idx
                        number_split[-(2 if noun else 1)] = item.word
                        resp_all[idx] = " ".join(number_split)
                        break

            if var_number >= 0:
                response = resp_all[var_number]
            else:
                response = resp_all[0] + (f" {suffix}" if suffix else "")

        if ordinal and noun and noun == noun_parse.normal_form:
            number_split = response.split(" ")
            last_number = number_split[-(2 if noun else 1)]
            case = self._morph.get_case(last_number)[:2]
            if number is None:
                number = self._morph.get_number(last_number)
            noun = self._morph.inflect(noun_parse, [case, number])
            number_split[-1] = noun
            response = " ".join(number_split)

        if prefix:
            response = f"{prefix} {response}"

        if digit_th:  # TODO: fix bug Cyriller
            response_th = self._cyriller_client.request_as_string(f"{digit_th}")
            response = f"{response_th} {response}"

        response = self._e2yo.replace(response)
        if "один" in response:
            response = response.replace("один", "оди+н")
        if "сорока" in response:
            response = response.replace("сорока", "сорока+")
        elif "сорок" in response:
            response = response.replace("сорок", "со+рок")
        elif "весьм" in response:
            response = response.replace("весьм", "восьм")
        return response
