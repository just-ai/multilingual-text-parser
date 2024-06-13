import re
import codecs
import warnings
import itertools

from multilingual_text_parser.thirdparty.ru.russian_g2p.modes.Phonetics import Phonetics
from multilingual_text_parser.thirdparty.ru.russian_g2p.RulesForGraphemes import (
    RulesForGraphemes,
)
from multilingual_text_parser.utils.fs import get_root_dir


class Grapheme2Phoneme(RulesForGraphemes):
    def __init__(self, users_mode="Modern", exception_for_nonaccented=False):
        RulesForGraphemes.__init__(self, users_mode)
        self.exception_for_nonaccented = exception_for_nonaccented

        self.vocals = Phonetics().vocals_phonemes

        self.__re_for_phrase_split = None

        self.__silence_name = "sil"

        # self.__function_words_1 = {'без', 'безо', 'близ', 'в', 'во', 'вне', 'для', 'до', 'за', 'из', 'изо', 'к', 'ко',
        #                           'меж', 'на', 'над', 'о', 'об', 'обо', 'от', 'ото', 'по', 'под', 'подо', 'пред',
        #                           'предо', 'перед', 'при', 'про', 'с', 'со', 'у', 'чрез', 'через', 'не', 'ни', 'из-за',
        #                           'из-подо', 'из-под', 'а-ля', 'по-над', 'по-за'}

        self.__function_words_1 = {
            "без",
            "близ",
            "в",
            "из",
            "меж",
            "над",
            "об",
            "под",
            "пред",
            "перед",
            "чрез",
            "через",
            "из-под",
            "по-над",
        }

        self.__function_words_2 = {
            "бы",
            "б",
            "де",
            "ли",
            "же",
            "-то",
            "-ка",
            "-либо",
            "-нибудь",
            "-таки",
        }

        self.__exclusions_dictionary = None
        data_dir = get_root_dir() / "data" / "ru" / "russian_g2p"
        exclusions_dictionary_name = data_dir / "data" / "Phonetic_Exclusions.txt"
        assert (
            exclusions_dictionary_name.is_file()
        ), f"File `{exclusions_dictionary_name}` does not exist!"
        exclusions_dictionary_name = str(exclusions_dictionary_name)

        self.__exclusions_dictionary = self.load_exclusions_dictionary(
            exclusions_dictionary_name
        )
        self.__re_for_phrase_split = re.compile(r"[\s\-]+", re.U)

    @property
    def russian_letters(self) -> list:
        return sorted(list(self.mode.all_russian_letters))

    @property
    def russian_phonemes(self) -> list:
        return sorted(list(self.mode.russian_phonemes_set))

    @property
    def silence_name(self) -> str:
        return self.__silence_name

    def load_exclusions_dictionary(self, file_name: str) -> dict:
        words_and_words = {}
        with codecs.open(
            file_name, mode="r", encoding="utf-8", errors="ignore"
        ) as dictionary_file:
            cur_line = dictionary_file.readline()
            cur_line_index = 1
            while len(cur_line):
                error_message = (
                    f"File `{file_name}`, line {cur_line_index}: description of this word and its transformation is "
                    "incorrect!"
                )
                prepared_line = cur_line.strip()
                if len(prepared_line):
                    words_of_line = prepared_line.lower().split()
                    nwords = len(words_of_line)
                    assert nwords == 2, error_message
                    word_original, word_transformed = words_of_line
                    assert any(
                        [
                            c in (self.mode.all_russian_letters | {"-", "+"})
                            for c in word_original
                        ]
                    ), error_message
                    assert any(
                        [
                            c in (self.mode.all_russian_letters | {"-", "+"})
                            for c in word_transformed
                        ]
                    ), error_message
                    assert (
                        len(word_original) > 0 and len(word_transformed) > 0
                    ), error_message
                    words_and_words[word_original] = word_transformed
                cur_line = dictionary_file.readline()
                cur_line_index += 1
        return words_and_words

    def check_word(self, checked_word: str):
        assert len(checked_word) > 0, "Checked word is empty string!"
        assert all(
            [
                c in (self.mode.all_russian_letters | {"+", "-"})
                for c in checked_word.lower()
            ]
        ), f"`{checked_word}`: this word contains inadmissible characters!"
        assert (
            len(
                list(
                    filter(
                        lambda c: c in self.mode.all_russian_letters,
                        checked_word.lower(),
                    )
                )
            )
            > 0
        ), f"`{checked_word}`: this word is incorrect!"

    def check_phrase(self, checked_phrase: str):
        assert len(checked_phrase) > 0, "Checked phrase is empty string!"
        assert all(
            [
                c in (self.mode.all_russian_letters | {" ", "+", "-"} | {"s", "i", "l"})
                for c in checked_phrase.lower()
            ]
        ), f"`{checked_phrase}`: this phrase contains inadmissible characters!"
        # for cur_word in self.__re_for_phrase_split.split(checked_phrase.lower()):
        # assert (len(list(filter(lambda c: c in self.all_russian_letters, cur_word))) > 0) \
        #      or (cur_word.lower() == 'sil'), f'`{checked_phrase}`: this phrase is incorrect!'

    def word_to_phonemes(self, source_word: str, next_phoneme: str = "sil") -> list:
        self.check_word(source_word)
        error_message = f"`{source_word}`: this word is incorrect!"
        prepared_word = source_word.lower()
        if prepared_word in self.__exclusions_dictionary:
            prepared_word = self.__exclusions_dictionary[prepared_word]
        if "+" not in prepared_word:
            counter = len(prepared_word) - len(re.sub(r"[аоуэыияёею]", "", prepared_word))
            if counter > 1:
                if self.exception_for_nonaccented:
                    raise ValueError(
                        f"`{source_word}`: the accent for this word is unknown!"
                    )
                # warnings.warn(f'`{source_word}`: the accent for this word is unknown!')
        if prepared_word in self.__exclusions_dictionary:
            prepared_word = self.__exclusions_dictionary[prepared_word]
        prepared_word = prepared_word.replace("'", "")
        if "-" in prepared_word:
            if (not self.in_function_words_1(prepared_word)) and (
                not self.in_function_words_2(prepared_word)
            ):
                word_parts = list(
                    filter(
                        lambda a: len(a) > 0,
                        map(lambda b: b.strip(), prepared_word.split("-")),
                    )
                )
                assert len(word_parts) > 0, error_message
                prepared_word_parts = [word_parts[0]]
                for cur_part in word_parts[1:]:
                    if self.in_function_words_1(
                        "-" + cur_part
                    ) or self.in_function_words_2("-" + cur_part):
                        prepared_word_parts.append("-" + cur_part)
                    else:
                        prepared_word_parts.append(cur_part)
                return self.phrase_to_phonemes(" ".join(prepared_word_parts))
            prepared_word = self.__remove_character(prepared_word, "-")
        letters_list = self.__word_to_letters_list(self.__prepare_word(prepared_word))
        n = len(letters_list)
        assert n > 0, error_message
        ind = n - 1
        # начинаем формировать транскрипцию
        transcription: list = []
        while ind >= 0:
            if ind >= 0 and letters_list[ind] in self.mode.hard_and_soft_signs:
                ind -= 1
                continue
            if letters_list[ind] in self.mode.vocals:
                new_phonemes = self.apply_rule_for_vocals(letters_list, ind)
            else:
                assert letters_list[ind] in self.mode.consonants, error_message
                new_phonemes = self.apply_rule_for_consonants(
                    letters_list, next_phoneme, ind
                )
            ind -= 1
            transcription = new_phonemes + transcription
            next_phoneme = new_phonemes[0]

        if len(transcription) == 0:
            print(f"`{source_word}`: this word cannot be transcribed!")
            return []
        return self.__remove_long_phonemes(
            self.__remove_repeats_from_transcription(transcription)
        )

    def phrase_to_phonemes(self, source_phrase: str) -> list:
        # error_message = f"`{source_phrase}`: this phrase is incorrect!"
        source_phrase = source_phrase.lower().replace("-", " ")
        self.check_phrase(source_phrase)
        words_in_phrase = source_phrase.split()
        num_words = len(words_in_phrase)
        for i in range(num_words):
            if words_in_phrase[i] in self.__exclusions_dictionary:
                words_in_phrase[i] = self.__exclusions_dictionary[words_in_phrase[i]]
            words_in_phrase[i] = self.__prepare_word(words_in_phrase[i])
        # формируем псевдослова, объединяя предлоги со стоящими после них словами
        new_words = []
        cur_word = ""
        last_letter = ""
        for i in range(0, num_words):
            clear_word = words_in_phrase[i].replace("+", "")
            to_append = (
                True  # (i == l - 1) or (clear_word not in self.__function_words_1)
            )
            if words_in_phrase[i][0] == "и":
                if last_letter not in (
                    self.mode.vocals | {"ь", ""} | self.mode.soft_consonants
                ):
                    words_in_phrase[i] = "ы" + words_in_phrase[i][1:]
            if words_in_phrase[i][0] in self.mode.double_vocals:
                words_in_phrase[i] = "ъ" + words_in_phrase[i]
            cur_word += words_in_phrase[i]
            if to_append:
                new_words.append(cur_word)
                cur_word = ""
            last_letter = clear_word[-1]
        # разбираем фразу
        next_phoneme = "sil"
        phrase_transcription: list = []
        remove_idx = []
        for i in range(len(new_words) - 1, -1, -1):
            new_transcription = self.word_to_phonemes(new_words[i], next_phoneme)
            if len(new_transcription) == 0:
                remove_idx.append(i)
                continue
            new_transcription = self.__remove_repeats_from_transcription(
                new_transcription
            )
            new_transcription = self.__remove_long_phonemes(new_transcription)
            phrase_transcription = [new_transcription] + phrase_transcription
            next_phoneme = new_transcription[0]
        for idx, id in enumerate(remove_idx):
            del new_words[id - idx]
        final_transcription = list(itertools.chain.from_iterable(phrase_transcription))
        final_transcription = self.__remove_repeats_from_transcription(
            final_transcription, False, False
        )
        final_transcription = self.__remove_long_phonemes(final_transcription)
        return [new_words, phrase_transcription, final_transcription]

    def in_function_words_1(self, source_word: str) -> bool:
        return (
            self.__remove_character(source_word, "+").lower() in self.__function_words_1
        )

    def in_function_words_2(self, source_word: str) -> bool:
        return (
            self.__remove_character(source_word, "+").lower() in self.__function_words_2
        )

    def __remove_character(self, source_word: str, removed_char: str) -> str:
        return "".join(list(filter(lambda a: a != removed_char, source_word.lower())))

    def __prepare_word(self, cur_word: str) -> str:
        prepared_word = cur_word.lower().strip()
        replace_pairs = [
            ("стн", "сн"),
            ("стл", "сл"),
            ("нтг", "нг"),
            ("здн", "зн"),
            ("здц", "зц"),
            ("ндц", "нц"),
            ("рдц", "рц"),
            ("ндш", "нш"),
            ("гдт", "гт"),
            ("лнц", "нц"),
        ]
        if (
            (len(prepared_word) > 2 and prepared_word[-3:] == "его")
            or (len(prepared_word) > 3 and prepared_word[-3:] == "ого")
            or (len(prepared_word) > 3 and prepared_word[-4:] in {"о+го", "е+го"})
        ):
            if prepared_word.replace("+", "") not in [
                "лего",
                "дорого",
            ]:  # TODO: put in exclusion dictionary
                prepared_word = prepared_word[:-2] + "ва"
        elif len(prepared_word) > 2 and prepared_word[-3:] == "тся":
            prepared_word = prepared_word[:-3] + "ца"
        elif len(prepared_word) > 3 and prepared_word[-4:] == "ться":
            prepared_word = prepared_word[:-4] + "ца"
        for repl_from, repl_to in replace_pairs:
            prepared_word = prepared_word.replace(repl_from, repl_to)
        return prepared_word

    def __word_to_letters_list(self, cur_word: str) -> list:
        vocal_letters = set(
            filter(lambda letter: not letter.endswith("+"), self.mode.vocals)
        )
        error_message = f"`{cur_word}`: this word is incorrect!"
        letters_list = []
        new_letter = ""
        for ind in range(len(cur_word)):
            if cur_word[ind] == "+":
                try:
                    assert new_letter in vocal_letters, error_message
                    new_letter += cur_word[ind]
                except Exception:
                    warnings.warn(error_message)
            else:
                assert cur_word[ind] in self.mode.all_russian_letters, error_message
                if len(new_letter):
                    letters_list.append(new_letter)
                new_letter = cur_word[ind]
        if len(new_letter):
            letters_list.append(new_letter)
        del vocal_letters
        return letters_list

    def __remove_repeats_from_transcription(
        self,
        source_transcription: list,
        reduce_equal: bool = True,  # True
        reduce_sibilant: bool = True,
    ) -> list:
        def equal(s_l: str, s_r: str) -> bool:
            s_l_ = re.sub(r"[l]", "", s_l)
            s_r_ = re.sub(r"[l]", "", s_r)
            ans = s_l_ == s_r_
            return ans

        def equal_almost(s_l: str, s_r: str) -> bool:
            s_l_ = re.sub(r"[l]", "", s_l)
            s_r_ = re.sub(r"[0l]", "", s_r)
            ans = s_l_ == s_r_
            return ans

        phomene_pairs = {
            ("Z", "ZH"): "ZH",
            ("Z", "ZH0"): "ZH0",
            ("Z0", "ZH0"): "ZH0",
            ("D", "Z"): "DZ",
            ("D", "DZ"): "DZ",
            ("D", "Z0"): "DZ0",
            ("D0", "Z0"): "DZ0",
            ("D", "DZ0"): "DZ0",
            ("D0", "DZ0"): "DZ0",
            ("D", "ZH"): "DZH",
            ("D", "DZH"): "DZH",
            ("D", "ZH0"): "DZ0",
            ("D0", "ZH0"): "DZH0",
            ("D", "DZH0"): "DZH0",
            ("D0", "DZH0"): "DZH0",
            ("T", "S"): "TS",
            ("T", "TS"): "TS",
            ("T", "S0"): "TS0",
            ("T0", "S0"): "TS0",
            ("T", "TS0"): "TS0",
            ("T0", "TS0"): "TS0",
            ("T", "SH"): "TSH",
            ("T", "TSH"): "TSH",
            ("T", "SH0"): "TSH0",
            ("T0", "SH0"): "TSH0",
            ("T", "TSH0"): "TSH0",
            ("T0", "TSH0"): "TSH0",
            ("S", "SH"): "SH",
            ("S", "TSH0"): "SH0",
            ("SH", "TSH0"): "SH0",
        }

        def conjugate(s_l: str, s_r: str) -> str:
            s_l_ = re.sub(r"[l]", "", s_l)
            s_r_ = re.sub(r"[l]", "", s_r)
            ans = phomene_pairs[(s_l_, s_r_)] if (s_l_, s_r_) in phomene_pairs else ""
            return ans

        prepared_transcription: list = []
        previous_phoneme = ""
        for current_phoneme in source_transcription:
            if reduce_equal and equal(previous_phoneme, current_phoneme):
                #  1st case: S0 S0 -> S0l
                prepared_transcription[-1] = current_phoneme + "l"
            elif reduce_equal and equal_almost(previous_phoneme, current_phoneme):
                #  2nd case: S S0 -> S0l
                prepared_transcription[-1] = current_phoneme + "l"
            elif reduce_sibilant:
                conj = conjugate(previous_phoneme, current_phoneme)
                if len(conj) > 0:
                    #  3rd case: S SH -> SHl
                    prepared_transcription[-1] = conj
                else:
                    prepared_transcription.append(current_phoneme)
            else:
                prepared_transcription.append(current_phoneme)

            previous_phoneme = prepared_transcription[-1]

        return prepared_transcription

    def __remove_long_phonemes(self, source_transcription: list) -> list:
        def postprocess_phoneme(src):
            if len(src) > 1 and src.endswith("l") and src not in self.vocals:
                return src[:-1]
            return src

        new_transcription = [postprocess_phoneme(ph) for ph in source_transcription]
        return list(filter(lambda it: len(it) > 0, new_transcription))
