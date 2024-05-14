from multilingual_text_parser.thirdparty.ru.russian_g2p.Grapheme2Phoneme import (
    Grapheme2Phoneme,
)
from multilingual_text_parser.thirdparty.ru.russian_g2p.Preprocessor import Preprocessor


class Transcription:
    def __init__(
        self,
        raise_exceptions: bool = False,
        batch_size: int = 64,
        verbose: bool = False,
    ):
        self._preprocessor = Preprocessor(batch_size=batch_size)
        self._g2p = Grapheme2Phoneme(exception_for_nonaccented=raise_exceptions)
        self.verbose = verbose

    def transcribe(self, texts: list):
        all_words_and_tags = self._preprocessor.preprocessing(texts)
        if self.verbose:
            print("All texts have been preprocessed...")
        n_texts = len(texts)
        n_data_parts = 100
        part_size = n_texts // n_data_parts
        while (part_size * n_data_parts) < n_texts:
            part_size += 1
        data_counter = 0
        part_counter = 0
        total_result = []
        for cur_words_and_tags in all_words_and_tags:
            if len(cur_words_and_tags) > 0:
                tmp = " ".join(cur_words_and_tags[0])
                tmp = " " + tmp
                phonetic_words = tmp.split(" <sil>")
                try:
                    result = []
                    for phonetic_word in phonetic_words:
                        if len(phonetic_word) != 0:
                            phonemes = self._g2p.phrase_to_phonemes(phonetic_word)

                            # ph_by_word = self._g2p.word_to_phonemes(phonetic_word.split()[0])
                            orig_words = phonetic_word.replace("+", "").split()
                            ratio = len(phonemes) / sum([len(w) for w in orig_words])
                            phonemes_by_word = []
                            a = b = 0
                            for idx, word in enumerate(orig_words):
                                b += int(len(word) * ratio)
                                if idx + 1 == len(orig_words):
                                    b = len(phonemes)
                                phonemes_by_word.append((word, phonemes[a:b]))
                                a = b
                            assert a == len(phonemes)
                            result.append(phonemes_by_word)
                except Exception:
                    result = []
            else:
                result = []
            total_result.append(result)
            data_counter += 1
            if (part_size > 0) and self.verbose:
                if (data_counter % part_size) == 0:
                    part_counter += 1
                    print(f"{part_counter}% of texts have been processed...")
        if (part_counter < n_data_parts) and self.verbose:
            print("100% of texts have been processed...")
        return total_result
