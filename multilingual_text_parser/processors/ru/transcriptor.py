from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.thirdparty.ru.russian_g2p.Transcription import Transcription

__all__ = ["TranscriptorRU"]


class TranscriptorRU(Transcription):
    def __init__(self):
        super().__init__()

    def transcribe(self, sent: Sentence):  # type: ignore
        text = sent.stress
        text = self._preprocessor.preprocessing(text)

        tmp = " " + " ".join(text)
        phonetic_words = tmp.split(" <sil>")

        result = []
        for phonetic_word in phonetic_words:
            if len(phonetic_word) == 0:
                continue

            words, ph_by_word, ph_by_phrase = self._g2p.phrase_to_phonemes(phonetic_word)

            ph_by_word = [tuple(ph) for ph in ph_by_word]
            ph_by_phrase = tuple(ph_by_phrase)
            result.append((words, ph_by_word, ph_by_phrase))

        return result

    def transcribe_word(self, token):
        text = token.stress
        if isinstance(text, list):
            text = text[0]
        phonemes = self._g2p.word_to_phonemes(text)
        return tuple(phonemes)
