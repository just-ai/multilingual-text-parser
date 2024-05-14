import logging
import itertools

import torch
import stanza

from multilingual_text_parser.data_types import Doc, Position, Sentence, Syntagma
from multilingual_text_parser.processors.base import (
    BaseSentenceProcessor,
    BaseTextProcessor,
)
from multilingual_text_parser.utils import lang_supported as stanza_utils
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir
from multilingual_text_parser.utils.log_utils import trace

__all__ = ["SyntaxAnalyzer", "BatchSyntaxAnalyzer"]

LOGGER = logging.getLogger("root")


class SyntaxAnalyzer(BaseSentenceProcessor):
    GPU_CAPABLE: bool = True
    MULTILANG: bool = True

    def __init__(self, lang: str = "EN", device: str = "cpu"):
        self._device = device
        if "cuda:" in device and device.replace("cuda:", "").isdigit():
            torch.cuda.set_device(int(device.replace("cuda:", "")))

        if lang == "PT-BR":
            stanza_lang = "pt"
        else:
            stanza_lang = lang.lower()

        if lang.lower() in stanza_utils.stanza_available_languages():
            """custom model dir
            stanza_model_dir = get_root_dir() / f"data/{lang.lower()}/stanza_resources"
            if not stanza_model_dir.exists():
                stanza_model_dir.mkdir(parents=True)
                LOGGER.info(f"Download Stanza models for {lang} language")
                stanza.download(lang=lang.lower(), model_dir=stanza_model_dir.as_posix())
            """
            self._nlp = stanza.Pipeline(
                stanza_lang,
                processors="tokenize,pos,lemma,depparse",
                tokenize_pretokenized=True,
                use_gpu=not device == "cpu",
                download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            )
        else:
            self._nlp = None

    def _preprocessing_sentence(self, sent: Sentence):
        tokens_orig = [token.text for token in sent._tokens]

        tokens_upper = []
        for token in tokens_orig:
            if token.upper() in sent.text_orig:
                tokens_upper.append(token.upper())
            elif token.capitalize() in sent.text_orig:
                tokens_upper.append(token.capitalize())
            else:
                tokens_upper.append(token)

        return tokens_upper

    def _postprocessing_sentence(self, sent: Sentence):
        syntagmas = []
        synt_tokens = []
        is_word = False
        is_punctuation = False
        for idx, token in enumerate(sent.tokens):
            synt_tokens.append(token)
            if not is_word and not token.is_punctuation:
                is_word = True
            if is_word and token.is_punctuation:
                is_punctuation = True
            if is_punctuation and (
                token == sent.tokens[-1] or not sent.tokens[idx + 1].is_punctuation
            ):
                syntagmas.append(Syntagma(synt_tokens))
                synt_tokens = []
                is_punctuation = False
                if token == sent.tokens[-1]:
                    break
        else:
            syntagmas.append(Syntagma(synt_tokens))

        if len(syntagmas) > 0:
            syntagmas[0].position = Position.first
        if len(syntagmas) > 1:
            syntagmas[-1].position = Position.last

        sent.syntagmas = syntagmas

    def _run_nlp(self, tokens_per_sent):
        result = None
        if self._nlp:
            try:
                if self._device.startswith("cuda:") and torch.cuda.is_available():
                    with torch.cuda.device(self._device):
                        result = self._nlp(tokens_per_sent)
                else:
                    result = self._nlp(tokens_per_sent)
            except Exception as e:
                LOGGER.error(trace(self, e))

        return result

    def _apply_result(self, sent: Sentence, predict_tokens):
        for i in range(len(sent.tokens)):
            if sent.tokens[i].is_punctuation:
                sent.tokens[i].rel = None
                sent.tokens[i].head_id = None
                sent.tokens[i].id = None
                sent.tokens[i].pos = "PUNCT"
            else:
                sent.tokens[i].rel = predict_tokens[i].deprel
                sent.tokens[i].head_id = f"1_{predict_tokens[i].head}"
                sent.tokens[i].id = f"1_{predict_tokens[i].id}"
                sent.tokens[i].pos = predict_tokens[i].pos
                sent.tokens[i].feats = (
                    {
                        token.split("=")[0]: token.split("=")[1]
                        for token in predict_tokens[i].feats.split("|")
                    }
                    if predict_tokens[i].feats
                    else None
                )

    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        tokens = self._preprocessing_sentence(sent)
        result = self._run_nlp([tokens])

        if result is not None:
            self._apply_result(sent, result.sentences[0].words)
        else:
            for token_i in sent.tokens:
                token_i.rel = "x"
                token_i.head_id = "1_0"
                token_i.id = "1_0"
                token_i.pos = "X"

        self._postprocessing_sentence(sent)


class BatchSyntaxAnalyzer(BaseTextProcessor):
    GPU_CAPABLE: bool = True
    MULTILANG: bool = True

    def __init__(self, lang: str = "EN", device: str = "cpu", batch_size: int = 16):
        self._syntax_analyzer = SyntaxAnalyzer(lang, device)
        self._batch_size = batch_size

    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        list_tokens = []
        for sent in doc.sents:
            tokens_upper = self._syntax_analyzer._preprocessing_sentence(sent)
            list_tokens.append(tokens_upper)

        predict = []
        iterator = iter(list_tokens)
        while True:
            batch = list(itertools.islice(iterator, self._batch_size))
            if len(batch) > 0:
                result = self._syntax_analyzer._run_nlp(batch)
                predict += result.sentences
            else:
                break

        assert len(doc.sents) == len(predict)

        for sent, predict_tokens in zip(doc.sents, predict):
            try:
                self._syntax_analyzer._apply_result(sent, predict_tokens.words)
            except:
                for token_i in sent.tokens:
                    token_i.rel = "x"
                    token_i.head_id = "1_0"
                    token_i.id = "1_0"
                    token_i.pos = "X"

            self._syntax_analyzer._postprocessing_sentence(sent)
