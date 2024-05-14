import re

import torch

from transformers import AutoModelForTokenClassification, AutoTokenizer

from multilingual_text_parser._constants import PUNCTUATION
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir

__all__ = ["TaggerRU"]


class TaggerRU(BaseSentenceProcessor):
    GPU_CAPABLE: bool = True

    def __init__(self, device: str = "cpu"):
        self._device = device
        self.num_to_class = [
            "same",
            "date",
            "time",
            "roman",
            "ordinal",
            "digit",
            "fraction",
        ]

        model_dir = get_root_dir() / "data/ru/tagger"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir / "tagger")
        self.model.to(torch.float32).to(self._device).eval()

        self._num = re.compile(r"\d+")
        self._clear = re.compile(f"[^а-яёА-ЯЁ{PUNCTUATION}\\s\t\n\r]")

        self._ordinal_suffix = [
            "-й",
            "-е",
            "-го",
            "-ый",
            "-ого",
            "-ому",
            "-ым",
            "-ом",
            "-ая",
            "-ой",
            "-ую",
            "-ое",
            "-ые",
            "-ых",
            "-ым",
            "-ыми",
        ]

        # регулярки на разные форматы дат dmy/mdy/ymd...
        self._date = re.compile(
            r"^([0-2]?\d|3[01])[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d{3}|\d{2}))$|^(0?\d|1[0-2])[\./\-]([0-2]?\d|3[01])([\./\-](([0-2]?\d{3})|\d{2}))$|^([0-2]?\d{3}|\d{2})[\./\-](0?\d|1[0-2])([\./\-]([0-2]?\d|3[01]))$|^([0-2]?\d{3}|\d{2})[\./\-]([0-2]?\d|3[01])([\./\-](0?\d|1[0-2]))$"
        )

    @exception_handler
    def _process_sentence(self, sent, **kwargs):
        if self._clear.search(sent.text):
            tokens = [token.text for token in sent.tokens]
            tags = self.get_preds(tokens)
            for i, token in enumerate(sent.tokens):
                if token.interpret_as in self.num_to_class:
                    token.tag = token.interpret_as
                elif self._num.search(token.text) and tags[i] == "same":
                    token.tag = "digit"
                else:
                    token.tag = tags[i]

                if token.tag == "digit":
                    if any(token.text.endswith(s) for s in self._ordinal_suffix):
                        token.tag = "ordinal"
                    elif self._date.search(token.text):
                        token.tag = "date"

    def get_preds(self, tokens: list) -> list:
        with torch.inference_mode():
            tokenized = self.tokenizer(
                tokens,
                truncation=True,
                is_split_into_words=True,
                return_tensors="pt",
                max_length=512,
            )
            pred = self.model(
                tokenized["input_ids"].to(self._device),
                attention_mask=tokenized["attention_mask"].to(self._device),
            ).logits.argmax(dim=2)[0]

        word_ids = tokenized.word_ids()
        res = []
        prev = None
        for i, j in enumerate(word_ids):
            if j is not None and prev != j:
                res.append(pred[i].item())
            prev = j

        return list(map(lambda x: self.num_to_class[x], res))
