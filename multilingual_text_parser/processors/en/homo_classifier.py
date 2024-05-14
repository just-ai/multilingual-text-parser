import os
import re
import json

import numpy as np
import torch

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir
from multilingual_text_parser.utils.model_loaders import load_transformer_model
from multilingual_text_parser.utils.profiler import Profiler

__all__ = ["HomographerEN"]


class HomographerEN(BaseSentenceProcessor):
    GPU_CAPABLE: bool = True

    def __init__(self, device: str = "cpu", window=10):
        import xgboost as xgb

        self.window = window
        self._device = device
        if not device == "cpu" and device.replace("cuda:", "").isdigit():
            torch.cuda.set_device(int(device.replace("cuda:", "")))

        self._lang_model_dir = get_root_dir() / "data/en/homo_classifier/albert-base-v2"
        self.model = load_transformer_model(
            self._lang_model_dir, output_hidden_states=True
        )
        self.model.to(self._device).eval()

        self._tokenizer_model_path = get_root_dir() / "data/en/tokenizer/albert-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_model_path,
            use_fast=True,
            add_prefix_space=True,
        )

        self.phonemes_name = get_root_dir() / "data/en/homo_classifier/phonemes.json"
        self.phonemes = json.loads(self.phonemes_name.read_text(encoding="utf-8"))

        self.dictionary_name = get_root_dir() / "data/en/homo_classifier/dictionary.json"
        try:
            self.dict = json.loads(self.dictionary_name.read_text(encoding="utf-8"))
        except json.decoder.JSONDecodeError:
            self.dict = {}

        self.classifiers_dir = get_root_dir() / "data/en/homo_classifier/classifiers"
        self.dict_clf = {}
        for homo in self.dict:
            file = self.dict[homo]["classifier"]
            clf = xgb.XGBClassifier()
            clf.load_model(os.path.join(self.classifiers_dir, file))
            self.dict_clf[homo] = clf

    @exception_handler
    def _process_sentence(self, sent, **kwargs):
        sent_text = sent.text
        for w in self.phonemes.keys():
            if re.search(w, sent_text):
                for tok_id, token in enumerate(sent.tokens):
                    if not token.stress:
                        if re.search(f"^{w}$", token.text):
                            if self.window:
                                a = max(tok_id - self.window, 0)
                                b = min(tok_id + self.window, len(sent.tokens))
                                context = [t.text for t in sent.tokens[a:b]]
                                answer = self.inference(context, w, tok_id - a)
                                token.phonemes = answer
                            else:
                                answer = self.inference(
                                    [t.text for t in sent.tokens],
                                    w,
                                    token.text,
                                    tok_id,
                                )
                                token.phonemes = answer

    def inference(self, context, regex, tok_id):
        clf = self.dict_clf[regex]
        emb = self._get_emb(context, tok_id)
        tres = self.dict[regex]["threshold"]
        cl = int(clf.predict_proba(emb.reshape(1, -1))[:, 1] >= tres)
        pos = self.dict[regex]["homographs"][cl]
        return self.phonemes[regex][pos]

    def validate_all(self):
        filename = get_root_dir() / "data/en/homo_classifier/results.txt"
        with open(filename, "w", encoding="utf-8") as file:
            all_y = np.array([])
            all_preds = np.array([])
            for homo in tqdm(self.corpus.keys()):
                if homo not in self.dict_clf:
                    continue
                var1, var2 = self.corpus[homo].keys()
                emb_1 = self._get_embs(self.corpus[homo][var1], homo)
                emb_2 = self._get_embs(self.corpus[homo][var2], homo)
                clf = self.dict_clf[homo]
                tres = self.dict[homo]["threshold"]
                y = np.concatenate(
                    (np.zeros(len(emb_1)), np.ones(len(emb_2))), axis=None
                ).astype(int)
                X = np.concatenate((emb_1, emb_2), axis=0)
                pred_test_proba = clf.predict_proba(X)[:, 1]
                file.write(homo + "\n")
                file.write(
                    metrics.classification_report(
                        y, pred_test_proba >= tres, target_names=[var1, var2]
                    )
                    + "\n\n"
                )
                all_y = np.append(all_y, y)
                all_preds = np.append(all_preds, (pred_test_proba >= tres).astype(int))
            file.write(metrics.classification_report(all_y, all_preds) + "\n\n")

    def add_homograph(
        self, homo, in_corpus=True, var1=None, var2=None, texts1=None, texts2=None
    ):
        if in_corpus:
            var1, var2 = self.corpus[homo]
            texts1 = self.corpus[homo][var1]
            texts2 = self.corpus[homo][var2]
        else:
            self.corpus[homo] = {var1: texts1, var2: texts2}
            with open(self.corpus_name, "w", encoding="utf-8") as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=4)

        self.train(texts1, texts2, homo, var1, var2)

    def train(self, corpus_1, corpus_2, re_expr, name_1, name_2):
        import xgboost as xgb

        emb_1 = self._get_embs(corpus_1, re_expr)
        emb_2 = self._get_embs(corpus_2, re_expr)

        y = np.concatenate((np.zeros(len(emb_1)), np.ones(len(emb_2))), axis=None).astype(
            int
        )
        X = np.concatenate((emb_1, emb_2), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        param_grid = {"max_depth": range(3, 10, 2), "min_child_weight": range(1, 6, 2)}

        grid = GridSearchCV(
            estimator=xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=150,
                objective="binary:logistic",
                nthread=4,
                scale_pos_weight=1,
                seed=27,
                eval_metric="auc",
                use_label_encoder=False,
            ),
            param_grid=param_grid,
            scoring="roc_auc",
            n_jobs=4,
            cv=5,
        )

        grid.fit(X_train, y_train)
        best_params = grid.best_params_

        clf = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=57,
            objective="binary:logistic",
            nthread=4,
            scale_pos_weight=1,
            seed=27,
            eval_metric="auc",
            use_label_encoder=False,
            max_depth=best_params["max_depth"],
            min_child_weight=best_params["min_child_weight"],
        )

        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresults = xgb.cv(
            xgb_param,
            xgtrain,
            num_boost_round=clf.get_params()["n_estimators"],
            nfold=5,
            metrics="auc",
            early_stopping_rounds=50,
        )
        clf.set_params(n_estimators=cvresults.shape[0])

        clf.fit(X_train, y_train)
        pred_test_proba = clf.predict_proba(X_test)[:, 1]

        precisions, recalls, thresholds = metrics.precision_recall_curve(
            y_test, pred_test_proba
        )
        tres = min([(abs(p - r), t) for p, r, t in zip(precisions, recalls, thresholds)])[
            1
        ]
        print(
            metrics.classification_report(
                y_test, pred_test_proba >= tres, target_names=[name_1, name_2]
            )
        )

        clf.save_model(os.path.join(self.classifiers_dir, re_expr + ".bin"))
        self.dict[re_expr] = {
            "classifier": f"{re_expr}.bin",
            "threshold": tres.item(),
            "homographs": (name_1, name_2),
        }
        with open(self.dictionary_name, "w", encoding="utf-8") as f:
            json.dump(self.dict, f, ensure_ascii=False, indent=4)

        self.dict_clf[re_expr] = clf

    def _get_emb(self, text, tok_id):
        inp = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            is_split_into_words=True,
        )
        outputs = self.model(
            input_ids=inp["input_ids"].to(self._device),
            attention_mask=inp["attention_mask"].to(self._device),
        )
        ids = inp.words()

        if tok_id + 1 in ids:
            embeds = (
                outputs.last_hidden_state[0][ids.index(tok_id) : ids.index(tok_id + 1)]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            embeds = (
                outputs.last_hidden_state[0][ids.index(tok_id) : -1]
                .detach()
                .cpu()
                .numpy()
            )
        return np.mean(embeds, axis=0)

    def _get_embs(self, examples, re_expr):
        def func(variable):
            if type(variable) == np.ndarray:
                return True
            else:
                return False

        embeds = []
        for example in tqdm(examples):
            idx = example.index(re_expr)
            embeds.append(self._get_emb(example, idx))
        return np.stack(list(filter(func, embeds)))


if __name__ == "__main__":
    homo = HomographerEN(window=10)

    from multilingual_text_parser import (
        Corrector,
        NormalizerEN,
        SentencesModifier,
        Sentenizer,
        SSMLApplier,
        SSMLCollector,
        SymbolsModifier,
        TextModifier,
        Tokenizer,
    )

    symb_mode = SymbolsModifier()
    text_mode = TextModifier()
    corrector = Corrector()
    sentenizer = Sentenizer()
    sent_mode = SentencesModifier()
    ssmlcol = SSMLCollector()
    tokenizer = Tokenizer()
    ssmlapp = SSMLApplier()
    normalizer = NormalizerEN()

    doc = Doc(
        """
        The siege ended at exactly 1541 GMT after UTK members stormed the building with tear gas and shot the suspect.
        """,
    )

    doc = text_mode(symb_mode(doc, **{"lang": "EN"}))
    doc = sentenizer(corrector(doc))
    doc = ssmlcol(sent_mode(doc))
    doc = ssmlapp(tokenizer(doc))
    doc = normalizer(doc)

    with Profiler(format=Profiler.Format.ms):
        doc = homo(doc)

    for sent in doc.sents:
        print(sent.get_attr("phonemes"))
