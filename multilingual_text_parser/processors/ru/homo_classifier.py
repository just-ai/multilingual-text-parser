import os
import re
import json
import hashlib

import numpy as np
import torch

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.processors.base import BaseRawTextProcessor
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.fs import get_root_dir
from multilingual_text_parser.utils.model_loaders import load_transformer_model
from multilingual_text_parser.utils.profiler import Profiler

__all__ = ["HomographerRU"]


class Homograph:
    def __init__(self, sent_id, tok_id, word, regex, homo_type, pos):
        self.sent_id = sent_id
        self.tok_id = tok_id
        self.word = word
        self.regex = regex
        self.embedding = None
        self.type = homo_type
        self.pos = pos


class HomographerRU(BaseRawTextProcessor):
    GPU_CAPABLE: bool = True

    def __init__(self, device: str = "cpu", window=10):
        import xgboost as xgb

        self.voc = "([аеиоуыэюяёАЕИОУЫЭЮЯЁ])"
        self.window = window
        self._device = device
        if not device == "cpu" and device.replace("cuda:", "").isdigit():
            torch.cuda.set_device(int(device.replace("cuda:", "")))

        # Роберта (sberbank-ai/ruRoberta-large)
        model_dir = get_root_dir() / "data/ru/homo_classifier"
        self.model = load_transformer_model(
            model_dir / "ruRoBerta", output_hidden_states=True
        )
        self.model.to(self._device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir / "tokenizer",
            use_fast=True,
            add_prefix_space=True,
        )

        # датасет
        # self.corpus_name = get_root_dir() / "data/ru/homo_classifier/dataset.json"
        # self.corpus = json.loads(self.corpus_name.read_text(encoding="utf-8"))

        # параметры
        self.dictionary_name = get_root_dir() / "data/ru/homo_classifier/dictionary.json"
        self.dict = json.loads(self.dictionary_name.read_text(encoding="utf-8"))
        self.short_keys = [f' {key[:key.find("(")]}' for key in self.dict.keys()]
        self.corpus_keys = list(self.dict.keys())

        # классификаторы
        self.classifiers_dir = get_root_dir() / "data/ru/homo_classifier/classifiers"
        self.dict_clf = {}
        for homo in self.dict:
            file = self.dict[homo]["classifier"]
            clf = xgb.XGBClassifier()
            clf.load_model(os.path.join(self.classifiers_dir, file))
            self.dict_clf[homo] = clf

        # грамматические омографы
        self.dictionary_feats_name = (
            get_root_dir() / "data/ru/homo_classifier/dictionary_feats.json"
        )
        self.dict_feats = json.loads(
            self.dictionary_feats_name.read_text(encoding="utf-8")
        )
        self.keys_feat = []
        for feat in self.dict_feats:
            file = self.dict_feats[feat]["classifier"]
            clf = xgb.XGBClassifier()
            clf.load_model(os.path.join(self.classifiers_dir, file))
            self.dict_clf[feat] = clf
            feat_keys = [f"{key}" for key in self.dict_feats[feat]["homographs"].keys()]
            self.keys_feat.extend(feat_keys)

    @exception_handler
    def _process_text(self, doc: Doc, **kwargs):
        batch = []
        for sent_id, sent in enumerate(doc.sents):
            is_in_batch = False
            sent_text = " " + sent.text
            homo_words = []

            for w, short_w in zip(self.corpus_keys, self.short_keys):
                if short_w in sent_text:
                    homo_words.append((short_w, w))

            for tok_id, token in enumerate(sent.tokens):
                if not token.stress:
                    if token.text in self.keys_feat:
                        if not is_in_batch:
                            batch.append(
                                {"batch": [t.text for t in sent.tokens], "homographs": []}
                            )
                            is_in_batch = True
                        batch[-1]["homographs"].append(
                            Homograph(
                                sent_id,
                                tok_id,
                                token.text,
                                token.text,
                                "grammatical",
                                token.pos,
                            )
                        )
                    elif homo_words:
                        for short_w, w in homo_words:
                            if short_w.strip() in token.text and re.search(
                                f"^{w}$", token.text
                            ):
                                if not is_in_batch:
                                    batch.append(
                                        {
                                            "batch": [t.text for t in sent.tokens],
                                            "homographs": [],
                                        }
                                    )
                                    is_in_batch = True
                                batch[-1]["homographs"].append(
                                    Homograph(
                                        sent_id,
                                        tok_id,
                                        token.text,
                                        w,
                                        "lexical",
                                        token.pos,
                                    )
                                )

        self.get_embeddings(batch)
        for sample in batch:
            for homograph in sample["homographs"]:
                doc.sents[homograph.sent_id].tokens[
                    homograph.tok_id
                ].stress = self.inference(homograph)

    def get_embeddings(self, batch, num_layer=24, is_split_into_words=True):
        for sample in batch:
            with torch.inference_mode():
                inp = self.tokenizer(
                    sample["batch"],
                    return_tensors="pt",
                    max_length=512,
                    is_split_into_words=is_split_into_words,
                    truncation=True,
                    padding=True,
                )
                outputs = self.model(
                    input_ids=inp["input_ids"].to(self._device),
                    attention_mask=inp["attention_mask"].to(self._device),
                )
                ids = inp[0].word_ids

            for homograph in sample["homographs"]:
                tok_id = homograph.tok_id
                if tok_id + 1 in ids:
                    embeds = (
                        outputs[2][num_layer][0][
                            ids.index(tok_id) : ids.index(tok_id + 1)
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    embeds = (
                        outputs[2][num_layer][0][ids.index(tok_id) : -1]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                homograph.embedding = np.mean(embeds, axis=0)

    def inference(self, homograph):
        emb = homograph.embedding
        if homograph.type == "grammatical":
            for feat in self.dict_feats:
                if (feat == "noun_num" and homograph.pos == "NOUN") or (
                    feat in ["verbs_mood", "verbs_part"] and homograph.pos == "VERB"
                ):
                    if homograph.regex in self.dict_feats[feat]["homographs"]:
                        clf = self.dict_clf[feat]
                        tres = self.dict_feats[feat]["threshold"]
                        prob = clf.predict_proba(emb.reshape(1, -1))[:, 1]
                        cl = int(prob >= tres)
                        if cl == 0:
                            return self.dict_feats[feat]["homographs"][
                                homograph.regex
                            ].split("|")[0]
                        else:
                            return self.dict_feats[feat]["homographs"][
                                homograph.regex
                            ].split("|")[1]
        elif homograph.type == "lexical":
            clf = self.dict_clf[homograph.regex]
            tres = self.dict[homograph.regex]["threshold"]
            prob = clf.predict_proba(emb.reshape(1, -1))[:, 1]
            cl = int(prob >= tres)
            pos = int(self.dict[homograph.regex]["homographs"][cl])
            chars = re.sub(self.voc, r"\1|", homograph.word).split("|")
            chars[pos - 1] += "+"
            return "".join(chars)
        else:
            return

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

    def validate_all(self):
        with open("results.txt", "w", encoding="utf-8") as file:
            all_y = np.array([])
            all_preds = np.array([])
            for homo in tqdm(self.corpus.keys()):
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
                break
            file.write(metrics.classification_report(all_y, all_preds) + "\n\n")

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

        short_hash = hashlib.md5(re_expr.encode("utf-8")).hexdigest()[:6]
        clf.save_model(os.path.join(self.classifiers_dir, short_hash + ".bin"))
        self.dict[re_expr] = {
            "classifier": f"{short_hash}.bin",
            "threshold": tres.item(),
            "homographs": (name_1, name_2),
        }
        with open(self.dictionary_name, "w", encoding="utf-8") as f:
            json.dump(self.dict, f, ensure_ascii=False, indent=4)

        self.dict_clf[re_expr] = clf

    def _get_emb(
        self, text, re_expr, num_layer=24, tok_id=None, is_split_into_words=True
    ):
        with torch.inference_mode():
            inp = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                is_split_into_words=is_split_into_words,
                truncation=True,
            )
            outputs = self.model(
                input_ids=inp["input_ids"].to(self._device),
                attention_mask=inp["attention_mask"].to(self._device),
            )
            ids = inp.words()

        if tok_id:
            if tok_id + 1 in ids:
                embeds = (
                    outputs[2][num_layer][0][ids.index(tok_id) : ids.index(tok_id + 1)]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                embeds = (
                    outputs[2][num_layer][0][ids.index(tok_id) : -1]
                    .detach()
                    .cpu()
                    .numpy()
                )
            return np.mean(embeds, axis=0)

        prev = None
        cur = ""
        start = 0

        tokens = [
            self.tokenizer.decode(inp["input_ids"][0][i])
            for i in range(len(inp["input_ids"][0]))
        ]

        for i, tok in enumerate(tokens):
            if prev == ids[i]:
                cur += tok
            else:
                prev = ids[i]
                cur = tok
                start = i
            if re.search(re_expr, cur):
                embeds = outputs[2][num_layer][0][start : i + 1].detach().cpu().numpy()
                if embeds.shape == (1024,):
                    return embeds
                return np.mean(embeds, axis=0)

    def _get_embs(self, examples, re_expr):
        def func(variable):
            if type(variable) == np.ndarray:
                return True
            else:
                return False

        embeds = []
        for example in tqdm(examples):
            embeds.append(self._get_emb(example, re_expr, is_split_into_words=False))
        return np.stack(list(filter(func, embeds)))


if __name__ == "__main__":
    from multilingual_text_parser import (
        Corrector,
        RuleBasedNormalizerRU,
        SentencesModifierRU,
        SentenizerRU,
        SymbolsModifier,
        SyntaxAnalyzerRU,
        TextModifier,
        TextModifierRU,
        Tokenizer,
    )

    symb_mode = SymbolsModifier()
    text_mode = TextModifier()
    text_mode_ru = TextModifierRU()
    sent_mode = SentencesModifierRU()
    sentenizer = SentenizerRU()
    syntaxer = SyntaxAnalyzerRU()
    tokenizer = Tokenizer()
    corrector = Corrector()
    normalizer = RuleBasedNormalizerRU()
    homo = HomographerRU(window=10, device="cuda:0")

    doc = Doc(
        """
        Он закрылся на &замо+к, и живет в замке.
        """,
    )

    doc = text_mode(symb_mode(doc, **{"lang": "RU"}))
    doc = text_mode_ru(symb_mode(doc))
    doc = sentenizer(corrector(doc))
    doc = tokenizer(sent_mode(doc))
    doc = syntaxer(doc)
    doc = text_mode.restore(doc)
    doc = normalizer(doc)

    with Profiler(format=Profiler.Format.ms):
        doc = homo(doc)

    for sent in doc.sents:
        print(sent.stress)
