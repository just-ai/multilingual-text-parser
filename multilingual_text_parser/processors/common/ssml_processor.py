import re
import time
import random
import logging
import xml.dom.minidom

from copy import deepcopy
from string import punctuation
from typing import Any, Dict, List, Tuple

import numpy as np

from multilingual_text_parser._constants import ALLOWED_SSML_TAGS
from multilingual_text_parser.data_types import Doc, Sentence
from multilingual_text_parser.processors.base import BaseSentenceProcessor
from multilingual_text_parser.utils.decorators import exception_handler
from multilingual_text_parser.utils.log_utils import trace

__all__ = ["SSMLCollector", "SSMLApplier"]

LOGGER = logging.getLogger("root")


class SSMLParsingError(Exception):
    pass


def parse_node_dfs(
    node: xml.dom.Node, modifiers: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, Dict]]:
    """
    Parses xml-structure, e.g. "<p> asdasd csad <b/> sa </p>"
    Args:
        node: Given xml Node w (w/o) children
        modifiers: accumulating term

    Returns:
        Sequence of modifiers for each substring
    """
    this_node_modifiers = deepcopy(modifiers)
    if node.attributes is not None:  # type: ignore
        attributes = {k: v.nodeValue for k, v in dict(node.attributes).items()}  # type: ignore
    else:
        attributes = {}

    if hasattr(node, "tagName"):
        tagname = node.tagName  # type: ignore
        if tagname != "speak":
            if node.tagName not in this_node_modifiers:
                this_node_modifiers.update({node.tagName: attributes})  # type: ignore
            else:
                this_node_modifiers[node.tagName].update(attributes)  # type: ignore
    parsed = []
    childnodes = node.childNodes  # type: ignore

    if len(childnodes):
        for child in childnodes:
            parsed.extend(parse_node_dfs(child, this_node_modifiers))
        return parsed

    else:
        if hasattr(node, "wholeText"):  # paired tags
            return [(node.wholeText, this_node_modifiers)]  # type: ignore
        else:
            return [("", this_node_modifiers)]


def get_random_replacement_string(n: int = 8):
    source = "абвгдежзиклмнопрст"
    idxs = np.random.choice(range(len(source)), n, replace=False)
    return "".join([source[i] for i in idxs]).upper()


def collect_ssml(doc: Doc) -> None:
    _str = doc.text
    _str = re.sub(r"([ \s\t\n\r]*)([<>])([ \s\t\n\r]*)", r"\2", _str)

    _tags_replacement_map = {}
    for instance in re.findall(r"<.+?>", _str):
        if not any([x in instance for x in ALLOWED_SSML_TAGS]):
            continue
        replacement = get_random_replacement_string()
        if instance in ["<s>", "</s>"]:
            _str = _str.replace(instance, ".", 1)
            continue
        if instance in ["<p>", "</p>"]:
            _str_updated = _str.replace(instance, f". {replacement} ", 1)
            instance = "<break time='0.3s'/>"
        else:
            _str_updated = _str.replace(instance, f" {replacement} ", 1)
        if _str != _str_updated:
            _tags_replacement_map.update({replacement: instance})
        _str = _str_updated
    doc.tags_replacement_map = _tags_replacement_map  # type: ignore
    doc.text = _str


def process_ssml_string(ssml_string: str):
    """
    parses ssml-tags for given string.
    Args:
        ssml_string: text with or without ssml tags

    Returns:

    """
    ssml_string = "<speak>" + ssml_string + "</speak>"
    try:
        dom = xml.dom.minidom.parseString(ssml_string)
    except xml.parsers.expat.ExpatError:
        raise SSMLParsingError(
            "Exception while parsing ssml-tags. Only in-sentence tags allowed."
        )
    parsed = parse_node_dfs(dom.firstChild, {})
    return parsed


def handle_ssml_tags(
    norm_sentence: Sentence, text_modifiers: List[Tuple[str, Dict]]
) -> Sentence:
    """
    Args:
        norm_sentence: Sentence
        text_modifiers: Sequence with mappings 'substring: modifiers'

    Returns:
        Sentence with set .modifiers attribute for each token
    """
    # TODO: problems with tags like от +6 до <prosody>−50 °C</prosody>, bcz "до -50°C" becomes single tag.
    #  We cant split modifier on tag parts.

    def safe_modifiers(modifiers):
        new_modifiers = []
        for modifier in modifiers:
            text, mod = modifier
            if len(re.findall(r"\S+", text)) > 0:
                text = text.replace(" ", "")
            new_modifiers.append((text, mod))
        return new_modifiers

    text_modifiers = safe_modifiers(text_modifiers)
    modifier_start_idx = 0
    for idx, token in enumerate(norm_sentence.tokens):
        token_text = token.text.replace("+", "")
        for modifier_idx in range(modifier_start_idx, len(text_modifiers)):
            substring, modifiers = text_modifiers[modifier_idx]
            for key in list(modifiers.keys()):
                if isinstance(modifiers[key], dict):
                    for k in list(modifiers[key].keys()):
                        modifiers[key][k.strip()] = modifiers[key].pop(k).strip()
                modifiers[key.strip()] = modifiers.pop(key)

            substring = substring.replace("+", "")
            if modifier_idx == modifier_start_idx and substring == "":
                norm_sentence.ssml_insertions.append((idx - 1, modifiers))
                modifier_start_idx += 1
            elif len(substring.strip()) == 0:
                modifier_start_idx += 1

            else:
                if substring.startswith(token_text.replace(" ", "")):
                    token.modifiers = modifiers

                    if "intonation" in modifiers and not token.is_punctuation:
                        label = modifiers["intonation"].get("label", "")
                        if label == "random":
                            label = str(int(9 * random.random()))
                        if label.lstrip("-").isdigit() and -1 <= int(label) < 10:
                            token.prosody = str(label)

                    if "emphasis" in modifiers:
                        token.emphasis = "accent"

                    if "sub" in modifiers:
                        alias = modifiers["sub"].get("alias", "")
                        if alias:
                            token.text = re.sub(r"[^a-zа-яё0-9]", "", alias.lower())

                    if "say-as" in modifiers:
                        stress = modifiers["say-as"].get("stress", "")
                        if stress.isdigit():
                            stress_index = int(stress)
                            vowel_index = 0
                            for idx, letter in enumerate(token.norm):
                                if letter in "ауоиэыяюеё":
                                    vowel_index += 1
                                    if stress_index == vowel_index:
                                        token.stress = (
                                            f"{token.text[:idx+1]}+{token.text[idx+1:]}"
                                        )
                                        break

                        interpret_as = modifiers["say-as"].get("interpret-as", "")
                        if interpret_as:
                            token.interpret_as = interpret_as
                            token.format = modifiers["say-as"].get("format")

                            if interpret_as == "date":
                                for sep in [".", "-", ":"]:
                                    if sep in token.text:
                                        token.text = token.text.replace(sep, ".")
                                        break

                            if interpret_as == "time":
                                for sep in [".", "-", ":"]:
                                    if sep in token.text:
                                        token.text = token.text.replace(sep, ":")
                                        break
                                else:
                                    if token.is_number and len(token.text) == 4:
                                        token.text = f"{token.text[:2]}:{token.text[2:]}"

                            if interpret_as in ["cardinal", "ordinal"]:
                                if token.format:
                                    token.text = token.text.replace(token.format, ".")

                    new_substring = substring.replace(token_text.replace(" ", ""), "", 1)

                    if len(new_substring.strip()) == 0:
                        modifier_start_idx += 1
                    else:
                        text_modifiers[modifier_idx] = (new_substring, modifiers)
                    break

    return norm_sentence


def is_closing(tag: str):
    return tag[1] == "/"


def is_single(tag: str):
    return tag[-2] == "/"


def are_same(tag1: str, tag2: str):
    return re.findall(r"\w+", tag1)[0] == re.findall(r"\w+", tag2)[0]


def construct_closing_tag(tag: str):
    tagtype = re.findall(r"\w+", tag)[0]
    return f"</{tagtype}>"


class SSMLCollector(BaseSentenceProcessor):
    def __call__(self, doc: Doc, **kwargs) -> Doc:
        """
        Part of text processor.
        Args:
            doc: Text structure having each sentence with filled attribute .tags_replacement_map from `collect_ssml`
            **kwargs:

        Returns:
            Text with computed .ssml_modifiers and .ssml_insertions attribute which has structure:
                [(substring1, modifiers1), (substring2, modifiers2) ... ]
                so substring_1 + ... + substrinng_n == sent.text

        """
        _tags_replacement_map = doc.tags_replacement_map
        if _tags_replacement_map is not None:
            for key in list(_tags_replacement_map.keys()):
                _tags_replacement_map[key.lower()] = _tags_replacement_map.pop(key)
        else:
            return doc

        processed_sentences = []
        open_tags_stack = []  # This stack is about all opened tags
        for idx, sentence in enumerate(doc.sents):
            _str: str = sentence.text.strip()

            # If we have opened tags we add them to beginning of processed string
            for o_replacement, o_tag in reversed(open_tags_stack):
                _str = f"{o_tag} " + _str

            # Parsing of tags for current sentence
            this_sentence_tags = []
            for replacement in _tags_replacement_map.keys():
                position = _str.find(replacement)
                if position != -1:
                    this_sentence_tags.append(
                        (replacement, _tags_replacement_map[replacement], position)
                    )
            this_sentence_tags = sorted(this_sentence_tags, key=lambda x: x[2])

            # From left to right:
            # if tag is single one - just replace
            # elif tag is open one - store in stack
            # if tag is closing - we check the last one n stack,
            #                     and if they are the same,
            #                     we delete open tag from stack and make replacements
            for replacement, tag, _ in this_sentence_tags:
                if is_single(tag):
                    _str = _str.replace(replacement, tag)
                elif not is_closing(tag):
                    open_tags_stack.append((replacement, tag))
                    _str = _str.replace(replacement, tag)
                else:
                    if len(open_tags_stack) == 0:
                        _str = _str.replace(replacement, tag)
                    elif are_same(open_tags_stack[-1][1], tag):
                        _ = open_tags_stack.pop()
                        _str = _str.replace(replacement, tag)

            # If we still have open tags, we close them for this sentence w/o deletion from stack
            # TODO: как сделать так, чтобы пунктуации тоже раздался последний тег?
            punct_symbols = []
            for symbol in _str[::-1]:
                if symbol in punctuation:
                    punct_symbols.append(symbol)
                else:
                    break
            punct_tail = "".join(reversed(punct_symbols))
            _str = _str[: -len(punct_tail)]
            for o_replacement, o_tag in reversed(open_tags_stack):
                _str += f" {construct_closing_tag(o_tag)}"
            _str += punct_tail

            temp = Doc(_str, sentenize=True).sents
            if not len(temp) == 1:
                raise RuntimeError(
                    f"Raised in splitting: issue with recomposing sentences. [{_str}]"
                )

            new_sentence = temp[0]
            new_sentence.text_orig = sentence.text_orig
            _str = new_sentence.text

            from multilingual_text_parser.processors.ru.modifiers import TextModifier

            _str = TextModifier._clean_text(_str)

            try:
                modifiers = process_ssml_string(_str)
                new_sentence.ssml_modidfiers = modifiers
            except Exception as e:
                LOGGER.error(trace(self, e, _str, full=False))

            _text = new_sentence.text
            for instance in re.findall(r"<.+?>", _str):
                _text = _text.replace(instance, "")
            new_sentence.text = _text
            processed_sentences.append(new_sentence)

        processed_sentences = [
            s for s in processed_sentences if len(re.findall(r"\w", s.text)) > 0
        ]
        doc.sents = processed_sentences
        return doc


class SSMLApplier(BaseSentenceProcessor):
    @exception_handler
    def _process_sentence(self, sent: Sentence, **kwargs):
        if sent.ssml_modidfiers is not None:
            text_modifiers = deepcopy(sent.ssml_modidfiers)
            handle_ssml_tags(sent, text_modifiers)
