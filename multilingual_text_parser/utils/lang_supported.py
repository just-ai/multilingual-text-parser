import json
import typing as tp
import logging

from pathlib import Path

import phonemizer

from multilingual_text_parser.utils.fs import get_root_dir

__all__ = ["espeak_available_languages", "stanza_available_languages"]

LOGGER = logging.getLogger("root")


def espeak_available_languages():
    try:
        return (
            phonemizer.backend.espeak.espeak.EspeakBackend("es")
            .supported_languages()
            .keys()
        )
    except RuntimeError:
        LOGGER.warning(
            "eSpeak not installed on your system (https://espeak.sourceforge.net/download.html)"
        )
        return tuple()


def stanza_available_languages(resources_path: tp.Optional[Path] = None):
    if resources_path is None:
        resources_path = get_root_dir() / "data/common/stanza_resources/resources.json"

    resources = json.loads(resources_path.read_text())
    languages = [
        lang
        for lang in resources
        if not isinstance(resources[lang], str) and "alias" not in resources[lang]
    ]
    languages = sorted(languages)
    return languages
