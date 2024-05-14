import re
import base64
import typing as tp

from pathlib import Path

from multilingual_text_parser.utils.fs import get_root_dir

__all__ = ["E2Yo"]


class E2Yo:
    """Class for replacing cyrillic character `е` to `ё`.

    Args:
        dict_path (Union[str, Path):
            Path to dict of words with `ё` character.

    """

    def __init__(self, dict_path: tp.Optional[tp.Union[str, Path]] = None):

        if dict_path is None:
            root_dir = get_root_dir()
            data_dit = "data/ru/e2yo/data"
            dict_name = "e2yo_safe_dict.txt"
            dict_path = root_dir / data_dit / dict_name

        self.dict_path = dict_path
        self.e2yo_mapping = E2Yo._load_dict(dict_path)

    def replace(self, string: str) -> str:
        """Function for replacing all words in a string.

        Args:
            string (str): String where to replace.
        Returns:
            replaced_string (str): String with replaced `e` to `ё` if needed.

        """
        replaced = []
        for match in re.finditer(r"(?P<Word>\w+)|(?P<Other>\W+)", string):
            if match.group("Word") is not None:
                replaced.append(self._replace_word(match.group("Word")))
            else:
                replaced.append(match.group("Other"))

        return "".join(replaced)

    def _replace_word(self, word: str) -> str:
        """Replacing function.

        Args:
            word (str): Word to replace `е` to `ё`.
        Returns:
            replaced_word (str): Word with replaced `e` to `ё` if needed.

        """
        word_lower = word.lower()
        list_word = list(word)
        if word_lower in self.e2yo_mapping.keys():
            true_word = self.e2yo_mapping[word_lower]
            for yo in re.finditer("ё", true_word):
                yo_pos = yo.start()
                if list_word[yo_pos] == "е":
                    list_word[yo_pos] = "ё"
                elif list_word[yo_pos] == "ё":
                    pass
                else:
                    list_word[yo_pos] = "Ё"
            return "".join(list_word)
        else:
            return word

    @staticmethod
    def _load_dict(path: tp.Union[str, Path]) -> tp.Mapping[str, str]:
        """Loads dict from file and returns mapping е -> ё.

        Args:
            path (Union[str, Path]):
                Path to dict of words with `ё` character.
        Returns:
            e2yo_mapping (Mapping[str, str]):
                Word-level mapping, e.g. подплетет -> подплетёт

        """
        e2yo_mapping: tp.Dict[str, str] = {}

        if isinstance(path, str):
            path = Path(path)

        if path.suffix == ".bin":
            file = base64.b64decode(path.read_bytes()).decode()
        else:
            file = path.read_text(encoding="utf-8")

        for row in file.split("\n"):
            row = row.replace("\n", "")
            # Check if word have multiple endings
            if re.match(r".*\(*\)", row):
                word, endings = row.split("(")
                endings = endings[:-1]
                endings = endings.split("|")
            else:
                word, endings = row, []

            if len(endings) == 0:
                e2yo_mapping[word.lower().replace("ё", "е")] = word.lower()
            else:
                for ending in endings:
                    e2yo_mapping[
                        "".join(
                            [
                                word.lower().replace("ё", "е"),
                                ending.lower().replace("ё", "е"),
                            ]
                        )
                    ] = "".join([word.lower(), ending])

        return e2yo_mapping
