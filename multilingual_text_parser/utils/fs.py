import os
import typing as tp
import logging

from pathlib import Path

__all__ = ["find_files", "get_root_dir"]

LOGGER = logging.getLogger("root")


def get_root_dir() -> Path:
    abs_path = Path(__file__).absolute()
    abs_path_parts = abs_path.parent.parts
    root_folder_idx = abs_path_parts[::-1].index("multilingual_text_parser")
    root_dir = abs_path.parents[root_folder_idx]
    return root_dir


def find_files(
    dir_path: str,
    extensions=(".*",),
    ext_lower=False,
) -> tp.List[str]:

    if ext_lower:
        file_list = [
            os.path.join(r, fn)  # type: ignore
            for r, ds, fs in os.walk(dir_path)
            for fn in fs
            if fn.lower().endswith(extensions)
        ]
    else:
        file_list = [
            os.path.join(r, fn)  # type: ignore
            for r, ds, fs in os.walk(dir_path)
            for fn in fs
            if fn.endswith(extensions)
        ]

    return file_list
