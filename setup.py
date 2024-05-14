from pathlib import Path
from typing import Any, Dict

import setuptools

# The directory containing this file
HERE = Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding="utf-8")

about: Dict[Any, Any] = {}
with open("multilingual_text_parser/_version.py") as f:
    exec(f.read(), about)


def _load_requirements(path_dir: Path, comment_char: str = "#"):
    requirements_directory = path_dir / "requirements.txt"
    requirements = []
    with requirements_directory.open("r") as file:
        for line in file.readlines():
            line = line.lstrip()
            # Filter all comments
            if comment_char in line:
                line = line[: line.index(comment_char)]
            if line:
                requirements.append(line)
    return requirements


flist = Path("multilingual_text_parser/data").rglob("*")
data = [path.relative_to("multilingual_text_parser").as_posix() for path in flist]

# This call to setup() does all the work
setuptools.setup(
    name="multilingual_text_parser",
    version=about["__version__"] + "-compiled",
    description="Text normalizer and phonemizer for TTS systems",
    long_description=README,
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=_load_requirements(HERE),
    package_data={"multilingual_text_parser": data},
    # https://nuitka.net/doc/user-manual.html#use-case-5-setuptools-wheels
    command_options={
        "nuitka": {
            "--python-flag": "no_docstrings",
            "--nofollow-import-to": [
                "tests.*",
            ],
        }
    },
)
