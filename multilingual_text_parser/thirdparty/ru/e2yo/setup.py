import os

from pathlib import Path

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def _get_version(filename):
    from re import findall

    with open(filename) as f:
        metadata = dict(findall("__([a-z]+)__ = '([^']+)'", f.read()))
    return metadata["version"]


def _load_requirements(path_dir=THIS_DIR, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt")) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)]
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name="e2yo",
    version=_get_version("e2yo/__init__.py"),
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=_load_requirements(THIS_DIR),
    package_dir={"e2yo": "e2yo"},
    package_data={"e2yo": ["data/*"]},
)
