import pytest

from thirdparty.e2yo.e2yo.core import E2Yo


@pytest.mark.parametrize(
    ("filename", "true_dict"),
    [
        (
            "./data/e2yo_test_dict.txt",
            {
                "артем": "артём",
                "артема": "артёма",
                "артемов": "артёмов",
                "артемчик": "артёмчик",
                "артемушка": "артёмушка",
            },
        )
    ],
)
def test_dict_reading(filename: str, true_dict: dict):
    e2yo = E2Yo(filename)
    assert e2yo.e2yo_mapping == true_dict
