import pytest

from thirdparty.e2yo.e2yo.core import E2Yo


@pytest.mark.parametrize(
    ("input_string", "true_string"),
    [
        ("Артём начал писать тест", "Артём начал писать тест"),
        ("Артем продолжил писать тест", "Артём продолжил писать тест"),
        (
            "Артемушка пишет тест с цифорками - 1, 2, 123!",
            "Артёмушка пишет тест с цифорками - 1, 2, 123!",
        ),
        ("АРТЕМЧИК ПИШЕТ ТЕКСТ С КАПСЛОКОМ!", "АРТЁМЧИК ПИШЕТ ТЕКСТ С КАПСЛОКОМ!"),
        (
            "АРТЕМ ПИШЕТ последний \n ТеСТ С Несколькими СтРоКаМи \n АРТЕМ",
            "АРТЁМ ПИШЕТ последний \n ТеСТ С Несколькими СтРоКаМи \n АРТЁМ",
        ),
    ],
)
def test_string_replacing(input_string, true_string):
    e2yo = E2Yo("./data/e2yo_test_dict.txt")
    assert true_string == e2yo.replace(input_string)
