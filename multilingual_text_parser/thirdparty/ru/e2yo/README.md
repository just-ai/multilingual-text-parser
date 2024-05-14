
# E2Yo

Маленькая библиотека для безопасной замены `e` на `ё` в словах.

### Usage as python module

```python
from e2yo.core import E2Yo

e2yo = E2Yo()

e2yo.replace("Артем - это имя с буквой е")

>>> "Артём это имя с буквой е"
```
Буква `е` на конце предложения заменена не была, так как она может быть интерпретирована и как буква `е` и как буква `ё`.


### Usage as CLI

Пример:
```
python e2yo.py --input_path text.txt --out_path out_text.txt
```
Аргументы:
```
usage: e2yo.py [-h] --input_path INPUT_PATH --out_path OUT_PATH
               [--encoding ENCODING]

Changes all "e" to "ё" in txt file.

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to input .txt file
  --out_path OUT_PATH   Path to output .txt file
  --encoding ENCODING   Encoding of input and output files.

```
### Installation

```
pip install -U -e .
```

### Known issues
Модель может плохо работать с сокращениями. Например, `мед. училище` -> `мёд. училище`.
