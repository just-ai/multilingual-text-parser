import time
import typing as tp
import logging

from dataclasses import dataclass
from enum import Enum

from multilingual_text_parser.utils.log_utils import trace

__all__ = ["Profiler"]

LOGGER = logging.getLogger("root")


@dataclass
class Profiler:
    class Format(Enum):
        h = 1.0 / (60.0**2)
        m = 1.0 / 60.0
        s = 1.0
        ms = 1000.0
        ns = 1000.0**2

    name: str = ""
    auto_logging: bool = True
    format: Format = Format.s
    enable: bool = True

    def __post_init__(self):
        self.reset()
        if self.auto_logging and not LOGGER.handlers:
            LOGGER.addHandler(logging.StreamHandler())
            LOGGER.setLevel(10)

    def __enter__(self):
        if self.enable:
            self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.total_time()
            self.logging()

    def reset(self):
        self._begin_time = time.perf_counter()
        self._tick_time = self._begin_time
        self._start_time = {}
        self._timings = {}

    def _add(self, name, begin_time):
        value = self._timings.setdefault(name, [0, 0])
        value[0] += time.perf_counter() - begin_time
        value[1] += 1

    def start(self, name: tp.Optional[str] = None):
        name = name if name else self.name
        self._start_time[name] = time.perf_counter()

    def stop(self, name: tp.Optional[str] = None):
        name = name if name else self.name
        self._add(name, self._start_time[name])

    def tick(self, name: tp.Optional[str] = None):
        name = name if name else self.name
        self._add(name, self._tick_time)
        self._tick_time = time.perf_counter()

    def total_time(self, name: tp.Optional[str] = None):
        name = name if name else self.name
        self._timings[name] = [0, 0]
        self._add(name, self._begin_time)

    def _get(self, name: tp.Optional[str] = None) -> tp.Tuple[float, int]:
        name = name if name else self.name
        value = self._timings.get(name)
        if value:
            return round((value[0] / value[1]) * self.format.value, 3), value[1]
        else:
            return (
                round((time.perf_counter() - self._begin_time) * self.format.value, 3),
                1,
            )

    def get_time(self, name: tp.Optional[str] = None) -> float:
        return self._get(name)[0]

    def get_counter(self, name: tp.Optional[str] = None) -> int:
        return self._get(name)[1]

    def logging(
        self,
        summary_writer: tp.Optional[tp.Any] = None,
        current_iter: tp.Optional[int] = 0,
    ):
        tm = []
        for key, value in self._timings.items():
            avg_time = round((value[0] / value[1]) * self.format.value, 2)
            tm.append(f"{key} time: {avg_time} {self.format.name}")
            if summary_writer:
                summary_writer.add_scalar(
                    f"{key} time", avg_time, global_step=current_iter
                )

        if self.auto_logging:
            LOGGER.info(trace(self, message="; ".join(tm)))

    @staticmethod
    def counter(multiple: Format = Format.s):
        return time.perf_counter() * multiple.value

    @staticmethod
    def sleep(seconds: float = 0):
        return time.sleep(seconds)


if __name__ == "__main__":
    profiler = Profiler(format=Profiler.Format.ms)
    profiler.start("test")

    time.sleep(0.05)
    profiler.tick("sleep1")

    for _ in range(10):
        time.sleep(0.1)
        profiler.tick("sleep2")

    time.sleep(0.5)
    profiler.total_time("total")

    profiler.logging()

    with Profiler("context"):
        time.sleep(0.75)

    with Profiler(format=Profiler.Format.ms, auto_logging=False) as prof:
        time.sleep(0.5)
    print("time", prof.get_time())

    profiler.stop("test")
    print("time test", profiler.get_time("test"))
