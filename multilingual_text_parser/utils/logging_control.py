import socket
import typing as tp
import logging

from multiprocessing import current_process
from pathlib import Path

import numpy as np

__all__ = ["create_logger"]

LOGGER = logging.getLogger("root")


def _check_if_port_used(addr: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        host, port = addr.split(":")
        return s.connect_ex((host, int(port))) == 0


def _create_log_file(log_file: Path, log_name: str) -> Path:
    if not log_file.exists():
        if log_file.suffix != ".txt":
            log_file /= "log.txt"
        log_dir = log_file.parents[0]
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        if log_file.is_dir():
            log_file /= "log.txt"

    return log_file.with_name(f"{log_name}_{log_file.name}")


def _get_formatter():
    processname = current_process().name
    formatter = logging.Formatter(
        f"[{processname}] %(asctime)s:%(levelname)s:%(message)s"
    )
    return formatter


def create_logger(
    log_name: str = "root",
    log_file: tp.Optional[tp.Union[str, Path]] = None,
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
    use_server_logging: bool = False,
    use_file_logging: bool = True,
    use_console_logging: bool = True,
):
    root_logger = logging.getLogger(log_name)

    # create loggere
    formatter = _get_formatter()
    root_logger.setLevel(file_level)
    root_logger.handlers.clear()
    np.set_printoptions(precision=5)

    # server logging
    if use_server_logging:
        raise NotImplementedError("Server logging is not supported in parser")

    # file logging
    if use_file_logging and log_file:
        if not any(type(hd) == logging.FileHandler for hd in root_logger.handlers):
            log_file = _create_log_file(Path(log_file), log_name)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(file_level)
            root_logger.addHandler(file_handler)

    # console logging
    if use_console_logging:
        if not any(type(hd) == logging.StreamHandler for hd in root_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(console_level)
            root_logger.addHandler(console_handler)

    return root_logger
