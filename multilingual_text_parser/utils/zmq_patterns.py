import time
import socket
import typing as tp
import logging

from contextlib import closing
from dataclasses import dataclass

import zmq
import zmq.asyncio

from multilingual_text_parser.utils.log_utils import trace

__all__ = [
    "ZMQPatterns",
    "ZMQClient",
    "find_free_port",
]

LOGGER = logging.getLogger("root")


@dataclass
class ZMQClient:
    context: zmq.Context
    socket: zmq.Socket

    def close(self):
        self.socket.close()

    def send(self, message, serialize: bool = True):
        try:
            self.socket.send_pyobj(message) if serialize else self.socket.send(message)
        except Exception as e:
            LOGGER.error(trace(self, e))

    def recv(
        self,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ):
        try:
            if timeout is not None and self.socket.poll(timeout=timeout) == 0:
                return None
            else:
                return (
                    self.socket.recv_pyobj()
                    if deserialize
                    else self.socket.recv_multipart()
                )
        except Exception as e:
            LOGGER.error(trace(self, e))

    def request(
        self,
        message,
        serialize: bool = True,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ):
        self.send(message, serialize)
        return self.recv(deserialize, timeout)

    def send_string(self, message: str):
        self.socket.send_string(message)

    def recv_string(self, timeout: tp.Optional[int] = None):  # in milliseconds
        try:
            if timeout is not None and self.socket.poll(timeout=timeout) == 0:  # wait
                return None  # timeout reached before any events were queued
            else:
                return self.socket.recv_string()  # events queued within our time limit
        except Exception as e:
            LOGGER.error(trace(self, e))

    def request_as_string(
        self, message: str, timeout: tp.Optional[int] = None  # in milliseconds
    ):
        self.send_string(message)
        return self.recv_string(timeout)


class ZMQPatterns:
    @staticmethod
    def __create_socket_and_connect(
        context: zmq.Context, addr: str, socket_type
    ) -> zmq.Socket:
        socket = context.socket(socket_type)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{addr}")
        return socket

    @staticmethod
    def __get_req(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.REQ)

    @classmethod
    def client(cls, server_addr: str) -> ZMQClient:
        context = zmq.Context()
        socket = cls.__get_req(context, server_addr)
        return ZMQClient(context=context, socket=socket)


def find_free_port():
    time.sleep(1)
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_nane = s.getsockname()[1]
        s.close()
        return port_nane
