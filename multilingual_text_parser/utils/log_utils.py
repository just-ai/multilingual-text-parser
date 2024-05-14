import typing as tp
import inspect
import traceback


def trace(
    self,
    exception: tp.Optional[tp.Union[Exception, str]] = None,
    message: tp.Optional[str] = None,
    full: bool = True,
) -> str:
    """Trace function for logging erorrs and warnings.

    :param self: reference the class in which the error occurred
    :param exception: exception info (optional)
    :param message: debug message (optional)
    :param full: print full stack trace
    :return: string with information about the location of the error

    """
    try:
        if full:
            exc = traceback.format_exc()
            if "NoneType: None" not in exc:
                exception = exc

        if isinstance(self, str):
            class_name = self
        else:
            class_name = self.__name__ if type(self) == type else self.__class__.__name__
        tr_msg = f"[{class_name}][{inspect.stack()[1][3]}:{inspect.stack()[1][2]}]"
        if message:
            tr_msg += f": {message}"
        if exception:
            tr_msg += f": {exception}"

    except Exception as e:
        tr_msg = f"[trace] {e}"

    return tr_msg
