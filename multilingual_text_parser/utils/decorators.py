from functools import wraps

from multilingual_text_parser.data_types import Sentence
from multilingual_text_parser.utils.log_utils import trace


def exception_handler(func):
    @wraps(func)
    def decorated_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            sent: Sentence = args[1]
            sent.exception_messages.append(
                f'{args[0].__class__.__name__}, failed on "{sent.text_orig}"|"{sent.text}" with {trace("exception_handler", e)}'
            )

    return decorated_func
