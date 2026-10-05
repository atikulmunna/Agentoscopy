import functools


def retry(times, exceptions):
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return function(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
        return wrapper
    return decorate


def is_transient(error):
    return isinstance(error, (ConnectionError, TimeoutError))
