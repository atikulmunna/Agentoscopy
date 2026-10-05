def retry(times, exceptions):
    raise NotImplementedError


def is_transient(error):
    return isinstance(error, (ConnectionError, TimeoutError))
