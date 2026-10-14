"""Stand-in for tenacity: gorilla wraps its own API calls (generate_with_backoff) with a retry decorator. rollout.py
overrides _query_FC and never uses that path, so the decorator is a pass-through and conditions are inert."""


def retry(*a, **k):
    if a and callable(a[0]) and not k:
        return a[0]

    def deco(f):
        return f
    return deco


class _Cond:
    def __init__(self, *a, **k):
        pass

    def __or__(self, other):
        return self

    __ror__ = __or__


def retry_if_exception_type(*a, **k):
    return _Cond()


def retry_if_exception_message(*a, **k):
    return _Cond()


def retry_any(*a, **k):
    return _Cond()


def wait_random_exponential(*a, **k):
    return None


def stop_after_attempt(*a, **k):
    return None
