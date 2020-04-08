import functools


class NotSupportedError(Exception):
    pass


def mark_not_supported(*args, **kwargs):
    raise NotSupportedError
