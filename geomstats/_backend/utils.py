import functools


class NotSupportedError(Exception):
    pass


def mark_not_supported(*args, **kwargs):
    raise NotSupportedError


def alias_argument_names(**mapping):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for name, alias in mapping.items():
                if name in kwargs:
                    kwargs[alias] = kwargs[name]
                    del kwargs[name]
            return function(*args, **kwargs)
        return wrapper
    return decorator
