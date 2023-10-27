import pytest


def _has_ignore_marker(markers):
    for marker in markers:
        if marker.name == "ignore":
            return True

    return False


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    selected = []
    deselected = []
    for item in items:
        if _has_ignore_marker(item.own_markers):
            deselected.append(item)
        else:
            selected.append(item)

    config.hook.pytest_deselected(items=deselected)

    items[:] = selected


def pytest_make_parametrize_id(config, val):
    if isinstance(val, tuple):
        return "-".join([repr(val_) for val_ in val])
    return repr(val)
