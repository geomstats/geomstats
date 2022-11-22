def _has_ignore_marker(markers):
    for marker in markers:
        if marker.name == "ignore":
            return True

    return False


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
