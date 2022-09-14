"""Base tools to handle datasets.

Note: deeply inspired by sklearn.
"""

import logging
import os
from collections import namedtuple
from urllib.request import urlretrieve

DEFAULT_DATA_DIR = os.path.expanduser(os.path.join("~", ".geomstats_data"))


RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url"])


def _get_data_home(data_home=None):
    if data_home is None:
        data_home = DEFAULT_DATA_DIR

    os.makedirs(data_home, exist_ok=True)

    return data_home


def _fetch_remote(url, filename, dirname=None):
    dirname = _get_data_home(data_home=dirname)

    file_path = os.path.join(dirname, filename)

    if os.path.exists(file_path):
        logging.info(
            f"Data has already been downloaded... using cached file ('{file_path}')."
        )
    else:
        logging.info(f"Downloading '{file_path}' from {url} to '{dirname}'.")
        urlretrieve(url, file_path)

    return file_path
