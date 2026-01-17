"""Base tools to handle datasets.

Note: deeply inspired by sklearn.
"""

import logging
import os
from collections import namedtuple

import requests

DEFAULT_DATA_DIR = os.path.expanduser(os.path.join("~", ".geomstats_data"))


FigshareMetadata = namedtuple("FigshareMetadata", ["filename", "article_id"])


def _get_data_home(data_home=None):
    if data_home is None:
        data_home = DEFAULT_DATA_DIR

    os.makedirs(data_home, exist_ok=True)

    return data_home


def download_figshare_zip(article_id, filename, dirname=None):
    url = f"https://api.figshare.com/v2/articles/{article_id}/download"

    if filename is None:
        filename = f"{article_id}.zip"

    dirname = _get_data_home(data_home=dirname)
    file_path = os.path.join(dirname, filename)

    if os.path.exists(file_path):
        logging.info(
            f"Data has already been downloaded... using cached file ('{file_path}')."
        )
    else:
        logging.info(f"Downloading '{file_path}' from {url} to '{dirname}'.")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    return file_path
