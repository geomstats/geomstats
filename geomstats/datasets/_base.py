"""Base tools to handle datasets.

Note: deeply inspired by sklearn.
"""

import logging
import os
import tempfile
import zipfile
from collections import namedtuple

import requests

logger = logging.getLogger("geomstats.datasets")

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

    if os.path.isfile(file_path) and zipfile.is_zipfile(file_path):
        logger.info(
            "Data has already been downloaded... using cached file ('%s').", file_path
        )
    else:
        logger.info("Downloading '%s' from %s to '%s'.", file_path, url, dirname)

        with tempfile.TemporaryDirectory(dir=dirname) as staging_dir:
            staged_path = os.path.join(staging_dir, "archive.zip")
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                with open(staged_path, "wb") as archive:
                    for chunk in response.iter_content(chunk_size=8192):
                        archive.write(chunk)

            if not zipfile.is_zipfile(staged_path):
                raise zipfile.BadZipFile("Figshare response is not a ZIP archive")
            os.replace(staged_path, file_path)

    return file_path
