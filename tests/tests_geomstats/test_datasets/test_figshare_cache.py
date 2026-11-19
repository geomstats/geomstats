import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from geomstats.datasets import _base


class FigshareCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / "data.zip"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("data", "content")
        self.payload = buffer.getvalue()

    def fetch(self, chunks):
        response = MagicMock()
        response.__enter__.return_value.iter_content.return_value = chunks
        with patch.object(_base.requests, "get", return_value=response):
            return _base.download_figshare_zip(123, "data.zip", self.root)

    def test_repair_and_reuse(self):
        self.target.write_bytes(b"interrupted")
        self.fetch([self.payload])
        self.assertEqual(self.target.read_bytes(), self.payload)
        with patch.object(_base.requests, "get", side_effect=AssertionError):
            _base.download_figshare_zip(123, "data.zip", self.root)

    def test_invalid_response(self):
        with self.assertRaises(zipfile.BadZipFile):
            self.fetch([b"<html>error</html>"])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_interrupted_response(self):
        def interrupted():
            yield b"partial"
            raise OSError("connection lost")

        with self.assertRaises(OSError):
            self.fetch(interrupted())
        self.assertEqual(list(self.root.iterdir()), [])
