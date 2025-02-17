from geomstats.test.test_case import TestCase


class GrassmannianConnectednessTestCase(TestCase):
    def test_is_connected(self, n, p, expected):
        msg = "All Grassmannians are connected."
        self.assertTrue(self.space.is_connected, msg=msg)


class GrassmannianCompactnessTestCase(TestCase):
    def test_is_compact(self):
        msg = "All Grassmannians are compact."
        self.assertTrue(self.space.is_compact, msg=msg)
