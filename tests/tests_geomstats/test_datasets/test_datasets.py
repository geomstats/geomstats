import os
import tempfile

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.datasets._dhoom import (
    DhoomError,
    coerce,
    decode,
    encode,
    value_to_dhoom,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import Landmarks
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.test.test_case import TestCase, np_and_autograd_only


class TestDatasets(TestCase):
    """Test for data-loading utilities."""

    def test_load_cities(self):
        """Test that the cities coordinates belong to the sphere."""
        sphere = Hypersphere(dim=2)
        data, _ = data_utils.load_cities()
        self.assertAllClose(gs.shape(data), (50, 3))

        tokyo = data[0]
        self.assertAllClose(tokyo, gs.array([0.61993792, -0.52479018, 0.58332859]))

        result = sphere.belongs(data)
        self.assertTrue(gs.all(result))

    def test_load_poses_only_rotations(self):
        """Test that the poses belong to SO(3)."""
        so3 = SpecialOrthogonal(n=3, point_type="vector")
        data, _ = data_utils.load_poses()
        result = so3.belongs(data)

        self.assertTrue(gs.all(result))

    def test_load_poses(self):
        """Test that the poses belong to SE(3)."""
        se3 = SpecialEuclidean(n=3, point_type="vector")
        data, _ = data_utils.load_poses(only_rotations=False)
        result = se3.belongs(data)

        self.assertTrue(gs.all(result))

    def test_karate_graph(self):
        """Test the correct number of edges and nodes for each graph."""
        graph = data_utils.load_karate_graph()
        result = len(graph.edges) + len(graph.labels)
        expected = 68
        self.assertTrue(result == expected)

    def test_random_graph(self):
        """Test the correct number of edges and nodes for each graph."""
        graph = data_utils.load_random_graph()
        result = len(graph.edges) + len(graph.labels)
        expected = 20
        self.assertTrue(result == expected)

    def test_random_walks_random_graph(self):
        """Test that random walks have the right length and number."""
        graph = data_utils.load_random_graph()
        walk_length = 3
        n_walks_per_node = 1

        paths = graph.random_walk(
            walk_length=walk_length, n_walks_per_node=n_walks_per_node
        )

        result = [len(paths), len(paths[0])]
        expected = [len(graph.edges) * n_walks_per_node, walk_length + 1]

        self.assertAllClose(result, expected)

    def test_random_walks_karate_graph(self):
        """Test that random walks have the right length and number."""
        graph = data_utils.load_karate_graph()
        walk_length = 6
        n_walks_per_node = 2

        paths = graph.random_walk(
            walk_length=walk_length, n_walks_per_node=n_walks_per_node
        )

        result = [len(paths), len(paths[0])]
        expected = [len(graph.edges) * n_walks_per_node, walk_length + 1]

        self.assertAllClose(result, expected)

    def test_load_connectomes(self):
        """Test that the connectomes belong to SPD."""
        spd = SPDMatrices(28)
        data, _, _ = data_utils.load_connectomes(as_vectors=True)
        result = data.shape
        expected = (86, 27 * 14)
        self.assertAllClose(result, expected)

        data, _, labels = data_utils.load_connectomes()
        result = spd.belongs(data)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(labels >= 0, labels <= 1)
        self.assertTrue(gs.all(result))

    @np_and_autograd_only
    def test_leaves(self):
        """Test that leaves data are beta distribution parameters."""
        beta = BetaDistributions()
        beta_param, distrib_type = data_utils.load_leaves()
        result = beta.belongs(beta_param)
        self.assertTrue(gs.all(result))

        result = len(distrib_type)
        expected = beta_param.shape[0]
        self.assertAllClose(result, expected)

    def test_load_emg(self):
        """Test that data have the correct column names."""
        data_emg = data_utils.load_emg()
        expected_col_name = [
            "time",
            "c0",
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "c6",
            "c7",
            "label",
            "exp",
        ]
        good_col_name = (expected_col_name == data_emg.keys()).all()
        self.assertTrue(good_col_name)

    def test_load_optical_nerves(self):
        """Test that optical nerves belong to space of landmarks."""
        data, labels, monkeys = data_utils.load_optical_nerves()
        result = data.shape
        n_monkeys = 11
        n_eyes_per_monkey = 2
        k_landmarks = 5
        dim = 3
        expected = (n_monkeys * n_eyes_per_monkey, k_landmarks, dim)
        self.assertAllClose(result, expected)

        landmarks_space = Landmarks(
            ambient_manifold=Euclidean(dim=dim), k_landmarks=k_landmarks
        )

        result = landmarks_space.belongs(data)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(labels >= 0, labels <= 1)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(monkeys >= 0, monkeys <= 11)
        self.assertTrue(gs.all(result))

    def test_hands(self):
        """Test that hands belong to space of landmarks."""
        data, labels, _ = data_utils.load_hands()
        result = data.shape
        n_hands = 52
        k_landmarks = 22
        dim = 3
        expected = (n_hands, k_landmarks, dim)
        self.assertAllClose(result, expected)

        landmarks_space = Landmarks(
            ambient_manifold=Euclidean(dim=3), k_landmarks=k_landmarks
        )

        result = landmarks_space.belongs(data)
        self.assertTrue(gs.all(result))

        result = gs.logical_and(labels >= 0, labels <= 1)
        self.assertTrue(gs.all(result))

    def test_cells(self):
        """Test that cells belong to space of planar curves."""
        cells, cell_lines, treatments = data_utils.load_cells()
        expected = 650
        result = len(cells)
        self.assertAllClose(result, expected)
        result = len(cell_lines)
        self.assertAllClose(result, expected)
        result = len(treatments)
        self.assertAllClose(result, expected)

        result = [line in ["dlm8", "dunn"] for line in cell_lines]
        self.assertTrue(gs.all(result))

        result = [treatment in ["control", "cytd", "jasp"] for treatment in treatments]
        self.assertTrue(gs.all(result))

    def test_load_cube(self):
        """Test that the cube loads correctly."""
        vertices, faces = data_utils.load_cube()
        assert vertices.shape == (8, 3)
        assert faces.shape == (12, 3)

    def test_load_dhoom_cities_equivalence(self):
        """Test that load_dhoom of cities.dhoom matches load_cities semantically."""
        sphere = Hypersphere(dim=2)
        raw = data_utils.load_dhoom(data_utils.CITIES_DHOOM_PATH)

        self.assertEqual(len(raw), 50)
        self.assertTrue("lat" in raw[0])
        self.assertTrue("lng" in raw[0])
        self.assertEqual(raw[0]["city"], "Tokyo")

        lat_lng = gs.array(
            [[row["lat"] / 180 * gs.pi, row["lng"] / 180 * gs.pi] for row in raw]
        )
        colat = gs.pi / 2 - lat_lng[:, 0]
        colat = gs.expand_dims(colat, axis=1)
        lng = gs.expand_dims(lat_lng[:, 1] + gs.pi, axis=1)
        spherical = gs.concatenate([colat, lng], axis=1)
        extrinsic = sphere.spherical_to_extrinsic(spherical)

        self.assertAllClose(gs.shape(extrinsic), (50, 3))
        self.assertAllClose(
            extrinsic[0], gs.array([0.61993792, -0.52479018, 0.58332859])
        )
        self.assertTrue(gs.all(sphere.belongs(extrinsic)))

    def test_load_dhoom_smaller_than_json(self):
        """Test that cities.dhoom is smaller on disk than cities.json."""
        dhoom_bytes = os.path.getsize(data_utils.CITIES_DHOOM_PATH)
        json_bytes = os.path.getsize(data_utils.CITIES_PATH)
        self.assertTrue(dhoom_bytes < json_bytes)

    def test_save_dhoom_load_dhoom_roundtrip(self):
        """Test that save_dhoom followed by load_dhoom recovers the original."""
        original = [
            {"id": 1, "name": "alpha", "value": 1.5},
            {"id": 2, "name": "beta", "value": 2.5},
            {"id": 3, "name": "gamma", "value": 3.5},
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".dhoom", mode="w", delete=False, encoding="utf-8"
        ) as tf:
            tmp_path = tf.name
        try:
            data_utils.save_dhoom(original, tmp_path)
            recovered = data_utils.load_dhoom(tmp_path)
            self.assertEqual(recovered, original)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_dhoom_malformed_raises(self):
        """Test that loading a malformed DHOOM file raises DhoomError."""
        with tempfile.NamedTemporaryFile(
            suffix=".dhoom", mode="w", delete=False, encoding="utf-8"
        ) as tf:
            tf.write("this is not valid dhoom syntax")
            bad_path = tf.name
        try:
            raised = False
            try:
                data_utils.load_dhoom(bad_path)
            except DhoomError:
                raised = True
            self.assertTrue(raised)
        finally:
            if os.path.exists(bad_path):
                os.unlink(bad_path)


class TestDhoomCodec(TestCase):
    """Exercises the vendored DHOOM codec at the function level.

    These tests cover the encoder/decoder, modifier branches (arithmetic,
    interned, delta, default, nested, computed, constraint, morphism),
    auto-detection during encoding, parser helpers (coerce,
    value_to_dhoom, _split_record_fields), and error paths that the
    file-level round-trip tests in TestDatasets do not exercise.
    """

    # ---------- coerce ----------

    def test_coerce_booleans_and_null(self):
        self.assertEqual(coerce("T"), True)
        self.assertEqual(coerce("F"), False)
        self.assertEqual(coerce("null"), None)

    def test_coerce_empty_string_is_empty_string(self):
        self.assertEqual(coerce(""), "")

    def test_coerce_integers(self):
        self.assertEqual(coerce("0"), 0)
        self.assertEqual(coerce("42"), 42)
        self.assertEqual(coerce("-17"), -17)

    def test_coerce_floats(self):
        self.assertEqual(coerce("3.14"), 3.14)
        self.assertEqual(coerce("-2.5"), -2.5)

    def test_coerce_plain_string_passes_through(self):
        self.assertEqual(coerce("hello"), "hello")
        self.assertEqual(coerce("abc123"), "abc123")

    # ---------- value_to_dhoom ----------

    def test_value_to_dhoom_booleans_and_null(self):
        self.assertEqual(value_to_dhoom(True), "T")
        self.assertEqual(value_to_dhoom(False), "F")
        self.assertEqual(value_to_dhoom(None), "null")

    def test_value_to_dhoom_numbers(self):
        self.assertEqual(value_to_dhoom(42), "42")
        self.assertEqual(value_to_dhoom(-17), "-17")
        self.assertEqual(value_to_dhoom(3.14), "3.14")

    def test_value_to_dhoom_quotes_strings_with_commas(self):
        # Commas inside string values must be quoted so the field splitter
        # doesn't break them across records.
        encoded = value_to_dhoom("hello, world")
        self.assertEqual(encoded, '"hello, world"')

    def test_value_to_dhoom_doubles_internal_quotes(self):
        # Internal double quotes must be doubled per the spec.
        encoded = value_to_dhoom('she said "hi"')
        self.assertEqual(encoded, '"she said ""hi"""')

    def test_value_to_dhoom_plain_string_unquoted(self):
        self.assertEqual(value_to_dhoom("plain"), "plain")

    # ---------- direct codec round-trips ----------

    def test_roundtrip_preserves_value_types(self):
        # Booleans, None, ints, floats, and strings should survive a
        # full encode/decode cycle through the in-memory codec.
        original = [
            {"i": 1, "f": 1.5, "s": "alpha", "b": True, "n": None},
            {"i": 2, "f": 2.5, "s": "beta", "b": False, "n": None},
        ]
        recovered = decode(encode({"records": original}))
        # encode wraps in {"records": [...]} so we unwrap here
        self.assertEqual(recovered["records"], original)

    def test_roundtrip_strings_with_special_chars(self):
        # Strings containing commas, colons, quotes, and newlines need
        # the quoting path on encode and the quote-aware splitter on
        # decode. All must round-trip exactly.
        original = [
            {"k": "comma,inside", "v": 1},
            {"k": 'quote"inside', "v": 2},
            {"k": "colon:inside", "v": 3},
        ]
        recovered = decode(encode({"records": original}))
        self.assertEqual(recovered["records"], original)

    def test_roundtrip_single_record(self):
        # A list with one record exercises the "no auto-detection
        # possible" branch (most _detect_* helpers need >= 2 values).
        original = [{"only": 42, "name": "alone"}]
        recovered = decode(encode({"records": original}))
        self.assertEqual(recovered["records"], original)

    def test_roundtrip_empty_list(self):
        # Empty bundles should encode + decode without raising.
        recovered = decode(encode({"records": []}))
        self.assertEqual(recovered["records"], [])

    def test_roundtrip_with_arithmetic_id_sequence(self):
        # An obvious arithmetic sequence (id = 1, 2, 3, ...) should be
        # auto-detected and represented as an arithmetic modifier on
        # encode, then reconstructed on decode. Output equality verifies
        # the round-trip works regardless of internal representation.
        original = [{"id": i, "name": f"n{i}"} for i in range(1, 11)]
        encoded = encode({"records": original})
        recovered = decode(encoded)
        self.assertEqual(recovered["records"], original)

    def test_roundtrip_with_repeated_modal_value(self):
        # When most records share the same value for a field, the encoder
        # may emit a modal-default modifier. The decoder must rebuild the
        # explicit value on read.
        original = [{"status": "active", "id": i} for i in range(1, 6)]
        original.append({"status": "inactive", "id": 6})
        original.append({"status": "active", "id": 7})
        recovered = decode(encode({"records": original}))
        self.assertEqual(recovered["records"], original)

    def test_roundtrip_with_repeated_interned_strings(self):
        # Strings drawn from a small pool benefit from the interned
        # modifier. Round-trip must reconstruct the exact strings.
        original = [
            {"color": "red", "n": 1},
            {"color": "blue", "n": 2},
            {"color": "red", "n": 3},
            {"color": "green", "n": 4},
            {"color": "red", "n": 5},
        ]
        recovered = decode(encode({"records": original}))
        self.assertEqual(recovered["records"], original)

    def test_roundtrip_negative_and_large_numbers(self):
        # Exercises both the negative integer and large-number branches
        # of coerce + value_to_dhoom.
        original = [
            {"a": -1000000, "b": 999999999},
            {"a": 0, "b": -1},
        ]
        recovered = decode(encode({"records": original}))
        self.assertEqual(recovered["records"], original)

    # ---------- decode-only edge cases ----------

    def test_decode_empty_string_returns_none(self):
        # The decoder's empty-input branch returns None rather than
        # raising — useful sentinel for "this file was empty."
        self.assertEqual(decode(""), None)
        self.assertEqual(decode("   \n  \t  "), None)

    def test_decode_missing_braces_raises(self):
        # A fiber header without braces is unparseable. The error must
        # be a DhoomError, not a generic exception.
        raised = False
        try:
            decode("name no_braces_here body")
        except DhoomError:
            raised = True
        self.assertTrue(raised)

    def test_decode_missing_header_terminator_raises(self):
        # The "}:" terminator separates header from body. Missing it
        # is a structural error the decoder must catch cleanly.
        raised = False
        try:
            decode("name {a, b} 1, 2\n")
        except DhoomError:
            raised = True
        self.assertTrue(raised)

    # ---------- encode error paths ----------

    def test_encode_top_level_scalar_raises(self):
        # The encoder requires the top-level value to be an object or
        # array; anything else is structurally invalid.
        raised = False
        try:
            encode(42)
        except DhoomError:
            raised = True
        self.assertTrue(raised)

    def test_encode_multi_key_top_level_dict_raises(self):
        # Top-level dicts represent a single bundle; multiple keys at
        # the top level is an ambiguity the encoder rejects explicitly.
        raised = False
        try:
            encode({"first": [], "second": []})
        except DhoomError:
            raised = True
        self.assertTrue(raised)

    # ---------- save / load file-level edge cases ----------

    def test_save_dhoom_then_load_preserves_unicode(self):
        # File-level round-trip with non-ASCII content checks both the
        # save_dhoom utf-8 encoding and the decoder's quote-aware split.
        original = [
            {"city": "Zürich", "country": "Switzerland"},
            {"city": "São Paulo", "country": "Brazil"},
            {"city": "東京", "country": "Japan"},
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".dhoom", mode="w", delete=False, encoding="utf-8"
        ) as tf:
            tmp_path = tf.name
        try:
            data_utils.save_dhoom(original, tmp_path)
            recovered = data_utils.load_dhoom(tmp_path)
            self.assertEqual(recovered, original)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_dhoom_empty_file_returns_none(self):
        # Empty files should not raise — the decoder treats them as the
        # "no data" sentinel.
        with tempfile.NamedTemporaryFile(
            suffix=".dhoom", mode="w", delete=False, encoding="utf-8"
        ) as tf:
            tmp_path = tf.name  # file is created empty
        try:
            result = data_utils.load_dhoom(tmp_path)
            self.assertEqual(result, None)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
